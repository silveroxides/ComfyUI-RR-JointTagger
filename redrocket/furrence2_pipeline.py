import os
import torch
import torch.nn as nn
from typing import Union, Any, Optional, Dict
from PIL import Image
import numpy as np

from .modeling_florence2 import Florence2ForConditionalGeneration
from .processing_florence2 import Florence2Processor
from .configuration_florence2 import Florence2Config

class Furrence2Pipeline(nn.Module):
    """
    Custom pipeline for Furrence-2-Large that bypasses the transformers .generate() mixin.
    This ensures we have a fully decoupled implementation.
    """
    def __init__(self, model_dir: str, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        self.config = Florence2Config.from_json_file(config_path)
        
        # Initialize model
        # Note: We use the local Florence2ForConditionalGeneration which no longer inherits from GenerationMixin
        self.model = Florence2ForConditionalGeneration(self.config)
        
        # Load processor
        self.processor = Florence2Processor.from_pretrained(model_dir)
        
        # Load weights
        self.load_custom_state_dict()
        
        self.model.to(self.device)
        self.model.eval()

        # Convenience accessors
        self.language_model = self.model.language_model
        self.vision_model = self.model.vision_tower

    def load_custom_state_dict(self):
        """Loads weights from model.safetensors or pytorch_model.bin"""
        import glob
        from safetensors.torch import load_file
        
        sf_path = os.path.join(self.model_dir, "model.safetensors")
        bin_path = os.path.join(self.model_dir, "pytorch_model.bin")
        
        if os.path.exists(sf_path):
            state_dict = load_file(sf_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            # Try to find any safetensors or bin file
            files = glob.glob(os.path.join(self.model_dir, "*.safetensors")) + glob.glob(os.path.join(self.model_dir, "*.bin"))
            if not files:
                raise FileNotFoundError(f"Could not find model weights in {self.model_dir}")
            
            if files[0].endswith(".safetensors"):
                state_dict = load_file(files[0], device="cpu")
            else:
                state_dict = torch.load(files[0], map_location="cpu")

        # Florence-2 weights often have 'base_model.' or other prefixes depending on how they were saved
        # Our local classes expect names exactly as defined in __init__
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Furrence2 weights load result: {msg}")

    def _encode_image(self, pixel_values):
        return self.model._encode_image(pixel_values)

    @torch.no_grad()
    def greedy_decode(
        self, 
        input_ids, 
        image_features, 
        max_length=1024, 
        pad_token_id=1, 
        eos_token_id=2, 
        decoder_start_token_id=2
    ):
        """
        Custom greedy decoding implementation to bypass transformers .generate().
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 1. Encode image features into language space using projection
        # image_features are from Davit, we need to pass them through the projection layers
        # In Florence2ForConditionalGeneration.forward, this is handled by _merge_input_ids_with_image_features
        # but we want to be explicit here.
        
        # Prepare inputs for the language model
        # We need the full merged embeddings for the encoder part of the language model
        inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(
            input_ids,
            image_features,
            None # attention_mask
        )
        
        # Run the encoder once
        encoder_outputs = self.language_model.get_encoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 2. Iterative Decoding
        decoder_input_ids = torch.ones((batch_size, 1), device=device, dtype=torch.long) * decoder_start_token_id
        past_key_values = None
        
        generated_tokens = []
        
        for i in range(max_length):
            # Run language model forward (decoder only if we have past_key_values)
            outputs = self.language_model(
                input_ids=None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            logits = self.language_model.lm_head(outputs.last_hidden_state[:, -1, :])
            past_key_values = outputs.past_key_values
            
            # Greedy select
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            # Update decoder_input_ids for next step (if not using cache we'd append, but with cache we just pass last)
            decoder_input_ids = next_token
            
            if next_token.item() == eos_token_id:
                break
                
        return torch.cat(generated_tokens, dim=1)

    def __call__(self, image: Union[torch.Tensor, Any], prompt: str, attn_implementation="sdpa", **kwargs):
        """
        Main entry point for the pipeline.
        """
        if isinstance(image, torch.Tensor):
            # Convert ComfyUI/Torch tensor [B, H, W, C] or [C, H, W] to PIL
            if image.ndim == 4:
                image = image[0]
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        else:
            pil_image = image

        # Preprocess
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        
        # Get image features from Davit
        image_features = self._encode_image(pixel_values)
        
        # Custom Decode
        generated_ids = self.greedy_decode(
            input_ids=input_ids,
            image_features=image_features,
            max_length=kwargs.get("max_new_tokens", 1024),
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            decoder_start_token_id=self.language_model.config.decoder_start_token_id
        )
        
        # Decode text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process (clean up florence specific tags)
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(pil_image.width, pil_image.height)
        )
        
        return parsed_answer
