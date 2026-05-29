import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, Tuple, List
import os
import folder_paths
import comfy.model_management as mm
import comfy.utils

from comfy.comfy_types import IO, ComfyNodeABC
from ..helpers.config import ComfyExtensionConfig
from ..redrocket.model_furrence2_manager import Furrence2ModelManager
from ..redrocket.cap_furrence2_manager import Furrence2CaptionManager

class RRFurrence2Captioner(ComfyNodeABC):
    def __init__(self) -> None:
        super().__init__()
        # Initialize singletons with base paths mirroring v3/dino structures
        model_basepath = os.path.join(folder_paths.models_dir, "RedRocket")
        models_dir = os.path.join(model_basepath, "furrence2")
        Furrence2ModelManager(model_basepath=models_dir)
        
        tags_dir = os.path.join(model_basepath, "tags")
        Furrence2CaptionManager(tags_basepath=tags_dir)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        config = ComfyExtensionConfig().get()
        models = list(config.get("models_furrence", {}).keys())
        if not models:
            models = ["furrence2-large"]

        return {
            "required": {
                "image": (IO.IMAGE, ),
                "tags": (IO.STRING, {"multiline": True, "forceInput": True}),
                "model": (models, {"default": models[0] if models else "furrence2-large"}),
                "expected_length": (IO.INT, {"default": 100, "min": 50, "max": 500, "step": 10}),
                "attn_mode": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
                "seq_len": (IO.INT, {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "num_beams": (IO.INT, {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES: Tuple[str] = (IO.STRING,)
    RETURN_NAMES: Tuple[str] = ("caption",)
    OUTPUT_IS_LIST: Tuple[bool] = (True,)
    FUNCTION: str = "caption"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "🐺 Furry Diffusion"

    def caption(self, image: torch.Tensor, tags: str, model: str, expected_length: int, attn_mode: str, seq_len: int, num_beams: int) -> Dict[str, Any]:
        device = mm.get_torch_device()
        
        # 1. Load Model
        if not Furrence2ModelManager.is_loaded(model):
            success = Furrence2ModelManager.load(model, device=device)
            if not success:
                return {"ui": {"caption": ["Error: Model failed to load"]}, "result": (["Error: Model failed to load"],)}

        # 2. Load Tag Constraints - Hardcoded to the specific version required by Furrence-2
        tags_csv_name = "tags-2024-05-05"
        if not Furrence2CaptionManager.is_loaded(tags_csv_name):
            success = Furrence2CaptionManager.load(tags_csv_name)
            if not success:
                return {"ui": {"caption": ["Error: Tags CSV failed to load"]}, "result": (["Error: Tags CSV failed to load"],)}

        # 3. Format Prompt
        prompt = Furrence2CaptionManager.format_prompt(tags_csv_name, tags, expected_length)
        
        # 4. Run Inference
        pipeline = Furrence2ModelManager.get_model(model)
        
        captions: List[str] = []
        # image is [B, H, W, C] float32 in [0, 1]
        tensor: np.ndarray = (image * 255).cpu().numpy().astype(np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        
        for i in range(tensor.shape[0]):
            pil_img = Image.fromarray(tensor[i]).convert("RGB")
            
            # The custom pipeline handles the pre-processing, encoding, generation, and post-processing
            res_text = pipeline(
                image=pil_img,
                prompt=prompt,
                attn_implementation=attn_mode,
                max_new_tokens=seq_len,
                num_beams=num_beams
            )
            
            captions.append(res_text)
            pbar.update(1)

        return {"ui": {"caption": captions}, "result": (captions,)}

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "RRFurrence2Captioner": RRFurrence2Captioner,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "RRFurrence2Captioner": "RedRocket Furrence-2 Captioner",
}
