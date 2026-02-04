from typing import Dict, Any, Tuple, List
import torch
import numpy as np
from PIL import Image
import comfy.model_management
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC

from ..redrocket.classifier_v3 import JtpInferenceV3
from ..redrocket.model_v3_manager import JtpModelV3Manager
from ..redrocket.tag_v3_manager import JtpTagV3Manager
from ..redrocket.image_v3_manager import JtpImageV3Manager
from ..helpers.config import ComfyExtensionConfig
from .rrtagger import download_progress_callback, download_complete_callback
import os
import folder_paths

class Jtp3HydraTagger(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Models list from config or manager
        # We need to initialize manager to list installed, or check config
        # Ideally we list installed + available in config? 
        # JtpModelV3Manager list_installed only lists files.
        # We should use config keys for dropdown?
        config = ComfyExtensionConfig().get()
        models = list(config.get("models_v3", {}).keys())
        if not models:
            models = ["jtp-3-hydra"] # Default fallback

        return {"required": {
            "image": (IO.IMAGE, ),
            "model": (models, {"default": models[0] if models else "jtp-3-hydra"}),
            "threshold": (IO.FLOAT, {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            "cam_depth": (IO.INT, {"default": 1, "min": 1, "max": 27, "step": 1, "display": "slider"}),
            "seqlen": (IO.INT, {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            "implications_mode": (["inherit", "constrain", "remove", "constrain-remove", "off"], {"default": "inherit"}),
            "exclude_categories": (IO.STRING, {"multiline": True, "tooltip": "Comma separated categories (e.g. artist, character, species, copyright, meta, lore). Requires tag metadata."}),
            "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES: Tuple[str] = (IO.STRING, IO.STRING, IO.IMAGE)
    RETURN_NAMES: Tuple[str] = ("tags", "scores", "attention_map")
    OUTPUT_IS_LIST: Tuple[bool] = (True, True, True)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "🐺 Furry Diffusion"

    def tag(self, 
            image: Image.Image, 
            model: str, 
            threshold: float, 
            cam_depth: int, 
            seqlen: int, 
            implications_mode: str, 
            seed: int,
            exclude_categories: str = "") -> Dict[str, Any]:
        
        device = comfy.model_management.get_torch_device()
        torch.manual_seed(seed)
        
        # Parse exclude categories
        categories_list = [c.strip() for c in exclude_categories.split(",") if c.strip()]
        
        tensor: np.ndarray = image * 255
        tensor = np.array(tensor, dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        
        tags_list: List[str] = []
        scores_list: List[Dict[str, float]] = []
        cams_list: List[torch.Tensor] = []
        
        for i in range(tensor.shape[0]):
            img: Image.Image = Image.fromarray(tensor[i]).convert("RGBA")
            
            tags_str, scores, cam_img = JtpInferenceV3.run_classifier(
                model_name=model,
                device=device,
                image=img,
                threshold=threshold,
                cam_depth=cam_depth,
                seqlen=seqlen,
                implications_mode=implications_mode,
                exclude_categories=categories_list,
                seed=seed
            )
            
            tags_list.append(tags_str)
            scores_list.append(scores)
            
            if cam_img:
                # Convert PIL to Tensor (HWC -> BHWC logic usually handled by Comfy, but output IMAGE expects float BHWC)
                cam_np = np.array(cam_img).astype(np.float32) / 255.0
                cam_tensor = torch.from_numpy(cam_np).unsqueeze(0) # BHWC
                cams_list.append(cam_tensor)
            else:
                # Empty black image if no CAM (shouldn't happen if logic allows, or just skip)
                # If we return a list, lengths must match?
                # Usually yes. Let's return a blank tensor of image size.
                blank = torch.zeros((1, img.height, img.width, 3), dtype=torch.float32)
                cams_list.append(blank)
                
            pbar.update(1)

        # Result formatting
        # tags: list of strings
        # scores: list of dicts (JSON stringified? Or just object? ComfyUI usually handles it if return type matches)
        # But scores is STRING output type in definition. We should probably JSON dump it or return repr?
        # Original node returns Dict[str, float] but type says IO.STRING.
        # Looking at original code: `return {"ui": {"tags": tags, "scores": scores}, "result": (tags, scores,)}`
        # `scores` is `List[Dict[str, float]]`.
        # So it returns the object directly to backend? ComfyUI might stringify it for frontend.
        # But for connection to other nodes, STRING type implies a string.
        # The original node had `OUTPUT_IS_LIST: Tuple[bool] = (True, True,)`.
        # So it returns a list of strings and a list of... dicts?
        # If specific nodes consume `scores`, they might expect dicts.
        # I'll keep it consistent with the original node logic.
        
        # Note on CAM output: OUTPUT_IS_LIST for IMAGE? 
        # ComfyUI handles list of tensors as batch?
        # If OUTPUT_IS_LIST is True for IMAGE, it expects a list of tensors [1, H, W, C].
        # Or a single tensor [B, H, W, C]?
        # If output is list, it processes batch element by element?
        # Usually IMAGE output is a single tensor [B, H, W, C].
        # But here we process inputs one by one.
        # If we use OUTPUT_IS_LIST = (True, True, False) for (tags, scores, cam), 
        # then we should return (tags_list, scores_list, torch.cat(cams_list, dim=0)).
        # Let's try to batch the CAMs into one tensor.
        
        if cams_list:
            final_cam = torch.cat(cams_list, dim=0) # [B, H, W, C]
        else:
            final_cam = torch.zeros((1, 64, 64, 3)) # Dummy
            
        # Return format: "result": (list, list, tensor)
        # But if OUTPUT_IS_LIST is True for tags, scores...
        # And False for CAM (since we combined it)?
        # Or True for CAM and return list of tensors?
        # Standard Comfy behavior: IMAGE is usually a batch tensor.
        # So I will set OUTPUT_IS_LIST = (True, True, False)
        
        return {
            "ui": {"tags": tags_list, "scores": scores_list}, 
            "result": (tags_list, scores_list, final_cam)
        }

    OUTPUT_IS_LIST: Tuple[bool] = (True, True, False)

model_basepath = os.path.join(folder_paths.models_dir, "RedRocket")
tags_basepath = os.path.join(model_basepath, "tags")

JtpModelV3Manager(model_basepath=model_basepath, download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpTagV3Manager(tags_basepath=tags_basepath, download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpImageV3Manager()
JtpInferenceV3(device=comfy.model_management.get_torch_device())
