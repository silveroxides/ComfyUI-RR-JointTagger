from pathlib import Path
from PIL import Image
import numpy as np
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
import hashlib
import threading
from torch.nn import Parameter

from .image_v3_manager import JtpImageV3Manager
from .tag_v3_manager import JtpTagV3Manager
from .model_v3_manager import JtpModelV3Manager
from ..helpers.cache import ComfyCache
from ..helpers.metaclasses import Singleton
from ..helpers.logger import ComfyLogger

class JtpInferenceV3(metaclass=Singleton):
    """
    Inference for JTP-3 Hydra.
    """
    def __init__(self, device: Optional[torch.device] = torch.device('cpu')) -> None:
        torch.set_grad_enabled(False)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_lock = threading.Lock()

    @classmethod
    def run_cam(cls, model, display_image, features, tag_idx, cam_depth, patch_coords, patch_valid):
        """
        Runs Class Activation Mapping (CAM) for a specific tag.
        """
        intermediates = features.get("image_intermediates")
        if intermediates is None:
            intermediates = features.get("intermediates")
            
        if intermediates is None:
             raise KeyError(f"Features dict missing 'image_intermediates' or 'intermediates'. Keys: {list(features.keys())}")

        # Handle cam_depth logic relative to available intermediates
        # JTP-3 app.py logic:
        if len(intermediates) > cam_depth:
            intermediates = intermediates[-cam_depth:]
        
        # We need to use the instance lock
        instance = cls()
        
        with instance.model_lock:
            saved_q = model.attn_pool.q
            saved_p = model.attn_pool.out_proj.weight

            try:
                # Slice weights for the specific tag
                model.attn_pool.q = Parameter(saved_q[:, [tag_idx], :], requires_grad=False)
                model.attn_pool.out_proj.weight = Parameter(saved_p[[tag_idx], :, :], requires_grad=False)

                with torch.enable_grad():
                    for intermediate in intermediates:
                        intermediate.requires_grad_(True).retain_grad()
                        # Forward pass through the head with sliced weights
                        # Note: forward_head usually takes (x, pre_logits=False).
                        # JTP-3 model.py forward_head implementation details?
                        # In JTP-3 app.py: model.forward_head(intermediate, patch_valid=patch_valid)[0, 0].backward()
                        model.forward_head(intermediate, patch_valid=patch_valid)[0, 0].backward()
            finally:
                model.attn_pool.q = saved_q
                model.attn_pool.out_proj.weight = saved_p

        cam_1d = None
        for intermediate in intermediates:
            if intermediate.grad is None: continue
            patch_grad = (intermediate.grad.float() * intermediate.sign()).sum(dim=(0, 2))
            intermediate.grad = None # Clear grad to free memory

            if cam_1d is None:
                cam_1d = patch_grad
            else:
                cam_1d.add_(patch_grad)

        if cam_1d is None:
            return display_image # Should not happen

        cam_2d = JtpImageV3Manager.unpatchify(cam_1d, patch_coords, patch_valid).cpu().numpy()
        return JtpImageV3Manager.cam_composite(display_image, cam_2d)

    @classmethod
    def run_classifier(
        cls,
        model_name: str,
        device: torch.device,
        image: Union[Image.Image, np.ndarray, Path],
        threshold: float,
        cam_depth: int,
        seqlen: int,
        implications_mode: str,
        exclude_tags: str,
        exclude_categories: str,
        prefix: str,
        original_tags: bool,
        seed: int,
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        cam_mode: str = "auto",
        cam_tag: str = ""
    ) -> Tuple[str, Dict[str, float], Optional[Image.Image]]:
        
        # Hash params for caching
        params_string = f"{model_name}|{threshold}|{cam_depth}|{seqlen}|{implications_mode}|{exclude_tags}|{exclude_categories}|{prefix}|{original_tags}|{seed}|{replace_underscore}|{trailing_comma}|{cam_mode}|{cam_tag}"
        params_key = hashlib.sha256(params_string.encode()).hexdigest()
        
        # Load Model
        if JtpModelV3Manager().is_installed(model_name) is False:
            if not JtpModelV3Manager().download(model_name):
                return "", {}, None
        
        if JtpTagV3Manager.is_loaded(model_name) is False:
             JtpTagV3Manager.download(model_name)
             if not JtpTagV3Manager.load(model_name):
                 ComfyLogger().log("Failed to load tags", "ERROR", True)
                 return "", {}, None

        loaded, tags = JtpModelV3Manager.load(model_name, device=device)
        if not loaded:
            return "", {}, None
            
        model = ComfyCache.get(f'model_v3.{model_name}.model')
        
        # Load/Process Image
        result = JtpImageV3Manager.load(image, device=device, seqlen=seqlen, params_key=params_key)
        
        if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], Dict): # Cached output (tags, scores, cam)
             return result[0], result[1], result[2]
        
        if result is None:
             return "", {}, None
             
        # It's tensors: patches, coords, valid
        patches, coords, valid = result
        
        # Prepare inputs
        p_d = patches.unsqueeze(0).to(dtype=torch.bfloat16).div_(127.5).sub_(1.0)
        pc_d = coords.unsqueeze(0).to(dtype=torch.int32)
        pv_d = valid.unsqueeze(0)
        
        # Inference
        # We need gradients for CAM if we are going to run it?
        # No, JTP-3 app.py runs forward with no_grad, then enables grad only for the backward pass on intermediates.
        
        with torch.no_grad():
            features = model.forward_intermediates(
                p_d,
                patch_coord=pc_d,
                patch_valid=pv_d,
                indices=cam_depth,
                output_dict=True,
                output_fmt='NLC'
            )
            logits = model.forward_head(features["image_features"], patch_valid=pv_d)
            
            probits = torch.sigmoid(logits[0].to(dtype=torch.float32))
            probits.mul_(2.0).sub_(1.0) # Scale to -1..1
            
            values, indices = probits.cpu().topk(250)
            
            predictions = {
                tags[idx.item()]: val.item()
                for idx, val in sorted(
                    zip(indices, values),
                    key=lambda item: item[1].item(),
                    reverse=True
                )
            }

        # Process tags (Implications, Thresholding)
        tag_str, final_scores, top_tag_raw = JtpTagV3Manager.process_tags(
            predictions, model_name, threshold, implications_mode, exclude_tags,
            exclude_categories, prefix, original_tags,
            replace_underscore=replace_underscore, trailing_comma=trailing_comma
        )
        
        cam_image = None
        
        # CAM Logic
        if cam_mode != "none":
            target_idx = -1
            
            if cam_mode == "auto":
                # Use top 1 valid tag from filtered results
                if top_tag_raw and top_tag_raw in tags:
                    try:
                        target_idx = tags.index(top_tag_raw)
                    except ValueError:
                        ComfyLogger().log(f"Auto CAM: '{top_tag_raw}' found in tags list but index retrieval failed.", "WARNING")
                
                # Fallback to raw model top 1 if filtering removed everything but auto was requested?
                # Or if process_tags returned nothing.
                if target_idx == -1 and indices.numel() > 0:
                     # If we have no valid tags after filtering, should we show the raw top tag?
                     # The user likely wants to see *why* it triggered or just the strongest signal.
                     # But if they excluded it, maybe they don't want to see it.
                     # Let's fallback to raw top 1 to avoid black square confusion,
                     # but ideally 'top_tag_raw' covers the "best valid" case.
                     target_idx = indices[0].item()

            elif cam_mode == "specific_tag" and cam_tag:
                # Find tag index
                # Clean up input (remove trailing commas, whitespace)
                clean_tag = cam_tag.strip().rstrip(",")
                
                try:
                    # Normalized check
                    if clean_tag in tags:
                        target_idx = tags.index(clean_tag)
                    else:
                        # Try replacing spaces with underscores
                        cam_tag_norm = clean_tag.replace(" ", "_")
                        if cam_tag_norm in tags:
                            target_idx = tags.index(cam_tag_norm)
                        else:
                            ComfyLogger().log(f"CAM Tag '{cam_tag}' (normalized: '{cam_tag_norm}') not found in model tags.", "WARNING")
                except ValueError:
                    ComfyLogger().log(f"CAM Tag '{cam_tag}' caused a ValueError during lookup.", "WARNING")
            
            if target_idx >= 0:
                # Need original PIL image for composite
                # If 'image' is PIL, use it. If tensor/numpy, we need to convert or use 'result' input?
                # JtpImageV3Manager.load processes image.
                # We can use the original 'image' if it's PIL.
                display_img = None
                if isinstance(image, Image.Image):
                    display_img = image.convert("RGB")
                elif isinstance(image, np.ndarray):
                    display_img = Image.fromarray(image).convert("RGB")
                elif isinstance(image, Path) and os.path.exists(str(image)):
                    display_img = Image.open(str(image)).convert("RGB")
                
                if display_img:
                    try:
                        cam_image = cls.run_cam(
                            model, display_img, features, target_idx, cam_depth,
                            pc_d, pv_d
                        )
                    except Exception as e:
                        import traceback
                        ComfyLogger().log(f"CAM generation failed: {e}\n{traceback.format_exc()}", "ERROR", True)

        # Cache result
        image_key = None
        if isinstance(image, Path): image_key = str(image)
        elif isinstance(image, np.ndarray): image_key = hashlib.sha256(image.tobytes()).hexdigest()
        elif isinstance(image, Image.Image): 
             img_arr = np.array(image.convert("RGB"))
             image_key = hashlib.sha256(img_arr.tobytes()).hexdigest()
             
        JtpImageV3Manager.commit_cache(image_key, (tag_str, final_scores, cam_image), params_key)
        
        return tag_str, final_scores, cam_image
