from pathlib import Path
from PIL import Image
import numpy as np
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
import hashlib

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
        exclude_categories: List[str],
        seed: int
    ) -> Tuple[str, Dict[str, float], Optional[Image.Image]]:
        
        # Hash params for caching
        params_string = f"{model_name}|{threshold}|{cam_depth}|{seqlen}|{implications_mode}|{exclude_categories}|{seed}"
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
        tag_str, final_scores = JtpTagV3Manager.process_tags(
            predictions, model_name, threshold, implications_mode, exclude_categories
        )
        
        cam_image = None
        
        # Cache result
        image_key = None
        if isinstance(image, Path): image_key = str(image)
        elif isinstance(image, np.ndarray): image_key = hashlib.sha256(image.tobytes()).hexdigest()
        elif isinstance(image, Image.Image): 
             img_arr = np.array(image.convert("RGB"))
             image_key = hashlib.sha256(img_arr.tobytes()).hexdigest()
             
        JtpImageV3Manager.commit_cache(image_key, (tag_str, final_scores, cam_image), params_key)
        
        return tag_str, final_scores, cam_image
