from pathlib import Path
from PIL import Image
import numpy as np
import torch
from typing import Optional, Tuple, Union, Dict, Any

from .image_manager import JtpImageManager

from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager
from ..helpers.cache import ComfyCache
from ..helpers.metaclasses import Singleton

class JtpInference(metaclass=Singleton):
    """
    A Clip Vision Classifier by RedRocket (inference code made robust by deitydurg)
    """
    def __init__(self, device: Optional[torch.device] = torch.device('cpu')) -> None:
        torch.set_grad_enabled(False)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def run_classifier(cls, model_name: str, tags_name: str, device: torch.device, image: Union[Image.Image, np.ndarray, Path], steps: float, threshold: float, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False) -> Tuple[str, Dict[str, float]]:
        from ..helpers.logger import ComfyLogger
        
        model_version: int = ComfyCache.get(f'config.models.{model_name}.version')
        tags_version: int = ComfyCache.get(f'config.tags.{tags_name}.version')
        ComfyLogger().log(message=f"Model version: {model_version}, Tags version: {tags_version}", type="DEBUG", always=True)

        # Load all the things
        if JtpModelManager().is_installed(model_name) is False:
            if not JtpModelManager().download(model_name):
                ComfyLogger().log(f"Model {model_name} could not be downloaded", "ERROR", True)
                return "", {}
        if JtpTagManager().is_installed(tags_name) is False:
            if not JtpTagManager().download(tags_name):
                ComfyLogger().log(f"Tags {tags_name} could not be downloaded", "ERROR", True)
                return "", {}            
        if JtpModelManager().is_loaded(model_name) is False:
            if not JtpModelManager().load(model_name=model_name, version=model_version, device=device):
                ComfyLogger().log(f"Model {model_name} could not be loaded", "ERROR", True)
                return "", {}
        if JtpTagManager().is_loaded(tags_name) is False:
            if not JtpTagManager().load(tags_name=tags_name, version=tags_version):
                ComfyLogger().log(f"Tags {tags_name} could not be loaded", "ERROR", True)
                return "", {}
        # if cls().device != device:
        #     JtpModelManager().switch_device(model_name, device)
        #     cls().device = device
        tensor = JtpImageManager().load(image=image, device=device)
        if tensor is None:
            ComfyLogger().log(message="Image could not be loaded", type="ERROR", always=True)
            return "", {}
        if isinstance(tensor, tuple):
            ComfyLogger().log(message="Returning cached result", type="DEBUG", always=True)
            return tensor[0], tensor[1]
        ComfyLogger().log(message=f"Classifying image with model {model_name} and tags {tags_name}", type="INFO", always=True)
        model_data: Union[torch.nn.Module, None] = ComfyCache.get(f'model.{model_name}.model')
        if model_data is None:
            ComfyLogger().log(message=f"Model data for {model_name} not found in cache", type="ERROR", always=True)
            return "", {}
        with torch.no_grad():
            if f"{model_version}" == "1":
                logits = model_data(tensor)
                probits = torch.nn.functional.sigmoid(logits[0]).cpu()
                values, indices = probits.topk(250)
            elif f"{model_version}" == "2":
                probits = model_data(tensor)[0].cpu()
                values, indices = probits.topk(250)
            else:
                ComfyLogger().log(message=f"Model version {model_version} not supported", type="ERROR", always=True)
                return "", {}
        ComfyLogger().log(message="Processing tags...", type="DEBUG", always=True)
        tags_str, tag_scores = JtpTagManager().process_tags(tags_name=tags_name, values=values, indices=indices, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
        if not JtpImageManager().commit_cache(image=image, output=(tags_str, tag_scores)):
            ComfyLogger().log(message="Image cache could not be committed", type="WARN", always=True)
            return tags_str, tag_scores
        ComfyLogger().log(message=f"Classification complete: {tags_str}", type="INFO", always=True)
        return tags_str, tag_scores
