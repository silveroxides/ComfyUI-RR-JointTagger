"""ComfyUI node for the DINOv3 Tagger."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
import comfy.model_management
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC

from ..redrocket.classifier_dino import DINOv3Inference
from ..redrocket.model_dino_manager import DINOv3ModelManager
from ..redrocket.tag_dino_manager import DINOv3TagManager
from ..redrocket.image_dino_manager import DINOv3ImageManager
from ..helpers.config import ComfyExtensionConfig
from .rrtagger import download_progress_callback, download_complete_callback
import folder_paths


class DINOv3Tagger(ComfyNodeABC):
    """DINOv3 ViT-H/16+ image tagger node for ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        config = ComfyExtensionConfig().get()
        models = list(config.get("models_dino", {}).keys())
        if not models:
            models = ["tagger-proto"]

        return {"required": {
            "image": (IO.IMAGE,),
            "model": (models, {"default": models[0] if models else "tagger-proto"}),
            "mode": (["topk", "threshold"], {"default": "topk", "tooltip": "Tag selection mode. 'topk' returns the top K tags; 'threshold' returns all tags above a score."}),
            "topk": (IO.INT, {"default": 40, "min": 1, "max": 500, "step": 1, "display": "slider", "tooltip": "Number of top tags to return (when mode is 'topk')."}),
            "threshold": (IO.FLOAT, {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01, "display": "slider", "tooltip": "Minimum score to include a tag (when mode is 'threshold')."}),
            "max_size": (IO.INT, {"default": 1024, "min": 64, "max": 4096, "step": 16, "tooltip": "Long-edge pixel cap. Image is resized to fit, snapped to 16px multiples."}),
            "exclude_tags": (IO.STRING, {"multiline": True, "tooltip": "Comma-separated tags to exclude from output."}),
            "replace_underscore": (IO.BOOLEAN, {"default": True, "tooltip": "Replace underscores with spaces in tags."}),
            "trailing_comma": (IO.BOOLEAN, {"default": False, "tooltip": "Add a trailing comma to the tag string."}),
            "prefix": (IO.STRING, {"default": "", "tooltip": "Text to prepend to the tags output."}),
            "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for deterministic execution."}),
        }}

    RETURN_TYPES: Tuple[str, ...] = (IO.STRING, IO.STRING)
    RETURN_NAMES: Tuple[str, ...] = ("tags", "scores")
    OUTPUT_IS_LIST: Tuple[bool, ...] = (True, True)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "🐺 Furry Diffusion"

    def tag(
        self,
        image: torch.Tensor,
        model: str,
        mode: str,
        topk: int,
        threshold: float,
        max_size: int,
        seed: int,
        exclude_tags: str = "",
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        prefix: str = "",
    ) -> Dict[str, Any]:
        device = comfy.model_management.get_torch_device()
        torch.manual_seed(seed)

        # image is [B, H, W, C] float32 in [0, 1]
        tensor: np.ndarray = (image * 255).numpy().astype(np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])

        tags_list: List[str] = []
        scores_list: List[Dict[str, float]] = []

        for i in range(tensor.shape[0]):
            img: Image.Image = Image.fromarray(tensor[i]).convert("RGB")

            tag_str, scores = DINOv3Inference.run_classifier(
                model_name=model,
                device=device,
                image=img,
                mode=mode,
                topk=topk,
                threshold=threshold,
                max_size=max_size,
                exclude_tags=exclude_tags,
                replace_underscore=replace_underscore,
                trailing_comma=trailing_comma,
                prefix=prefix,
                seed=seed,
            )

            tags_list.append(tag_str)
            scores_list.append(scores)
            pbar.update(1)

        return {
            "ui": {"tags": tags_list, "scores": scores_list},
            "result": (tags_list, scores_list),
        }


# ---------------------------------------------------------------------------
# Module-level initialisation (runs when ComfyUI loads the node)
# ---------------------------------------------------------------------------

model_basepath = os.path.join(folder_paths.models_dir, "RedRocket")
tags_basepath = os.path.join(model_basepath, "tags")

DINOv3ModelManager(
    model_basepath=model_basepath,
    download_progress_callback=download_progress_callback,
    download_complete_callback=download_complete_callback,
)
DINOv3TagManager(
    tags_basepath=tags_basepath,
    download_progress_callback=download_progress_callback,
    download_complete_callback=download_complete_callback,
)
DINOv3ImageManager()
DINOv3Inference(device=comfy.model_management.get_torch_device())
