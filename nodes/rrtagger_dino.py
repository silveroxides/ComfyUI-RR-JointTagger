"""ComfyUI nodes for the DINOv3 Tagger."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

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

# Custom ComfyUI type for the per-category config dict.
DINO_CATEGORY_CONFIG_TYPE = "DINO_CATEGORY_CONFIG"

# Canonical category names used as config-key prefixes (no slashes).
_CATEGORY_KEYS: List[str] = [
    "unassigned",
    "general",
    "artist",
    "contributor",
    "copyright",
    "character",
    "species_meta",
    "disambiguation",
    "meta",
    "lore",
]


class DINOv3CategoryConfig(ComfyNodeABC):
    """Supplementary node that outputs per-category topk/threshold settings.

    Connect its output to the *category_config* optional input on the
    **DINOv3 Tagger** node.  Any category left at the defaults
    (``topk=0, threshold=0.0``) will use the tagger's global settings.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for cat in _CATEGORY_KEYS:
            inputs[f"{cat}_topk"] = (IO.INT, {
                "default": 0, "min": 0, "max": 500, "step": 1,
                "tooltip": f"Top-K for '{cat}'. 0 with threshold 0.0 = use global.",
            })
            inputs[f"{cat}_threshold"] = (IO.FLOAT, {
                "default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01,
                "tooltip": f"Threshold for '{cat}'. 0.0 with topk 0 = use global.",
            })
        return {"required": inputs}

    RETURN_TYPES: Tuple[str, ...] = (DINO_CATEGORY_CONFIG_TYPE,)
    RETURN_NAMES: Tuple[str, ...] = ("category_config",)
    FUNCTION: str = "configure"
    CATEGORY: str = "🐺 Furry Diffusion"

    def configure(self, **kwargs: Any) -> Tuple[Dict[str, Any]]:
        """Pack all per-category values into a single dict."""
        config: Dict[str, Any] = {}
        for cat in _CATEGORY_KEYS:
            config[f"{cat}_topk"] = kwargs.get(f"{cat}_topk", 0)
            config[f"{cat}_threshold"] = kwargs.get(f"{cat}_threshold", 0.0)
        return (config,)


class DINOv3Tagger(ComfyNodeABC):
    """DINOv3 ViT-H/16+ image tagger node for ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        config = ComfyExtensionConfig().get()
        models = list(config.get("models_dino", {}).keys())
        if not models:
            models = ["tagger-proto"]

        return {
            "required": {
                "image": (IO.IMAGE,),
                "model": (models, {"default": models[0] if models else "tagger-proto"}),
                "mode": (["topk", "threshold"], {"default": "topk", "tooltip": "Tag selection mode. 'topk' returns the top K tags; 'threshold' returns all tags above a score. Threshold always acts as a minimum score floor in both modes."}),
                "topk": (IO.INT, {"default": 40, "min": 1, "max": 500, "step": 1, "display": "slider", "tooltip": "Number of top tags to return (when mode is 'topk')."}),
                "threshold": (IO.FLOAT, {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01, "display": "slider", "tooltip": "Minimum score to include a tag. Applied in both modes."}),
                "max_size": (IO.INT, {"default": 1024, "min": 64, "max": 4096, "step": 16, "tooltip": "Long-edge pixel cap. Image is resized to fit, snapped to 16px multiples."}),
                "implications_mode": (["off", "inherit", "constrain", "remove", "constrain-remove"], {"default": "off", "tooltip": "How to handle implied tags (e.g. if 'cat' is present, 'feline' is implied). Requires tag metadata CSV."}),
                "max_tags": (IO.INT, {"default": 0, "min": 0, "max": 500, "step": 1, "tooltip": "Maximum number of tags in the final output (applied after implications). 0 = unlimited."}),
                "exclude_tags": (IO.STRING, {"multiline": True, "tooltip": "Comma-separated tags to exclude from output."}),
                "exclude_categories": (IO.STRING, {"multiline": True, "tooltip": "Comma-separated categories to exclude: unassigned, general, artist, contributor, copyright, character, species/meta (or species), disambiguation, meta, lore."}),
                "trailing_comma": (IO.BOOLEAN, {"default": False, "tooltip": "Add a trailing comma to the tag string."}),
                "prefix": (IO.STRING, {"default": "", "tooltip": "Text to prepend to the tags output."}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for deterministic execution."}),
            },
            "optional": {
                "category_config": (DINO_CATEGORY_CONFIG_TYPE, {"tooltip": "Per-category topk/threshold overrides from the DINOv3 Category Config node."}),
                "check_updates": (IO.BOOLEAN, {"default": False, "tooltip": "Check for model updates on first run after ComfyUI launch. Compares the local SHA-256 against the remote repository and re-downloads if they differ."}),
            },
        }

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
        implications_mode: str,
        max_tags: int,
        seed: int,
        exclude_tags: str = "",
        exclude_categories: str = "",
        trailing_comma: bool = False,
        prefix: str = "",
        category_config: Optional[Dict[str, Any]] = None,
        check_updates: bool = False,
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
                implications_mode=implications_mode,
                exclude_tags=exclude_tags,
                exclude_categories=exclude_categories,
                max_tags=max_tags,
                trailing_comma=trailing_comma,
                prefix=prefix,
                seed=seed,
                category_config=category_config,
                check_updates=check_updates,
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
