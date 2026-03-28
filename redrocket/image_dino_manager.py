"""DINOv3 Tagger image preprocessing manager."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from PIL import Image

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

from .dinov3_backbone import PATCH_SIZE

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _snap(x: int, m: int) -> int:
    """Round *down* to nearest multiple of ``m``, minimum ``m``."""
    return max(m, (x // m) * m)


class DINOv3ImageManager(metaclass=Singleton):
    """Singleton manager for DINOv3 image preprocessing.

    Converts PIL / numpy / ComfyUI tensors into ImageNet-normalised
    ``[1, 3, H, W]`` tensors with dimensions snapped to multiples of
    ``PATCH_SIZE`` (16).  Results are cached under ``image_dino``.
    """

    def __init__(self) -> None:
        ComfyCache.set_max_size("image_dino", 1)
        ComfyCache.set_cachemethod("image_dino", CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush("image_dino")
        import gc
        gc.collect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def preprocess(
        cls,
        image: Union[Image.Image, np.ndarray, Path],
        device: torch.device,
        max_size: int = 1024,
        params_key: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[str, Dict[str, float]], None]:
        """Preprocess an image for DINOv3 inference.

        Returns
        -------
        torch.Tensor
            ``[1, 3, H, W]`` float32 ImageNet-normalised tensor on *device*.
        tuple
            Cached ``(tag_string, scores_dict)`` if a matching result exists.
        None
            If the image could not be processed.
        """
        image_key, pil_image = cls._resolve_image(image)
        if image_key is None or pil_image is None:
            ComfyLogger().log("DINOv3 image: no valid image provided", "ERROR", True)
            return None

        # Check output cache
        cache_key = f"image_dino.{image_key}"
        cached_output = ComfyCache.get(f"{cache_key}.output")
        if params_key and cached_output and isinstance(cached_output, dict):
            if params_key in cached_output:
                ComfyLogger().log(
                    f"DINOv3 image: returning cached result for {image_key[:16]}…",
                    "DEBUG", True,
                )
                return cached_output[params_key]

        # Check preprocessed tensor cache (keyed by max_size)
        cached_input = ComfyCache.get(f"{cache_key}.input")
        if cached_input is not None and isinstance(cached_input, dict):
            if max_size in cached_input:
                return cached_input[max_size].to(device)

        # Process
        tensor = cls._to_tensor(pil_image, max_size)

        # Store in input cache
        if cached_input is None:
            cached_input = {}
        cached_input[max_size] = tensor
        ComfyCache.set(f"{cache_key}.input", cached_input)

        # Ensure output dict exists
        if ComfyCache.get(f"{cache_key}.output") is None:
            ComfyCache.set(f"{cache_key}.output", {})

        return tensor.to(device)

    @classmethod
    def commit_cache(
        cls,
        image_key: str,
        output: Any,
        params_key: str,
    ) -> bool:
        """Store inference results in the output cache."""
        if not image_key:
            return False
        cache_key = f"image_dino.{image_key}"
        current_output = ComfyCache.get(f"{cache_key}.output")
        if current_output is None:
            current_output = {}
        current_output[params_key] = output
        ComfyCache.set(f"{cache_key}.output", current_output)
        return True

    @classmethod
    def image_key_for(cls, image: Union[Image.Image, np.ndarray, Path]) -> Optional[str]:
        """Return a stable hash key for the given image (for use by the classifier)."""
        key, _ = cls._resolve_image(image)
        return key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_image(
        cls,
        image: Union[Image.Image, np.ndarray, Path, None],
    ) -> Tuple[Optional[str], Optional[Image.Image]]:
        """Return ``(image_key, pil_image)`` from various input types."""
        if image is None:
            return None, None

        if isinstance(image, Path):
            pil = Image.open(str(image)).convert("RGB")
            return str(image), pil

        if isinstance(image, np.ndarray):
            key = hashlib.sha256(image.tobytes()).hexdigest()
            pil = Image.fromarray(image).convert("RGB")
            return key, pil

        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
            key = hashlib.sha256(arr.tobytes()).hexdigest()
            return key, image.convert("RGB")

        return None, None

    @staticmethod
    def _to_tensor(img: Image.Image, max_size: int) -> torch.Tensor:
        """Resize + ImageNet normalise → ``[1, 3, H, W]`` float32."""
        w, h = img.size
        scale = min(1.0, max_size / max(w, h))
        new_w = _snap(round(w * scale), PATCH_SIZE)
        new_h = _snap(round(h * scale), PATCH_SIZE)

        transform = v2.Compose([
            v2.Resize((new_h, new_w), interpolation=v2.InterpolationMode.LANCZOS),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        return transform(img).unsqueeze(0)
