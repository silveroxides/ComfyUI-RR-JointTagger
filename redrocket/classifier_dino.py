"""DINOv3 Tagger inference orchestrator."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..helpers.cache import ComfyCache
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

from .image_dino_manager import DINOv3ImageManager
from .model_dino_manager import DINOv3ModelManager
from .tag_dino_manager import DINOv3TagManager


class DINOv3Inference(metaclass=Singleton):
    """Inference wrapper for the DINOv3 Tagger pipeline.

    Orchestrates model loading, vocab loading, image preprocessing, and
    forward pass.  Returns formatted tag strings and score dicts.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_lock = threading.Lock()

    @classmethod
    def run_classifier(
        cls,
        model_name: str,
        device: torch.device,
        image: Union[Image.Image, np.ndarray, Path],
        mode: str = "topk",
        topk: int = 40,
        threshold: float = 0.35,
        max_size: int = 1024,
        exclude_tags: str = "",
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        prefix: str = "",
        seed: int = 0,
    ) -> Tuple[str, Dict[str, float]]:
        """Run the full DINOv3 tagging pipeline on a single image.

        Parameters
        ----------
        model_name:
            Model config key (e.g. ``"tagger-proto"``).
        device:
            Torch device to use.
        image:
            Input image as PIL, numpy array, or file Path.
        mode:
            ``"topk"`` or ``"threshold"``.
        topk:
            Number of top tags to return when *mode* is ``"topk"``.
        threshold:
            Minimum score to include when *mode* is ``"threshold"``.
        max_size:
            Long-edge pixel cap (snapped to 16px multiples).
        exclude_tags:
            Comma-separated tags to exclude from the output.
        replace_underscore:
            Replace underscores with spaces in output tags.
        trailing_comma:
            Append a trailing comma to the output tag string.
        prefix:
            Text to prepend to the tag string.
        seed:
            Random seed (unused by inference, kept for ComfyUI determinism).

        Returns
        -------
        (tag_string, scores_dict)
        """
        # Build a params hash for caching
        params_string = (
            f"{model_name}|{mode}|{topk}|{threshold}|{max_size}|"
            f"{exclude_tags}|{replace_underscore}|{trailing_comma}|"
            f"{prefix}|{seed}"
        )
        params_key = hashlib.sha256(params_string.encode()).hexdigest()

        # ----------------------------------------------------------
        # 1. Ensure vocab is available
        # ----------------------------------------------------------
        if not DINOv3TagManager.is_loaded(model_name):
            # Try downloading if file doesn't exist
            vocab_path_check = f"{model_name}-vocab.json"
            tags_basepath = DINOv3TagManager().tags_basepath
            import os
            if not os.path.exists(os.path.join(tags_basepath, vocab_path_check)):
                if not DINOv3TagManager.download(model_name):
                    ComfyLogger().log("DINOv3: failed to download vocab", "ERROR", True)
                    return "", {}
            if not DINOv3TagManager.load(model_name):
                ComfyLogger().log("DINOv3: failed to load vocab", "ERROR", True)
                return "", {}

        num_tags = DINOv3TagManager.get_num_tags(model_name)
        idx2tag = DINOv3TagManager.get_idx2tag(model_name)
        if not idx2tag or num_tags == 0:
            ComfyLogger().log("DINOv3: vocab is empty", "ERROR", True)
            return "", {}

        # ----------------------------------------------------------
        # 2. Ensure model is available
        # ----------------------------------------------------------
        if not DINOv3ModelManager.is_installed(model_name):
            if not DINOv3ModelManager.download(model_name):
                ComfyLogger().log("DINOv3: failed to download model", "ERROR", True)
                return "", {}

        if not DINOv3ModelManager.is_loaded(model_name):
            if not DINOv3ModelManager.load(model_name, num_tags=num_tags, device=device):
                ComfyLogger().log("DINOv3: failed to load model", "ERROR", True)
                return "", {}

        model = ComfyCache.get(f"model_dino.{model_name}.model")
        model_dtype = ComfyCache.get(f"model_dino.{model_name}.dtype") or torch.bfloat16

        # ----------------------------------------------------------
        # 3. Preprocess image (with caching)
        # ----------------------------------------------------------
        result = DINOv3ImageManager.preprocess(
            image, device=device, max_size=max_size, params_key=params_key,
        )

        # Cached output?
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]

        if result is None:
            return "", {}

        pixel_values: torch.Tensor = result  # [1, 3, H, W]

        # ----------------------------------------------------------
        # 4. Forward pass
        # ----------------------------------------------------------
        instance = cls()
        with instance.model_lock:
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=model_dtype,
            ):
                logits = model(pixel_values)[0]  # [num_tags]

        scores = torch.sigmoid(logits.float())

        # ----------------------------------------------------------
        # 5. Top-K or threshold selection
        # ----------------------------------------------------------
        if mode == "topk":
            k = min(topk, num_tags)
            values, indices = scores.topk(k)
        else:
            # Threshold mode
            mask = scores >= threshold
            indices = mask.nonzero(as_tuple=True)[0]
            values = scores[indices]
            order = values.argsort(descending=True)
            indices, values = indices[order], values[order]

        raw_results: List[Tuple[str, float]] = [
            (idx2tag[i], float(v))
            for i, v in zip(indices.tolist(), values.tolist())
        ]

        # ----------------------------------------------------------
        # 6. Process and format tags
        # ----------------------------------------------------------
        tag_string, scores_dict = DINOv3TagManager.process_tags(
            results=raw_results,
            exclude_tags=exclude_tags,
            replace_underscore=replace_underscore,
            trailing_comma=trailing_comma,
            prefix=prefix,
        )

        # ----------------------------------------------------------
        # 7. Cache result
        # ----------------------------------------------------------
        image_key = DINOv3ImageManager.image_key_for(image)
        if image_key:
            DINOv3ImageManager.commit_cache(
                image_key, (tag_string, scores_dict), params_key,
            )

        return tag_string, scores_dict
