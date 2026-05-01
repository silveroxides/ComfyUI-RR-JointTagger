"""DINOv3 Tagger inference orchestrator."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import comfy.model_management as mm

from ..helpers.cache import ComfyCache
from ..helpers.logger import ComfyLogger

from .image_dino_manager import DINOv3ImageManager
from .model_dino_manager import DINOv3ModelManager
from .tag_dino_manager import CATEGORY_ID_TO_NAME, DINOv3TagManager


class DINOv3Inference:
    """Inference wrapper for the DINOv3 Tagger pipeline.

    Orchestrates model loading, vocab loading, image preprocessing, and
    forward pass.  Returns formatted tag strings and score dicts.
    """

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
        implications_mode: str = "off",
        exclude_tags: str = "",
        exclude_categories: str = "",
        max_tags: int = 0,
        trailing_comma: bool = False,
        prefix: str = "",
        seed: int = 0,
        category_config: Optional[Dict[str, Any]] = None,
        check_updates: bool = False,
        use_aliases: bool = False,
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
            Minimum score to include.
        max_size:
            Long-edge pixel cap (snapped to 16px multiples).
        implications_mode:
            How to handle implied tags: ``"inherit"``, ``"constrain"``,
            ``"remove"``, ``"constrain-remove"``, or ``"off"``.
        exclude_tags:
            Comma-separated tags to exclude from the output.
        exclude_categories:
            Comma-separated category names to exclude
            (e.g. ``"copyright, character, meta"``).
        trailing_comma:
            Append a trailing comma to the output tag string.
        prefix:
            Text to prepend to the tag string.
        seed:
            Random seed (unused by inference, kept for ComfyUI determinism).
        category_config:
            Optional per-category topk/threshold overrides dict from
            ``DINOv3CategoryConfig``.  Keys are ``"{cat}_topk"`` and
            ``"{cat}_threshold"``.  Categories with both 0 use globals.
        use_aliases:
            If ``True``, use aliased tag names in the output.

        Returns
        -------
        (tag_string, scores_dict)
        """
        # Build a params hash for caching
        cc_hash = ""
        if category_config:
            cc_hash = "|".join(f"{k}={v}" for k, v in sorted(category_config.items()))
        params_string = (
            f"{model_name}|{mode}|{topk}|{threshold}|{max_size}|"
            f"{implications_mode}|{exclude_tags}|{exclude_categories}|"
            f"{max_tags}|{trailing_comma}|{prefix}|{seed}|{cc_hash}|{use_aliases}"
        )
        params_key = hashlib.sha256(params_string.encode()).hexdigest()

        # ----------------------------------------------------------
        # 1. Ensure model is available
        # ----------------------------------------------------------
        import os
        tags_basepath = DINOv3TagManager().tags_basepath

        if DINOv3ModelManager.is_installed(model_name):
            # Check for remote update (once per session, only if enabled)
            if check_updates:
                if DINOv3ModelManager.check_for_update(model_name):
                    # Model updated, clear vocab so it gets re-downloaded
                    for ext in ["-vocab.json", "-cat-vocab.json", "-tag2implicit.json"]:
                        p = os.path.join(tags_basepath, f"{model_name}{ext}")
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception as e:
                                ComfyLogger().log(f"Failed to remove {p}: {e}", "WARNING", True)
                    # Clear vocab cache
                    ComfyCache.set(f"tags_dino.{model_name}", None)
        else:
            if not DINOv3ModelManager.download(model_name):
                ComfyLogger().log("DINOv3: failed to download model", "ERROR", True)
                return "", {}

        # ----------------------------------------------------------
        # 2. Ensure vocab is available
        # ----------------------------------------------------------
        if not DINOv3TagManager.is_loaded(model_name):
            # Prefer categorised vocab (gives idx2tag + tag2category)
            cat_vocab_path = os.path.join(
                tags_basepath, f"{model_name}-cat-vocab.json",
            )
            loaded_cat = False
            if not os.path.exists(cat_vocab_path):
                loaded_cat = DINOv3TagManager.download_cat_vocab(model_name)
            else:
                loaded_cat = True

            if loaded_cat:
                loaded_cat = DINOv3TagManager.load_cat_vocab(model_name)

            # Fall back to plain vocab if cat-vocab unavailable
            if not loaded_cat:
                ComfyLogger().log(
                    "DINOv3: categorised vocab unavailable, "
                    "falling back to plain vocab",
                    type="WARNING", always=True,
                )
                vocab_path = os.path.join(
                    tags_basepath, f"{model_name}-vocab.json",
                )
                if not os.path.exists(vocab_path):
                    if not DINOv3TagManager.download(model_name):
                        ComfyLogger().log(
                            "DINOv3: failed to download vocab",
                            type="ERROR", always=True,
                        )
                        return "", {}
                if not DINOv3TagManager.load(model_name):
                    ComfyLogger().log(
                        "DINOv3: failed to load vocab",
                        type="ERROR", always=True,
                    )
                    return "", {}

        # Ensure tag2category is loaded even if vocab was already cached
        # (handles the case where a previous run loaded the plain vocab)
        if not DINOv3TagManager.has_categories(model_name):
            cat_vocab_path = os.path.join(
                tags_basepath, f"{model_name}-cat-vocab.json",
            )
            if not os.path.exists(cat_vocab_path):
                DINOv3TagManager.download_cat_vocab(model_name)
            if os.path.exists(cat_vocab_path):
                DINOv3TagManager.load_cat_vocab(model_name)

        num_tags = DINOv3TagManager.get_num_tags(model_name)
        idx2tag = DINOv3TagManager.get_idx2tag(model_name)
        if not idx2tag or num_tags == 0:
            ComfyLogger().log("DINOv3: vocab is empty", "ERROR", True)
            return "", {}

        # Load tag metadata (implications) if needed and available.
        # Category exclusion uses tag2category from cat-vocab independently.
        if implications_mode != "off" and not DINOv3TagManager.has_metadata(model_name):
            json_path = os.path.join(tags_basepath, f"{model_name}-tag2implicit.json")
            if not os.path.exists(json_path):
                DINOv3TagManager.download_metadata(model_name)
            DINOv3TagManager.load_metadata(model_name)

        # Now load model
        if not DINOv3ModelManager.is_loaded(model_name):
            if not DINOv3ModelManager.load(model_name, num_tags=num_tags, device=device):
                ComfyLogger().log("DINOv3: failed to load model", "ERROR", True)
                return "", {}

        model_data = ComfyCache.get(f"model_dino.{model_name}")
        if not model_data or not isinstance(model_data, dict):
            ComfyLogger().log("DINOv3: model cache corrupted", "ERROR", True)
            return "", {}

        model = model_data["model"]
        model_dtype = model_data.get("dtype", torch.bfloat16)

        # Move model to inference device (may have been offloaded to CPU)
        model.to(device)

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
        try:
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=model_dtype,
            ):
                logits = model(pixel_values)[0]  # [num_tags]

            scores = torch.sigmoid(logits.float())

            # ----------------------------------------------------------
            # 5. Top-K + threshold selection
            # ----------------------------------------------------------
            tag2category = DINOv3TagManager.get_tag2category(model_name)

            # Use per-category selection when config is provided and categories
            # are available; otherwise fall back to the global logic.
            use_per_cat = (
                category_config is not None
                and tag2category is not None
                and any(
                    category_config.get(f"{cat}_topk", 0) != 0
                    or category_config.get(f"{cat}_threshold", 0.0) != 0.0
                    for cat in CATEGORY_ID_TO_NAME.values()
                )
            )

            if use_per_cat:
                raw_results = cls._per_category_select(
                    scores=scores,
                    idx2tag=idx2tag,
                    tag2category=tag2category,
                    category_config=category_config,
                    global_mode=mode,
                    global_topk=topk,
                    global_threshold=threshold,
                )
            else:
                # Global selection (original logic)
                mask = scores >= threshold
                valid_indices = mask.nonzero(as_tuple=True)[0]
                valid_values = scores[valid_indices]

                order = valid_values.argsort(descending=True)
                valid_indices = valid_indices[order]
                valid_values = valid_values[order]

                if mode == "topk":
                    k = min(topk, len(valid_indices))
                    valid_indices = valid_indices[:k]
                    valid_values = valid_values[:k]

                raw_results: List[Tuple[str, float]] = [
                    (idx2tag[i], float(v))
                    for i, v in zip(valid_indices.tolist(), valid_values.tolist())
                ]

            # ----------------------------------------------------------
            # 6. Process and format tags
            # ----------------------------------------------------------
            tag_string, scores_dict = DINOv3TagManager.process_tags(
                results=raw_results,
                model_name=model_name,
                implications_mode=implications_mode,
                threshold=threshold,
                max_tags=max_tags,
                exclude_tags=exclude_tags,
                exclude_categories=exclude_categories,
                trailing_comma=trailing_comma,
                prefix=prefix,
                use_aliases=use_aliases,
            )
        finally:
            # ----------------------------------------------------------
            # 7. Offload model to free GPU memory
            # ----------------------------------------------------------
            model.to("cpu")
            mm.soft_empty_cache()

        # ----------------------------------------------------------
        # 8. Cache result
        # ----------------------------------------------------------
        image_key = DINOv3ImageManager.image_key_for(image)
        if image_key:
            DINOv3ImageManager.commit_cache(
                image_key, (tag_string, scores_dict), params_key,
            )

        return tag_string, scores_dict

    # ------------------------------------------------------------------
    # Per-category tag selection
    # ------------------------------------------------------------------

    @staticmethod
    def _per_category_select(
        scores: torch.Tensor,
        idx2tag: List[str],
        tag2category: Dict[str, int],
        category_config: Dict[str, Any],
        global_mode: str,
        global_topk: int,
        global_threshold: float,
    ) -> List[Tuple[str, float]]:
        """Select tags using per-category topk/threshold overrides.

        For each tag index, look up its category via ``tag2category``.
        Group indices by category, then apply the per-category settings
        from ``category_config``.  Categories where both topk and
        threshold are 0 fall back to the global settings.

        Returns a list of ``(tag, score)`` tuples sorted by descending
        score.
        """
        # Group tag indices by category ID
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(idx2tag)):
            tag_name = idx2tag[i]
            cat_id = tag2category.get(tag_name, -1)
            groups[cat_id].append(i)

        all_selected: List[Tuple[str, float]] = []

        for cat_id, indices_list in groups.items():
            cat_name = CATEGORY_ID_TO_NAME.get(cat_id, "unassigned")
            cat_topk = category_config.get(f"{cat_name}_topk", 0)
            cat_threshold = category_config.get(f"{cat_name}_threshold", 0.0)

            # Determine effective settings for this category
            if cat_topk == 0 and cat_threshold == 0.0:
                # Use global settings
                eff_threshold = global_threshold
                eff_topk = global_topk if global_mode == "topk" else 0
            elif cat_topk == 0:
                # Threshold-only mode for this category
                eff_threshold = cat_threshold
                eff_topk = 0
            else:
                # TopK mode for this category, threshold as floor
                eff_threshold = cat_threshold if cat_threshold > 0.0 else global_threshold
                eff_topk = cat_topk

            # Convert to tensor for efficient operations
            idx_tensor = torch.tensor(indices_list, dtype=torch.long, device=scores.device)
            cat_scores = scores[idx_tensor]

            # Apply threshold
            mask = cat_scores >= eff_threshold
            valid_pos = mask.nonzero(as_tuple=True)[0]
            valid_idx = idx_tensor[valid_pos]
            valid_vals = cat_scores[valid_pos]

            # Sort descending
            order = valid_vals.argsort(descending=True)
            valid_idx = valid_idx[order]
            valid_vals = valid_vals[order]

            # Apply topk cap
            if eff_topk > 0:
                k = min(eff_topk, len(valid_idx))
                valid_idx = valid_idx[:k]
                valid_vals = valid_vals[:k]

            for i_val, v_val in zip(valid_idx.tolist(), valid_vals.tolist()):
                all_selected.append((idx2tag[i_val], float(v_val)))

        # Final sort by score across all categories
        all_selected.sort(key=lambda x: x[1], reverse=True)
        return all_selected
