import os
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import comfy.model_management as mm

from .image_manager import JtpImageManager

from ..redrocket.tag_manager import CATEGORY_ID_TO_NAME, JtpTagManager
from ..redrocket.model_manager import JtpModelManager
from ..helpers.cache import ComfyCache
class JtpInference:
    """
    A Clip Vision Classifier by RedRocket (inference code made robust by deitydurg)
    """

    @classmethod
    def run_classifier(
        cls,
        model_name: str,
        tags_name: str,
        device: torch.device,
        image: Union[Image.Image, np.ndarray, Path],
        steps: float,
        threshold: float,
        exclude_tags: str = "",
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        implications_mode: str = "off",
        exclude_categories: str = "",
        prefix: str = "",
        mode: str = "topk",
        topk: int = 40,
        max_tags: int = 0,
        category_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, float]]:
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

        # Load implications metadata if needed (also required for per-category config)
        needs_metadata = implications_mode != "off" or category_config is not None
        if needs_metadata:
            if not JtpTagManager().has_metadata(tags_name):
                # Download CSV if not on disk yet
                csv_path = os.path.join(JtpTagManager().tags_basepath, f"{tags_name}-tags.csv")
                if not os.path.exists(csv_path):
                    JtpTagManager().download_metadata(model_name)
                JtpTagManager().load_metadata(tags_name)

        # Create a unique key for the parameters to handle caching correctly
        import hashlib
        cc_hash = ""
        if category_config:
            cc_hash = "|".join(f"{k}={v}" for k, v in sorted(category_config.items()))
        params_string = f"{model_name}|{tags_name}|{steps}|{threshold}|{exclude_tags}|{replace_underscore}|{trailing_comma}|{implications_mode}|{exclude_categories}|{prefix}|{mode}|{topk}|{max_tags}|{cc_hash}"
        params_key = hashlib.sha256(params_string.encode()).hexdigest()

        tensor = JtpImageManager().load(image=image, device=device, params_key=params_key)
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
        # Move model to inference device (may have been offloaded to CPU)
        model_data.to(device)
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
        # Offload model to free GPU memory
        offload_device = mm.unet_offload_device()
        model_data.to(offload_device)
        mm.soft_empty_cache()

        # ----------------------------------------------------------
        # Top-K + threshold selection
        # ----------------------------------------------------------
        metadata = ComfyCache.get(f'tags.{tags_name}.metadata')

        # Check if per-category selection should be used
        use_per_cat = (
            category_config is not None
            and metadata is not None
            and any(
                category_config.get(f"{cat}_topk", 0) != 0
                or category_config.get(f"{cat}_threshold", 0.0) != 0.0
                for cat in CATEGORY_ID_TO_NAME.values()
            )
        )

        if use_per_cat:
            # Build tag list from top-250 for per-category filtering
            tags_data = ComfyCache.get(f'tags.{tags_name}.tags')
            idx2tag: List[str] = [""] * len(tags_data) if tags_data else []
            if tags_data:
                for key, val in tags_data.items():
                    tag = key.replace("_", " ")
                    if val < len(idx2tag):
                        idx2tag[val] = tag

            raw_results = cls._per_category_select(
                values=values,
                indices=indices,
                idx2tag=idx2tag,
                metadata=metadata,
                category_config=category_config,
                global_mode=mode,
                global_topk=topk,
                global_threshold=threshold,
            )

            # Convert raw_results back to tensors for process_tags
            # We re-build values/indices from the selected results
            if raw_results:
                sel_indices = torch.tensor(
                    [list(tags_data.values())[list(tags_data.keys()).index(
                        next(k for k, v_map in tags_data.items() if k.replace("_", " ") == tag)
                    )] for tag, _ in raw_results],
                    dtype=torch.long,
                )
                sel_values = torch.tensor([s for _, s in raw_results], dtype=torch.float32)
            else:
                sel_indices = torch.tensor([], dtype=torch.long)
                sel_values = torch.tensor([], dtype=torch.float32)

            values = sel_values
            indices = sel_indices
        else:
            # Global selection: apply mode-based filtering on existing top-250
            if mode == "topk":
                # Already have top-250 sorted by score; apply threshold floor then cap to topk
                mask = values >= threshold
                values = values[mask]
                indices = indices[mask]
                k = min(topk, len(values))
                values = values[:k]
                indices = indices[:k]

        ComfyLogger().log(message="Processing tags...", type="DEBUG", always=True)
        tags_str, tag_scores = JtpTagManager().process_tags(
            tags_name=tags_name,
            values=values,
            indices=indices,
            threshold=threshold if not use_per_cat else -1.0,  # Skip threshold in process_tags if per-cat already applied
            exclude_tags=exclude_tags,
            replace_underscore=replace_underscore,
            trailing_comma=trailing_comma,
            implications_mode=implications_mode,
            exclude_categories=exclude_categories,
            prefix=prefix,
            max_tags=max_tags,
        )
        if not JtpImageManager().commit_cache(image=image, output=(tags_str, tag_scores), params_key=params_key):
            ComfyLogger().log(message="Image cache could not be committed", type="WARN", always=True)
            return tags_str, tag_scores
        ComfyLogger().log(message=f"Classification complete: {tags_str}", type="INFO", always=True)
        return tags_str, tag_scores

    # ------------------------------------------------------------------
    # Per-category tag selection
    # ------------------------------------------------------------------

    @staticmethod
    def _per_category_select(
        values: torch.Tensor,
        indices: torch.Tensor,
        idx2tag: List[str],
        metadata: Dict[str, Tuple],
        category_config: Dict[str, Any],
        global_mode: str,
        global_topk: int,
        global_threshold: float,
    ) -> List[Tuple[str, float]]:
        """Select tags using per-category topk/threshold overrides.

        Groups the top-250 model outputs by category (from metadata CSV),
        then applies per-category settings from *category_config*.
        Categories where both topk and threshold are 0 fall back to global
        settings.

        Returns a list of ``(tag, score)`` tuples sorted by descending score.
        """
        # Build per-category groups from top-250 results
        groups: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        for i in range(indices.size(0)):
            idx_val = indices[i].item()
            score = values[i].item()
            if idx_val < len(idx2tag) and idx2tag[idx_val]:
                tag = idx2tag[idx_val]
                # Look up category from metadata
                cat_id = metadata[tag][0] if tag in metadata else 0
                groups[cat_id].append((tag, score))

        all_selected: List[Tuple[str, float]] = []

        for cat_id, tag_scores in groups.items():
            cat_name = CATEGORY_ID_TO_NAME.get(cat_id, "general")
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

            # Sort by score descending (should already be sorted, but be safe)
            tag_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply threshold
            filtered = [(t, s) for t, s in tag_scores if s >= eff_threshold]

            # Apply topk cap
            if eff_topk > 0:
                filtered = filtered[:eff_topk]

            all_selected.extend(filtered)

        # Final sort by score across all categories
        all_selected.sort(key=lambda x: x[1], reverse=True)
        return all_selected
