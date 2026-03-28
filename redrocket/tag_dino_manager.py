"""DINOv3 Tagger vocabulary manager — download, load, process tags.

Supports optional implications metadata loaded from a JTP-3-style CSV.
All tag keys are normalised to spaces (no underscores) to match DINOv3's
native output format.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import traceback
from typing import Callable, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

# Category IDs used in the JTP-3 tag CSV
TAG_CATEGORIES = {
    "general": 0,
    "copyright": 3,
    "character": 4,
    "species": 5,
    "meta": 7,
    "lore": 8,
}


class DINOv3TagManager(metaclass=Singleton):
    """Singleton manager for DINOv3 tagger vocabulary and optional metadata.

    The vocabulary is a JSON file with ``{"idx2tag": ["tag0", "tag1", ...]}``.
    Tag metadata (implications, categories) is loaded from a JTP-3-style CSV
    and normalised so all keys use spaces instead of underscores.

    Cached under the ``tags_dino`` namespace.
    """

    def __init__(
        self,
        tags_basepath: str,
        download_progress_callback: Callable[[int, str], None],
        download_complete_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.tags_basepath = tags_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        ComfyCache.set_max_size("tags_dino", 1)
        ComfyCache.set_cachemethod("tags_dino", CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush("tags_dino")
        gc.collect()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        return ComfyCache.get(f"tags_dino.{model_name}") is not None

    @classmethod
    def has_metadata(cls, model_name: str) -> bool:
        return ComfyCache.get(f"tags_dino.{model_name}.metadata") is not None

    @classmethod
    def get_idx2tag(cls, model_name: str) -> Optional[List[str]]:
        return ComfyCache.get(f"tags_dino.{model_name}.idx2tag")

    @classmethod
    def get_num_tags(cls, model_name: str) -> int:
        idx2tag = cls.get_idx2tag(model_name)
        return len(idx2tag) if idx2tag else 0

    # ------------------------------------------------------------------
    # Load vocabulary (JSON)
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_name: str) -> bool:
        vocab_path = os.path.join(cls().tags_basepath, f"{model_name}-vocab.json")
        if not os.path.exists(vocab_path):
            ComfyLogger().log(
                f"DINOv3 vocab file not found: {vocab_path}",
                type="ERROR", always=True,
            )
            return False

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            idx2tag: List[str] = data["idx2tag"]
            ComfyCache.set(f"tags_dino.{model_name}", {
                "idx2tag": idx2tag,
                "num_tags": len(idx2tag),
            })
            ComfyLogger().log(
                f"DINOv3 vocab loaded: {len(idx2tag):,} tags for {model_name}",
                type="INFO", always=True,
            )
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Error loading DINOv3 vocab: {e}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False

    # ------------------------------------------------------------------
    # Load metadata / implications (CSV)
    # ------------------------------------------------------------------

    @classmethod
    def load_metadata(cls, model_name: str) -> bool:
        """Load tag metadata (categories + implications) from a JTP-3-style
        CSV.  All tag keys and implication references are normalised to use
        spaces instead of underscores to match DINOv3 output format.
        """
        csv_path = os.path.join(cls().tags_basepath, f"{model_name}-tags.csv")
        if not os.path.exists(csv_path):
            ComfyLogger().log(
                f"DINOv3 tag metadata CSV not found: {csv_path} "
                "(implications will be unavailable)",
                type="WARNING", always=True,
            )
            return False

        try:
            metadata: Dict[str, Tuple[int, List[str]]] = {}
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Normalise tag: underscores -> spaces
                    tag = row["tag"].replace("_", " ")
                    category = int(row.get("category", 0))
                    # Normalise implications the same way
                    raw_impl = row.get("implications", "")
                    implications = [
                        imp.replace("_", " ")
                        for imp in raw_impl.split()
                        if imp
                    ]
                    metadata[tag] = (category, implications)

            ComfyCache.set(f"tags_dino.{model_name}.metadata", metadata)
            ComfyLogger().log(
                f"DINOv3 tag metadata loaded: {len(metadata):,} entries for {model_name}",
                type="INFO", always=True,
            )
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Error loading DINOv3 tag metadata: {e}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False

    # ------------------------------------------------------------------
    # Download vocabulary (JSON)
    # ------------------------------------------------------------------

    @classmethod
    def download(cls, model_name: str) -> bool:
        os.makedirs(cls().tags_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        vocab_url = ComfyExtensionConfig().get(
            property=f"models_dino.{model_name}.vocab_url",
        )
        if not vocab_url:
            ComfyLogger().log(
                f"No vocab URL for DINOv3 model {model_name} in config",
                type="ERROR", always=True,
            )
            return False
        vocab_url = vocab_url.replace("{HF_ENDPOINT}", hf_endpoint)

        dest = os.path.join(cls().tags_basepath, f"{model_name}-vocab.json")
        ComfyLogger().log(
            f"Downloading DINOv3 vocab from {vocab_url}",
            type="INFO", always=True,
        )

        try:
            response = requests.get(vocab_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with open(dest, "wb") as f, tqdm(
                desc=f"{model_name}-vocab",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
                    downloaded += len(data)
                    if cls().download_progress_callback and total_size > 0:
                        cls().download_progress_callback(
                            int(downloaded / total_size * 100),
                            f"{model_name}-vocab.json",
                        )

            if cls().download_complete_callback:
                cls().download_complete_callback(f"{model_name}-vocab.json")
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Failed to download DINOv3 vocab: {e}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False

    # ------------------------------------------------------------------
    # Download metadata (CSV)
    # ------------------------------------------------------------------

    @classmethod
    def download_metadata(cls, model_name: str) -> bool:
        """Download tag metadata CSV from the configured ``data_url``."""
        os.makedirs(cls().tags_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        data_url = ComfyExtensionConfig().get(
            property=f"models_dino.{model_name}.data_url",
        )
        if not data_url:
            ComfyLogger().log(
                f"No data_url for DINOv3 model {model_name} — "
                "implications will not be available",
                type="WARNING", always=True,
            )
            return False
        data_url = data_url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not data_url.endswith("/"):
            data_url += "/"

        # The CSV filename follows the convention from the data_url source.
        # The config's data_url_tag_file overrides the default if set.
        tag_file = ComfyExtensionConfig().get(
            property=f"models_dino.{model_name}.data_url_tag_file",
        )
        if not tag_file or not isinstance(tag_file, str):
            # Default: derive from the JTP-3 naming convention
            tag_file = ComfyExtensionConfig().get(
                property=f"models_dino.{model_name}.data_tag_file",
            )
        if not tag_file or not isinstance(tag_file, str):
            tag_file = "jtp-3-hydra-tags.csv"

        url = f"{data_url}{tag_file}"
        dest = os.path.join(cls().tags_basepath, f"{model_name}-tags.csv")

        ComfyLogger().log(
            f"Downloading DINOv3 tag metadata from {url}",
            type="INFO", always=True,
        )

        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                ComfyLogger().log(
                    f"Failed to download tag metadata: HTTP {response.status_code}",
                    type="WARNING", always=True,
                )
                return False

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with open(dest, "wb") as f, tqdm(
                desc=f"{model_name}-tags.csv",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                downloaded = 0
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    bar.update(len(chunk))
                    downloaded += len(chunk)
                    if cls().download_progress_callback and total_size > 0:
                        cls().download_progress_callback(
                            int(downloaded / total_size * 100),
                            f"{model_name}-tags.csv",
                        )

            if cls().download_complete_callback:
                cls().download_complete_callback(f"{model_name}-tags.csv")
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Failed to download DINOv3 tag metadata: {e}\n{traceback.format_exc()}",
                type="WARNING", always=True,
            )
            return False

    # ------------------------------------------------------------------
    # Tag processing (with optional implications)
    # ------------------------------------------------------------------

    @classmethod
    def process_tags(
        cls,
        results: List[Tuple[str, float]],
        model_name: str = "",
        implications_mode: str = "off",
        threshold: float = 0.0,
        max_tags: int = 0,
        exclude_tags: str = "",
        exclude_categories: str = "",
        trailing_comma: bool = False,
        prefix: str = "",
    ) -> Tuple[str, Dict[str, float]]:
        """Format raw (tag, score) list into a tag string and scores dict.

        DINOv3 tags are already space-separated (no underscores), so no
        ``replace_underscore`` option is needed.

        Parameters
        ----------
        results:
            List of ``(tag, score)`` tuples, already sorted by descending score.
        model_name:
            Config key — needed to look up metadata for implications.
        implications_mode:
            ``"inherit"``, ``"constrain"``, ``"remove"``,
            ``"constrain-remove"``, or ``"off"``.
        threshold:
            Minimum score floor — re-applied after implications processing
            so that constrained tags dropping below threshold are removed.
        max_tags:
            Maximum number of tags to return (applied after implications and
            filtering).  0 = unlimited.
        exclude_tags:
            Comma-separated tags to exclude.
        exclude_categories:
            Comma-separated category names to exclude
            (e.g. ``"copyright, character, meta"``).
        trailing_comma:
            Append a trailing comma to the tag string.
        prefix:
            Text to prepend.
        """
        metadata = ComfyCache.get(f"tags_dino.{model_name}.metadata")

        # Build exclusion set (normalised to lowercase spaces)
        excluded: Set[str] = set()
        if exclude_tags:
            for t in exclude_tags.split(","):
                t = t.strip()
                if t:
                    excluded.add(t.replace("_", " ").lower())

        # Parse excluded categories
        excluded_cats: Set[int] = set()
        if exclude_categories and metadata:
            for cat in exclude_categories.split(","):
                cat = cat.strip().lower()
                if cat in TAG_CATEGORIES:
                    excluded_cats.add(TAG_CATEGORIES[cat])

        # Build labels dict from results
        labels: Dict[str, float] = {tag: score for tag, score in results}

        # ----- Implications processing (only if metadata is available) -----
        if metadata and implications_mode != "off":
            def inherit_implications(lbls: Dict[str, float], antecedent: str) -> None:
                if antecedent not in metadata:
                    return
                p = lbls.get(antecedent, 0.0)
                for consequent in metadata[antecedent][1]:
                    if lbls.get(consequent, 0.0) < p:
                        lbls[consequent] = p
                    inherit_implications(lbls, consequent)

            def constrain_implications(
                lbls: Dict[str, float], antecedent: str, _target: Optional[str] = None,
            ) -> None:
                if _target is None:
                    _target = antecedent
                if antecedent not in metadata:
                    return
                for consequent in metadata[antecedent][1]:
                    p = lbls.get(consequent, 0.0)
                    if lbls.get(_target, 0.0) > p:
                        lbls[_target] = p
                    constrain_implications(lbls, consequent, _target=_target)

            def remove_implications(lbls: Dict[str, float], antecedent: str) -> None:
                if antecedent not in metadata:
                    return
                for consequent in metadata[antecedent][1]:
                    lbls.pop(consequent, None)
                    remove_implications(lbls, consequent)

            tag_keys = list(labels.keys())

            if implications_mode == "inherit":
                for tag in tag_keys:
                    inherit_implications(labels, tag)
            elif implications_mode in ("constrain", "constrain-remove"):
                for tag in tag_keys:
                    constrain_implications(labels, tag)

        # ----- Filtering (threshold re-applied after implications) -----
        filtered: Dict[str, float] = {}
        for tag, prob in labels.items():
            # Re-apply threshold (catches scores lowered by constrain)
            if prob < threshold:
                continue

            # Check exclusion by name
            tag_check = tag.replace("_", " ").lower()
            if tag_check in excluded:
                continue

            # Check exclusion by category
            if excluded_cats and metadata and tag in metadata:
                cat_id = metadata[tag][0]
                if cat_id in excluded_cats:
                    continue

            filtered[tag] = prob

        # ----- Remove implications step (after filtering) -----
        if metadata and implications_mode in ("remove", "constrain-remove"):
            keys_to_check = list(filtered.keys())
            for tag in keys_to_check:
                if tag in filtered:
                    if tag not in metadata:
                        continue
                    for consequent in metadata[tag][1]:
                        filtered.pop(consequent, None)
                        # Recurse
                        cls._remove_implications_recursive(
                            filtered, consequent, metadata,
                        )

        # ----- Format output -----
        sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        # Apply max_tags cap (post-implications limit)
        if max_tags > 0 and len(sorted_items) > max_tags:
            sorted_items = sorted_items[:max_tags]

        tag_list: List[str] = []
        scores_dict: Dict[str, float] = {}

        for tag, score in sorted_items:
            scores_dict[tag] = round(score, 4)
            tag_list.append(tag)

        # Prefix handling
        if prefix:
            prefix_val = prefix.strip()
            if prefix_val:
                if prefix_val in tag_list:
                    tag_list.remove(prefix_val)
                tag_list.insert(0, prefix_val)

        tag_string = ", ".join(tag_list)
        if trailing_comma and tag_string:
            tag_string += ","

        return tag_string, scores_dict

    @staticmethod
    def _remove_implications_recursive(
        labels: Dict[str, float],
        antecedent: str,
        metadata: Dict[str, Tuple[int, List[str]]],
    ) -> None:
        if antecedent not in metadata:
            return
        for consequent in metadata[antecedent][1]:
            labels.pop(consequent, None)
            DINOv3TagManager._remove_implications_recursive(
                labels, consequent, metadata,
            )
