"""DINOv3 Tagger vocabulary manager — download, load, process tags.

Supports per-tag category data loaded from a categorised vocab JSON
(``tagger_vocab_with_categories.json``) and optional implications metadata
loaded from a JTP-3-style CSV.  All tag keys are normalised to spaces (no
underscores) to match DINOv3's native output format.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import traceback
from fnmatch import fnmatchcase
from typing import Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm
import comfy.utils

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

# Raw category IDs from the DINOv3 categorised vocabulary.
TAG_CATEGORIES: Dict[str, int] = {
    "unassigned":    -1,
    "general":        0,
    "artist":         1,
    "contributor":    2,
    "copyright":      3,
    "character":      4,
    "species/meta":   5,
    "species":        5,   # alias so users can type either form
    "disambiguation": 6,
    "meta":           7,
    "lore":           8,
}

# Reverse lookup: raw category ID → canonical config key name.
# Used by per-category selection to map from tag2category IDs to the
# config dict keys produced by DINOv3CategoryConfig.
CATEGORY_ID_TO_NAME: Dict[int, str] = {
    -1: "unassigned",
     0: "general",
     1: "artist",
     2: "contributor",
     3: "copyright",
     4: "character",
     5: "species_meta",
     6: "disambiguation",
     7: "meta",
     8: "lore",
}


class DINOv3TagManager(metaclass=Singleton):
    """Singleton manager for DINOv3 tagger vocabulary and optional metadata.

    Vocabulary can be loaded from either:

    * **Categorised vocab** (preferred) — a JSON file with
      ``{"idx2tag": [...], "tag2category": {...}}``.  Provides both the
      ordered tag list and per-tag category IDs in a single download.
    * **Plain vocab** (fallback) — a JSON file with
      ``{"idx2tag": ["tag0", "tag1", ...]}``.  No category data.

    Tag metadata (implications) is loaded separately from a JTP-3-style CSV
    and normalised so all keys use spaces instead of underscores.

    Cached under the ``tags_dino`` namespace.
    """

    def __init__(self, tags_basepath: str) -> None:
        self.tags_basepath = tags_basepath
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
    def has_categories(cls, model_name: str) -> bool:
        """Return ``True`` if per-tag category data is cached."""
        return ComfyCache.get(f"tags_dino.{model_name}.tag2category") is not None

    @classmethod
    def get_tag2category(cls, model_name: str) -> Optional[Dict[str, int]]:
        """Return the ``{tag_name: raw_category_id}`` dict, or ``None``."""
        return ComfyCache.get(f"tags_dino.{model_name}.tag2category")

    @classmethod
    def get_tag2alias(cls, model_name: str) -> Optional[Dict[str, str]]:
        """Return the ``{tag_name: alias}`` dict, or ``None``."""
        return ComfyCache.get(f"tags_dino.{model_name}.tag2alias")

    @classmethod
    def get_idx2tag(cls, model_name: str) -> Optional[List[str]]:
        return ComfyCache.get(f"tags_dino.{model_name}.idx2tag")

    @classmethod
    def get_num_tags(cls, model_name: str) -> int:
        idx2tag = cls.get_idx2tag(model_name)
        return len(idx2tag) if idx2tag else 0

    # ------------------------------------------------------------------
    # Load vocabulary (JSON) — plain vocab (fallback)
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_name: str) -> bool:
        """Load the plain vocab JSON (``idx2tag`` only, no categories)."""
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
    # Load categorised vocabulary (JSON) — preferred
    # ------------------------------------------------------------------

    @classmethod
    def load_cat_vocab(cls, model_name: str) -> bool:
        """Load the categorised vocab JSON.

        The file contains both ``idx2tag`` (ordered tag list) and
        ``tag2category`` (tag name → raw category ID).  This is the
        preferred vocab source because it provides category data needed
        for ``exclude_categories`` without requiring the CSV metadata.
        """
        cat_vocab_path = os.path.join(
            cls().tags_basepath, f"{model_name}-cat-vocab.json",
        )
        if not os.path.exists(cat_vocab_path):
            ComfyLogger().log(
                f"DINOv3 categorised vocab not found: {cat_vocab_path}",
                type="WARNING", always=True,
            )
            return False

        try:
            with open(cat_vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            idx2tag: List[str] = data["idx2tag"]
            tag2category: Dict[str, int] = data.get("tag2category", {})
            tag2alias: Dict[str, str] = data.get("tag2alias", {})

            # Populate the same cache keys as load() so downstream is unaffected
            ComfyCache.set(f"tags_dino.{model_name}", {
                "idx2tag": idx2tag,
                "num_tags": len(idx2tag),
            })
            # Store category mapping separately
            ComfyCache.set(f"tags_dino.{model_name}.tag2category", tag2category)
            # Store alias mapping separately
            ComfyCache.set(f"tags_dino.{model_name}.tag2alias", tag2alias)

            ComfyLogger().log(
                f"DINOv3 categorised vocab loaded: {len(idx2tag):,} tags, "
                f"{len(tag2category):,} category entries, "
                f"{len(tag2alias):,} alias entries for {model_name}",
                type="INFO", always=True,
            )
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Error loading DINOv3 categorised vocab: {e}\n"
                f"{traceback.format_exc()}",
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

            pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None
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
                    if pbar is not None:
                        pbar.update_absolute(downloaded, total_size)

            return True

        except Exception as e:
            ComfyLogger().log(
                f"Failed to download DINOv3 vocab: {e}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False

    # ------------------------------------------------------------------
    # Download categorised vocabulary (JSON)
    # ------------------------------------------------------------------

    @classmethod
    def download_cat_vocab(cls, model_name: str) -> bool:
        """Download the categorised vocab JSON from ``cat_vocab_url``."""
        os.makedirs(cls().tags_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        cat_vocab_url = ComfyExtensionConfig().get(
            property=f"models_dino.{model_name}.cat_vocab_url",
        )
        if not cat_vocab_url:
            ComfyLogger().log(
                f"No cat_vocab_url for DINOv3 model {model_name} in config",
                type="WARNING", always=True,
            )
            return False
        cat_vocab_url = cat_vocab_url.replace("{HF_ENDPOINT}", hf_endpoint)

        dest = os.path.join(cls().tags_basepath, f"{model_name}-cat-vocab.json")
        ComfyLogger().log(
            f"Downloading DINOv3 categorised vocab from {cat_vocab_url}",
            type="INFO", always=True,
        )

        try:
            response = requests.get(cat_vocab_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None
            with open(dest, "wb") as f, tqdm(
                desc=f"{model_name}-cat-vocab",
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
                    if pbar is not None:
                        pbar.update_absolute(downloaded, total_size)

            return True

        except Exception as e:
            ComfyLogger().log(
                f"Failed to download DINOv3 categorised vocab: {e}\n"
                f"{traceback.format_exc()}",
                type="WARNING", always=True,
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

            pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None
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
                    if pbar is not None:
                        pbar.update_absolute(downloaded, total_size)

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
        use_aliases: bool = False,
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
        use_aliases:
            If ``True``, use aliased tag names from tag2alias in the output.
        """
        metadata = ComfyCache.get(f"tags_dino.{model_name}.metadata")
        tag2category: Optional[Dict[str, int]] = cls.get_tag2category(model_name)

        # Build exclusion sets (normalised to lowercase spaces).
        # Entries containing '*' are glob-style wildcard patterns;
        # everything else is an exact-match entry.
        excluded_exact: Set[str] = set()
        excluded_wildcards: List[str] = []
        if exclude_tags:
            for t in exclude_tags.split(","):
                t = t.strip()
                if not t:
                    continue
                normalised = t.replace("_", " ").lower()
                if "*" in normalised:
                    excluded_wildcards.append(normalised)
                else:
                    excluded_exact.add(normalised)

        # Parse excluded categories — works with tag2category (preferred)
        # or CSV metadata (fallback).  No dependency on implications.
        excluded_cats: Set[int] = set()
        if exclude_categories:
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

            # Check exclusion by name (exact match or wildcard)
            tag_check = tag.replace("_", " ").lower()
            if tag_check in excluded_exact:
                continue
            if excluded_wildcards and any(fnmatchcase(tag_check, p) for p in excluded_wildcards):
                continue

            # Check exclusion by category — prefer tag2category from cat-vocab,
            # fall back to CSV metadata tuple[0] (category ID).
            if excluded_cats:
                cat_id: Optional[int] = None
                if tag2category is not None and tag in tag2category:
                    cat_id = tag2category[tag]
                elif metadata and tag in metadata:
                    cat_id = metadata[tag][0]
                if cat_id is not None and cat_id in excluded_cats:
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
            display_tag = tag
            if use_aliases:
                tag2alias = cls.get_tag2alias(model_name)
                if tag2alias and tag in tag2alias:
                    display_tag = tag2alias[tag]
            tag_list.append(display_tag)

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
