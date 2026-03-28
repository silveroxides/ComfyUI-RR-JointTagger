"""DINOv3 Tagger vocabulary manager — download, load, process tags."""

from __future__ import annotations

import gc
import json
import os
import re
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton


class DINOv3TagManager(metaclass=Singleton):
    """Singleton manager for DINOv3 tagger vocabulary.

    The vocabulary is a JSON file with ``{"idx2tag": ["tag0", "tag1", ...]}``.
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
    def get_idx2tag(cls, model_name: str) -> Optional[List[str]]:
        return ComfyCache.get(f"tags_dino.{model_name}.idx2tag")

    @classmethod
    def get_num_tags(cls, model_name: str) -> int:
        idx2tag = cls.get_idx2tag(model_name)
        return len(idx2tag) if idx2tag else 0

    # ------------------------------------------------------------------
    # Load
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
    # Download
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
    # Tag processing
    # ------------------------------------------------------------------

    @classmethod
    def process_tags(
        cls,
        results: List[Tuple[str, float]],
        exclude_tags: str = "",
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        prefix: str = "",
    ) -> Tuple[str, Dict[str, float]]:
        """Format raw (tag, score) list into a tag string and scores dict.

        Parameters
        ----------
        results:
            List of ``(tag, score)`` tuples, already sorted by descending score.
        exclude_tags:
            Comma-separated tags to exclude.
        replace_underscore:
            Replace underscores with spaces in output tags.
        trailing_comma:
            Append a trailing comma to the tag string.
        prefix:
            Text to prepend.
        """
        # Build exclusion set (normalised to spaces)
        excluded = set()
        if exclude_tags:
            for t in exclude_tags.split(","):
                t = t.strip()
                if t:
                    excluded.add(t.replace("_", " ").lower())

        tag_list: List[str] = []
        scores_dict: Dict[str, float] = {}

        for tag, score in results:
            # Normalised check for exclusion
            tag_check = tag.replace("_", " ").lower()
            if tag_check in excluded:
                continue

            scores_dict[tag] = round(score, 4)

            display_tag = tag
            if replace_underscore:
                display_tag = display_tag.replace("_", " ")
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
