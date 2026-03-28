import csv
import gc
import os
import traceback
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import msgspec
import requests
import torch
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


class JtpTagManager(metaclass=Singleton):
    """
    The JTP Tag Manager class is a singleton class that manages the loading, unloading, downloading, and installation of JTP Vision Transformer tags.
    """
    def __init__(self, tags_basepath: str, download_progress_callback: Callable[[int, int], None], download_complete_callback: Optional[Callable[[str], None]] = None) -> None:
        self.tags_basepath = tags_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        ComfyCache.set_max_size('tags', 1)
        ComfyCache.set_cachemethod('tags', CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush('tags')
        gc.collect()

    @classmethod
    def is_loaded(cls, tags_name: str) -> bool:
        """
        Check if tags are loaded into memory
        """
        return ComfyCache.get(f'tags.{tags_name}') is not None and ComfyCache.get(f'tags.{tags_name}.tags') is not None

    @classmethod
    def load(cls, tags_name: str, version: int) -> bool:
        """
        Mount the tags for a model into memory
        """
        from ..helpers.logger import ComfyLogger

        tags_path = os.path.join(cls().tags_basepath, f"{tags_name}.json")
        if cls.is_loaded(tags_name):
            ComfyLogger().log(f"Tags for model {tags_name} already loaded", "WARNING", True)
            return True
        if not os.path.exists(tags_path):
            ComfyLogger().log(f"Tags for model {tags_name} not found in path: {tags_path}", "ERROR", True)
            return False

        # TODO: Add a check for the version of the tags file differs
        try:
            with open(tags_path, "r") as file:
                ComfyCache.set(f'tags.{tags_name}', {
                    "tags": msgspec.json.decode(file.read(), type=Dict[str, float], strict=False),
                    "version": version
                })
            count = len(ComfyCache.get(f'tags.{tags_name}.tags'))
            ComfyLogger().log(f"Loaded {count} tags for model {tags_name}", "INFO", True)
            return True
        except Exception as err:
            ComfyLogger().log(f"Error loading tags for model {tags_name}: {err}\n{traceback.format_exc()}", "ERROR", True)
            return False

    @classmethod
    def unload(cls, tags_name: str) -> bool:
        """
        Unmount the tags for a model from memory
        """
        from ..helpers.logger import ComfyLogger

        if not cls.is_loaded(tags_name):
            ComfyLogger().log(f"Tags for model {tags_name} not loaded, nothing to do here", "WARNING", True)
            return True
        ComfyCache.flush(f'tags.{tags_name}')
        gc.collect()
        return True

    @classmethod
    def is_installed(cls, tags_name: str) -> bool:
        """
        Check if a tags file is installed in a directory
        """
        return any(tags_name + ".json" in s for s in cls.list_installed())

    @classmethod
    def list_installed(cls) -> List[str]:
        """
        Get a list of installed tags files in a directory
        """
        from ..helpers.logger import ComfyLogger
        tags_path = os.path.abspath(cls().tags_basepath)
        if not os.path.exists(tags_path):
            ComfyLogger().log(f"Tags path {tags_path} does not exist, it is being created", "WARN", True)
            os.makedirs(os.path.abspath(tags_path))
            return []
        tags = list(filter(
            lambda x: x.endswith(".json"), os.listdir(tags_path)))
        return tags

    @classmethod
    def download(cls, tags_name: str) -> bool:
        """
        Download tags for a model from a URL and save them to a file.
        """
        os.makedirs(cls().tags_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config["huggingface_endpoint"]
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        if hf_endpoint.endswith("/"):
            hf_endpoint = hf_endpoint.rstrip("/")
        tags_path = os.path.join(cls().tags_basepath, f"{tags_name}.json")

        url: str = ComfyExtensionConfig().get(property=f"tags.{tags_name}.url")
        url = url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not url.endswith("/"):
            url += "/"
        if not url.endswith(".json"):
            url += f"tags.json"

        ComfyLogger().log(f"Downloading tags {tags_name} from {url}", "INFO", True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with open(tags_path, "wb") as f, tqdm(
                desc=f"{tags_name}.json",
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
                            f"{tags_name}.json",
                        )

            if cls().download_complete_callback:
                cls().download_complete_callback(tags_name)
            return True
        except Exception as err:
            ComfyLogger().log(
                f"Unable to download tags. Download files manually or try using a HF mirror/proxy in your config.json: {err}\n{traceback.format_exc()}",
                "ERROR", True,
            )
            return False

    # ------------------------------------------------------------------
    # Metadata (implications / categories from JTP-3 CSV)
    # ------------------------------------------------------------------

    @classmethod
    def has_metadata(cls, tags_name: str) -> bool:
        """Check whether metadata (implications CSV) is loaded for *tags_name*."""
        return ComfyCache.get(f'tags.{tags_name}.metadata') is not None

    @classmethod
    def load_metadata(cls, tags_name: str) -> bool:
        """Load tag metadata (categories + implications) from a JTP-3-style CSV.

        All tag keys and implication references are normalised to use spaces
        instead of underscores to match V1/V2 in-memory tag format.
        """
        csv_path = os.path.join(cls().tags_basepath, f"{tags_name}-tags.csv")
        if not os.path.exists(csv_path):
            ComfyLogger().log(
                f"Tag metadata CSV not found: {csv_path} "
                "(implications will be unavailable)",
                "WARNING", True,
            )
            return False

        try:
            metadata: Dict[str, Tuple[int, List[str]]] = {}
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Normalise tag: underscores -> spaces (V1 stores with spaces)
                    tag = row["tag"].replace("_", " ")
                    category = int(row.get("category", 0))
                    raw_impl = row.get("implications", "")
                    implications = [
                        imp.replace("_", " ")
                        for imp in raw_impl.split()
                        if imp
                    ]
                    metadata[tag] = (category, implications)

            ComfyCache.set(f'tags.{tags_name}.metadata', metadata)
            ComfyLogger().log(
                f"Tag metadata loaded: {len(metadata):,} entries for {tags_name}",
                "INFO", True,
            )
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Error loading tag metadata: {e}\n{traceback.format_exc()}",
                "ERROR", True,
            )
            return False

    @classmethod
    def download_metadata(cls, model_name: str) -> bool:
        """Download tag metadata CSV from the configured ``data_url``.

        The model_name here is the key used in config.json ``models`` section.
        The CSV is saved as ``{tags_basepath}/{model_name}-tags.csv``.
        """
        os.makedirs(cls().tags_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        data_url = ComfyExtensionConfig().get(
            property=f"models.{model_name}.data_url",
        )
        if not data_url:
            ComfyLogger().log(
                f"No data_url for model {model_name} — "
                "implications will not be available",
                "WARNING", True,
            )
            return False
        data_url = data_url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not data_url.endswith("/"):
            data_url += "/"

        tag_file = ComfyExtensionConfig().get(
            property=f"models.{model_name}.data_tag_file",
        )
        if not tag_file or not isinstance(tag_file, str):
            tag_file = "jtp-3-hydra-tags.csv"

        url = f"{data_url}{tag_file}"
        dest = os.path.join(cls().tags_basepath, f"{model_name}-tags.csv")

        ComfyLogger().log(f"Downloading tag metadata from {url}", "INFO", True)

        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                ComfyLogger().log(
                    f"Failed to download tag metadata: HTTP {response.status_code}",
                    "WARNING", True,
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
                f"Failed to download tag metadata: {e}\n{traceback.format_exc()}",
                "WARNING", True,
            )
            return False

    # ------------------------------------------------------------------
    # Tag processing (with optional implications)
    # ------------------------------------------------------------------

    @classmethod
    def process_tags(
        cls,
        tags_name: str,
        indices: Union[torch.Tensor, None],
        values: Union[torch.Tensor, None],
        threshold: float,
        exclude_tags: str = "",
        replace_underscore: bool = True,
        trailing_comma: bool = False,
        implications_mode: str = "off",
        exclude_categories: str = "",
        prefix: str = "",
    ) -> Tuple[str, Dict[str, float]]:
        """Process raw model output into a formatted tag string and scores dict.

        Parameters
        ----------
        tags_name:
            Cache key for the loaded tag vocabulary.
        indices, values:
            Top-k indices and values from the model output.
        threshold:
            Minimum score floor — also re-applied after implications.
        exclude_tags:
            Comma-separated tags to exclude.
        replace_underscore:
            If True tags are returned with spaces; if False with underscores.
        trailing_comma:
            Append a trailing comma to the tag string.
        implications_mode:
            ``"inherit"``, ``"constrain"``, ``"remove"``,
            ``"constrain-remove"``, or ``"off"``.
        exclude_categories:
            Comma-separated category names to exclude
            (e.g. ``"copyright, character, meta"``).
        prefix:
            Text to prepend to the tag string.
        """
        from ..helpers.logger import ComfyLogger

        tag_score: Dict[str, float] = {}

        if indices is None or indices.size(0) == 0:
            ComfyLogger().log(
                message=f"No indices found for model {tags_name}",
                type="WARNING", always=True,
            )
            return "", tag_score

        if values is None or values.size(0) == 0:
            ComfyLogger().log(
                message=f"No values found for model {tags_name}",
                type="WARNING", always=True,
            )
            return "", tag_score

        ComfyLogger().log(
            message=f"Processing {len(indices)}:{len(values)} tags for model {tags_name}",
            type="INFO", always=True,
        )

        tags_data: Union[Dict[str, float], None] = ComfyCache.get(f'tags.{tags_name}.tags')
        if tags_data is None or len(tags_data) == 0:
            ComfyLogger().log(
                message=f"Tags data for {tags_name} not found in cache",
                type="ERROR", always=True,
            )
            return "", tag_score

        # Build labels dict from top-k.
        # V1 JSON tags have underscores, V2 has spaces — normalise all to
        # spaces so they match the metadata (also normalised to spaces).
        for i in range(indices.size(0)):
            raw_tag = [key for key, value in tags_data.items() if value == indices[i].item()][0]
            tag = raw_tag.replace("_", " ")
            tag_score[tag] = values[i].item()

        # Apply initial threshold
        labels: Dict[str, float] = {
            tag: score for tag, score in tag_score.items() if score > threshold
        }

        # --- Implications processing (only if metadata is available) ---
        metadata = ComfyCache.get(f'tags.{tags_name}.metadata')

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

        # --- Filtering ---

        # Build exclusion set (normalised to spaces for matching)
        excluded: Set[str] = set()
        if exclude_tags:
            for t in exclude_tags.split(","):
                t = t.strip()
                if t:
                    excluded.add(t.replace("_", " "))

        # Parse excluded categories
        excluded_cats: Set[int] = set()
        if exclude_categories and metadata:
            for cat in exclude_categories.split(","):
                cat = cat.strip().lower()
                if cat in TAG_CATEGORIES:
                    excluded_cats.add(TAG_CATEGORIES[cat])

        # Re-apply threshold after implications + apply exclusions
        filtered: Dict[str, float] = {}
        for tag, prob in labels.items():
            # Threshold (catches scores lowered by constrain)
            if prob <= threshold:
                continue

            # Exclude by tag name
            if tag in excluded:
                continue

            # Exclude by category
            if excluded_cats and metadata and tag in metadata:
                cat_id = metadata[tag][0]
                if cat_id in excluded_cats:
                    continue

            filtered[tag] = prob

        # --- Remove implications step (after filtering) ---
        if metadata and implications_mode in ("remove", "constrain-remove"):
            keys_to_check = list(filtered.keys())
            for tag in keys_to_check:
                if tag in filtered and tag in metadata:
                    for consequent in metadata[tag][1]:
                        filtered.pop(consequent, None)
                        cls._remove_implications_recursive(filtered, consequent, metadata)

        # --- Format output ---
        sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        tag_list: List[str] = []
        scores_dict: Dict[str, float] = {}

        for tag, score in sorted_items:
            scores_dict[tag] = score
            display_tag = tag
            if not replace_underscore:
                display_tag = display_tag.replace(" ", "_")
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
            JtpTagManager._remove_implications_recursive(labels, consequent, metadata)
