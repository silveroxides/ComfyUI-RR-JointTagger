import gc
import os
from typing import Callable, List, Optional, Union, Tuple, Dict, Any, Set
import requests
import csv
from tqdm import tqdm
from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

TAG_CATEGORIES = {
    "general": 0,
    # "artist": 1,
    "copyright": 3,
    "character": 4,
    "species": 5,
    "meta": 7,
    "lore": 8,
}

class JtpTagV3Manager(metaclass=Singleton):
    """
    Manager for JTP-3 CSV tags and metadata.
    """
    def __init__(self, tags_basepath: str, download_progress_callback: Callable[[int, int], None], download_complete_callback: Optional[Callable[[str], None]] = None) -> None:
        self.tags_basepath = tags_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        ComfyCache.set_max_size('tags_v3', 1)
        ComfyCache.set_cachemethod('tags_v3', CacheCleanupMethod.ROUND_ROBIN)
    
    def __del__(self) -> None:
        ComfyCache.flush('tags_v3')
        gc.collect()

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        # We assume tag file name matches model name suffix or is configured
        # Actually JTP-3 seems to have a specific tag file "jtp-3-hydra-tags.csv"
        # We'll use the model name to look up the data config
        return ComfyCache.get(f'tags_v3.{model_name}') is not None

    @classmethod
    def load(cls, model_name: str) -> bool:
        tags_filename = f"{model_name}-tags.csv" # Standardize name
        # But wait, config has the url.
        # Let's assume we downloaded it as {model_name}-tags.csv
        
        # Actually the user message said: "data/jtp-3-hydra-tags.csv"
        # So we should probably stick to that naming or use the config to map it.
        # For simplicity, I'll name the file on disk based on the model name key in config + "-tags.csv"
        
        tags_path = os.path.join(cls().tags_basepath, tags_filename)
        
        if not os.path.exists(tags_path):
             ComfyLogger().log(f"Tags file not found: {tags_path}", "ERROR", True)
             return False

        try:
            metadata = {}
            with open(tags_path, "r", encoding="utf-8", newline="") as metadata_file:
                reader = csv.DictReader(metadata_file)
                for row in reader:
                    # Parse implications
                    implications = [tag for tag in row.get("implications", "").split()]
                    category = int(row.get("category", 0))
                    tag = row["tag"]
                    metadata[tag] = (category, implications)
            
            ComfyCache.set(f'tags_v3.{model_name}', metadata)
            return True
        except Exception as e:
            ComfyLogger().log(f"Error loading tags: {e}", "ERROR", True)
            return False

    @classmethod
    def download(cls, model_name: str) -> bool:
        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config["huggingface_endpoint"]
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        if hf_endpoint.endswith("/"):
            hf_endpoint = hf_endpoint.rstrip("/")
            
        data_url = ComfyExtensionConfig().get(property=f"models_v3.{model_name}.data_url")
        if not data_url:
             ComfyLogger().log(f"No data URL for {model_name}", "ERROR", True)
             return False
             
        data_url = data_url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not data_url.endswith("/"):
            data_url += "/"
            
        # We need to download jtp-3-hydra-tags.csv
        # The user provided URL is for the *folder*.
        # We'll assume the file is named "{model_name}-tags.csv" on the server?
        # User said: "jtp-3-hydra-tags.csv" and "jtp-3-hydra-val.csv" are in that folder.
        # If our model key is "jtp-3-hydra", we can infer the filename.
        
        files_to_download = [f"{model_name}-tags.csv"] # We assume the key matches the file prefix
        
        for fname in files_to_download:
            url = f"{data_url}{fname}"
            dest = os.path.join(cls().tags_basepath, fname)
            try:
                ComfyLogger().log(f"Downloading {fname} from {url}", "INFO", True)
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                
                with open(dest, 'wb') as file, tqdm(
                    desc=fname,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    downloaded = 0
                    for data in response.iter_content(block_size):
                        file.write(data)
                        bar.update(len(data))
                        downloaded += len(data)
                        if cls().download_progress_callback and total_size > 0:
                            cls().download_progress_callback(int(downloaded / total_size * 100), fname)
                            
                cls().download_complete_callback(fname)
            except Exception as e:
                ComfyLogger().log(f"Failed to download {fname}: {e}", "ERROR", True)
                return False
        return True

    @classmethod
    def process_tags(cls, 
                     predictions: Dict[str, float], 
                     model_name: str,
                     threshold: float,
                     implications_mode: str = "inherit",
                     exclude_categories: List[str] = []) -> Tuple[str, Dict[str, float]]:
        
        metadata = ComfyCache.get(f'tags_v3.{model_name}')
        if not metadata:
            return "", {}

        # Convert exclude categories strings to ints
        exclude_ids = {TAG_CATEGORIES[cat] for cat in exclude_categories if cat in TAG_CATEGORIES}

        labels = predictions.copy()

        def inherit_implications(labels, antecedent):
            if antecedent not in metadata: return
            p = labels.get(antecedent, 0.0)
            for consequent in metadata[antecedent][1]:
                if labels.get(consequent, 0.0) < p:
                    labels[consequent] = p
                inherit_implications(labels, consequent)

        def constrain_implications(labels, antecedent, _target=None):
            if _target is None: _target = antecedent
            if antecedent not in metadata: return
            for consequent in metadata[antecedent][1]:
                p = labels.get(consequent, 0.0)
                if labels.get(_target, 0.0) > p:
                    labels[_target] = p
                constrain_implications(labels, consequent, _target=_target)

        def remove_implications(labels, antecedent):
            if antecedent not in metadata: return
            for consequent in metadata[antecedent][1]:
                labels.pop(consequent, None)
                remove_implications(labels, consequent)

        # Apply implications
        tags = list(labels.keys())
        
        if implications_mode == "inherit":
            for tag in tags:
                inherit_implications(labels, tag)
        elif implications_mode in ["constrain", "constrain-remove"]:
            for tag in tags:
                constrain_implications(labels, tag)
        
        # Filter by threshold and categories
        filtered_labels = {}
        for tag, prob in labels.items():
            if prob < threshold:
                continue
            
            if tag in metadata:
                cat_id = metadata[tag][0]
                if cat_id in exclude_ids:
                    continue
            
            filtered_labels[tag] = prob

        # Remove implications step
        if implications_mode in ["remove", "constrain-remove"]:
            tags_to_check = list(filtered_labels.keys())
            for tag in tags_to_check:
                if tag in filtered_labels:
                    remove_implications(filtered_labels, tag)

        # Format output
        # Sort by score
        sorted_items = sorted(filtered_labels.items(), key=lambda x: x[1], reverse=True)
        final_tags = {k: v for k, v in sorted_items}
        
        tag_string = ", ".join(final_tags.keys())
        return tag_string, final_tags
