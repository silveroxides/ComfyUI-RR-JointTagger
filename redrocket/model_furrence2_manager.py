import gc
import os
import traceback
from typing import Optional, Tuple, List, Any
import requests
import torch
import comfy.model_management as mm
import comfy.utils
from tqdm import tqdm

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

class Furrence2ModelManager(metaclass=Singleton):
    """
    The Furrence2 Model Manager class handles the loading, unloading, and strict local downloading
    of the Florence-2 model files.
    """
    def __init__(self, model_basepath: str) -> None:
        self.model_basepath = model_basepath
        ComfyCache.set_max_size('model_furrence2', 1)
        ComfyCache.set_cachemethod('model_furrence2', CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush('model_furrence2')
        gc.collect()

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        return ComfyCache.get(f'model_furrence2.{model_name}') is not None

    @classmethod
    def is_installed(cls, model_name: str) -> bool:
        """
        Check if the base model files are present in the local directory.
        """
        model_dir = os.path.join(cls().model_basepath, model_name)
        if not os.path.exists(model_dir):
            return False

        # Check for essential files indicating a successful download
        required_files = ["config.json", "model.safetensors"]
        return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

    @classmethod
    def load(cls, model_name: str, device: torch.device = torch.device('cpu')) -> bool:
        """
        Load a Furrence-2 model via a custom pipeline to bypass Transformers API versioning issues.
        """
        try:
            from .furrence2_pipeline import Furrence2Pipeline
        except ImportError as e:
            ComfyLogger().log(f"Required local modules for custom pipeline not found: {e}", "ERROR", True)
            return False

        if cls.is_loaded(model_name):
            ComfyLogger().log(f"Furrence2 Model {model_name} already loaded", "WARNING", True)
            return True

        if not cls.is_installed(model_name):
            ComfyLogger().log(f"Model {model_name} is not installed locally. Initiating download.", "INFO", True)
            success = cls.download(model_name)
            if not success:
                ComfyLogger().log(f"Failed to download model {model_name}.", "ERROR", True)
                return False

        model_dir = os.path.join(cls().model_basepath, model_name)
        ComfyLogger().log(f"Loading Furrence2 model {model_name} using custom pipeline from {model_dir}...", "INFO", True)

        try:
            # Instantiate the custom pipeline
            pipeline = Furrence2Pipeline(model_dir=model_dir, device=device)

            ComfyCache.set(f'model_furrence2.{model_name}', {
                "pipeline": pipeline,
                "device": device
            })
            ComfyLogger().log(f"Furrence2 Custom Pipeline loaded successfully for {model_name}", "INFO", True)
            return True

        except Exception as err:
            ComfyLogger().log(f"Error loading Furrence2 pipeline {model_name}: {err}\n{traceback.format_exc()}", "ERROR", True)
            return False

    @classmethod
    def get_model(cls, model_name: str) -> Optional[Any]:
        if not cls.is_loaded(model_name):
            return None
        data = ComfyCache.get(f'model_furrence2.{model_name}')
        return data["pipeline"]

    @classmethod
    def unload(cls, model_name: str) -> bool:
        if not cls.is_loaded(model_name):
            return True
        ComfyCache.flush(f'model_furrence2.{model_name}')
        gc.collect()
        mm.soft_empty_cache()
        return True

    @classmethod
    def download(cls, model_name: str) -> bool:
        """
        Download a Furrence-2 model and its associated configs using the endpoints defined in config.json.
        """
        model_dir = os.path.join(cls().model_basepath, model_name)
        tags_dir = os.path.join(os.path.dirname(cls().model_basepath), "tags", model_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tags_dir, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        model_config = config.get("models_furrence", {}).get(model_name)
        if not model_config:
            ComfyLogger().log(f"Configuration for furrence model '{model_name}' not found in config.json.", "ERROR", True)
            return False

        all_success = True

        # 1. Download Model Files
        base_url: str = model_config.get("url")
        if base_url:
            base_url = base_url.replace("{HF_ENDPOINT}", hf_endpoint)
            if not base_url.endswith("/"):
                base_url += "/"
            
            files_to_download: List[str] = model_config.get("files", [])
            for file_name in files_to_download:
                if not cls._download_file(f"{base_url}{file_name}", os.path.join(model_dir, file_name)):
                    all_success = False

        # 2. Download Tag Files from separate repo
        tags_url: str = model_config.get("tags_url")
        if tags_url:
            tags_url = tags_url.replace("{HF_ENDPOINT}", hf_endpoint)
            if not tags_url.endswith("/"):
                tags_url += "/"
            
            tag_files = model_config.get("tags_files", [])
            for file_name in tag_files:
                if not cls._download_file(f"{tags_url}{file_name}", os.path.join(tags_dir, file_name)):
                    all_success = False

        return all_success

    @classmethod
    def _download_file(cls, url: str, dest_path: str) -> bool:
        file_name = os.path.basename(dest_path)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return True

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None
            with open(dest_path, "wb") as f, tqdm(
                desc=file_name,
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
        except Exception as err:
            ComfyLogger().log(f"Unable to download {file_name}: {err}", "ERROR", True)
            return False
