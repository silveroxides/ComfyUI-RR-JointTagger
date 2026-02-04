import gc
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
from aiohttp import web
import aiohttp
import msgspec
import torch

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.http import ComfyHTTP
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton


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
            ComfyLogger().log(f"Error loading tags for model {tags_name}: {err}", "ERROR", True)
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
        Load tags for a model from a URL and save them to a file.
        """
        from ..helpers.http import ComfyHTTP
        from ..helpers.config import ComfyExtensionConfig
        from ..helpers.logger import ComfyLogger
        
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
        with aiohttp.ClientSession() as session:
            try:
                ComfyHTTP().download_to_file(f"{url}", tags_path, cls().download_progress_callback, session=session)
            except aiohttp.client_exceptions.ClientConnectorError as err:
                ComfyLogger().log("Unable to download tags. Download files manually or try using a HF mirror/proxy in your config.json", "ERROR", True)
                return False
            cls().download_complete_callback(tags_name)
        return True

    @classmethod
    def process_tags(cls, tags_name: str, indices: Union[torch.Tensor, None], values: Union[torch.Tensor, None], exclude_tags: str, replace_underscore: bool, threshold: float, trailing_comma: bool) -> Tuple[str, Dict[str, float]]:
        from ..helpers.logger import ComfyLogger
        corrected_excluded_tags = [tag.replace("_", " ").strip() for tag in exclude_tags.split(",") if not tag.isspace()]
        print(corrected_excluded_tags)
        tag_score = {}
        if indices is None or indices.size(0) == 0:
            ComfyLogger().log(message=f"No indicies found for model {tags_name}", type="WARNING", always=True)
            return "", tag_score
        if values is None or values.size(0) == 0:
            ComfyLogger().log(message=f"No values found for model {tags_name}", type="WARNING", always=True)
            return "", tag_score
        ComfyLogger().log(message=f"Processing {len(indices)}:{len(values)} tags for model {tags_name}", type="INFO", always=True)
        tags_data: Union[Dict[str, float], None] = ComfyCache.get(f'tags.{tags_name}.tags')
        if tags_data is None or len(tags_data) == 0:
            ComfyLogger().log(message=f"Tags data for {tags_name} not found in cache", type="ERROR", always=True)
            return "", tag_score
        for i in range(indices.size(0)):
            tag = [key for key, value in tags_data.items() if value == indices[i].item()][0]
            if tag not in corrected_excluded_tags:
                tag_score[tag] = values[i].item()
        if not replace_underscore:
            tag_score = {key.replace(" ", "_"): value for key, value in tag_score.items()}
        tag_score = dict(sorted(tag_score.items(), key=lambda item: item[1], reverse=True))
        tag_score = {key: value for key, value in tag_score.items() if value > threshold}
        return ", ".join(tag_score.keys()) + ("," if trailing_comma else ""), tag_score
