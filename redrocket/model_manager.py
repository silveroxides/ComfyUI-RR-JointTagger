import gc
import os
import traceback
from typing import Callable, List, Optional, Union
from aiohttp import web
import aiohttp
import torch
import timm
import safetensors.torch

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.http import ComfyHTTP
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton


class V2GatedHead(torch.nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes * 2)
        self.act = torch.nn.Sigmoid()
        self.gate = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.act(x[:, :self.num_classes]) * self.gate(x[:, self.num_classes:])
        return x


class JtpModelManager(metaclass=Singleton):
    """
    The RedRocket JTP Model Manager class is a singleton class that manages the loading, unloading, downloading, and installation of JTP Vision Transformer models.
    """
    def __init__(self, model_basepath: str, download_progress_callback: Callable[[int, int], None], download_complete_callback: Optional[Callable[[str], None]] = None) -> None:
        self.model_basepath = model_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        ComfyCache.set_max_size('model', 1)  # Adjust the max size as needed
        ComfyCache.set_cachemethod('model', CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush('model')
        gc.collect()

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        """
        Check if a RedRocket JTP Vision Transformer model is loaded into memory
        """
        return ComfyCache.get(f'model.{model_name}') is not None and ComfyCache.get(f'model.{model_name}.model') is not None

    @classmethod
    def load(cls, model_name: str, version: int = 1, device: torch.device = torch.device('cpu')) -> bool:
        """
        Load a RedRocket JTP Vision Transformer model into memory
        """
        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")
        if cls.is_loaded(model_name):
            ComfyLogger().log(message=f"Model {model_name} already loaded", type="WARNING", always=True)
            return True
        if not os.path.exists(model_path):
            ComfyLogger().log(message=f"Model {model_name} not found in path: {model_path}",type= "ERROR", always=True)
            return False

        ComfyLogger().log(message=f"Loading model {model_name} (version: {version}) from {model_path}...", type="INFO", always=True)
        model: torch.nn.Module = timm.create_model("vit_so400m_patch14_siglip_384.webli", pretrained=False, num_classes=9083)
        if f"{version}" == "2":
            model.head = V2GatedHead(min(model.head.weight.shape), 9083)
        safetensors.torch.load_model(model=model, filename=model_path)
        if torch.cuda.is_available() and device.type == "cuda":
            model.cuda()
            model.to(dtype=torch.float16, memory_format=torch.channels_last)
        model.eval()
        ComfyCache.set(f'model.{model_name}', {
            "model": model,
            "version": version,
            "device": device
        })
        ComfyLogger().log(message=f"Model {model_name} loaded successfully", type="INFO", always=True)
        return True

    @classmethod
    def switch_device(cls, model_name: str, device: torch.device) -> bool:
        """
        Switch the device of a RedRocket JTP Vision Transformer model
        """
        if not cls.is_loaded(model_name):
            ComfyLogger().log(message=f"Model {model_name} not loaded, nothing to do here", type="WARNING", always=True)
            return False
        model: Union[torch.nn.Module, None] = ComfyCache.get(f'model.{model_name}.model')
        if model is None:
            ComfyLogger().log(message=f"Model {model_name} is not loaded, cannot switch it to another device", type="ERROR", always=True)
            return False
        if device.type == "cuda" and not torch.cuda.is_available():
            ComfyLogger().log(message="CUDA is not available, cannot switch to GPU", type="ERROR", always=True)
            return False
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
            model.cuda()
            model = model.to(dtype=torch.float16, memory_format=torch.channels_last)
            ComfyLogger().log(message="Switched to GPU with mixed precision", type="INFO", always=True)
        elif device.type == "cuda" and torch.cuda.get_device_capability()[0] < 7:
            model.cuda()
            model = model.to(dtype=torch.float32, memory_format=torch.channels_last)
            ComfyLogger().log(message="Switched to GPU without mixed precision", type="WARNING", always=True)
        else:
            model.cpu()
            model = model.to(device=device)
            ComfyLogger().log(message="Switched to CPU", type="INFO", always=True)
        ComfyCache.set(f'model.{model_name}.device', device)
        ComfyCache.set(f'model.{model_name}.model', model)
        return True

    @classmethod
    def unload(cls, model_name: str) -> bool:
        """
        Unload a RedRocket JTP Vision Transformer model from memory
        """
        if not cls.is_loaded(model_name):
            ComfyLogger().log(message=f"Model {model_name} not loaded, nothing to do here", type="WARNING", always=True)
            return True
        ComfyCache.flush(f'model.{model_name}')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    @classmethod
    def is_installed(cls, model_name: str) -> bool:
        """
        Check if a vision transformer model is installed in a directory
        """
        return any(model_name + ".safetensors" in s for s in cls.list_installed())

    @classmethod
    def list_installed(cls) -> List[str]:
        """
        Get a list of installed vision transformer models in a directory
        """
        model_path = os.path.abspath(cls().model_basepath)
        if not os.path.exists(model_path):
            ComfyLogger().log(message=f"Model path {model_path} does not exist, it is being created", type="WARN", always=True)
            os.makedirs(os.path.abspath(model_path))
            return []
        models = list(filter(
            lambda x: x.endswith(".safetensors"), os.listdir(model_path)))
        return models

    @classmethod
    def download(cls, model_name: str) -> bool:
        """
        Download a RedRocket JTP Vision Transformer model from a URL
        """
        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config["huggingface_endpoint"]
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        if hf_endpoint.endswith("/"):
            hf_endpoint = hf_endpoint.rstrip("/")

        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")

        url: str = ComfyExtensionConfig().get(property=f"models.{model_name}.url")
        url = url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not url.endswith("/"):
            url += "/"
        if not url.endswith(".safetensors"):
            url += f"{model_name}.safetensors"

        ComfyLogger().log(message=f"Downloading model {model_name} from {url}", type="INFO", always=True)
        with aiohttp.ClientSession() as session:
            try:
                ComfyHTTP().download_to_file(url=url, destination=model_path, update_callback=cls().download_progress_callback, session=session)
            except aiohttp.client_exceptions.ClientConnectorError as err:
                ComfyLogger().log(message=f"Unable to download model. Download files manually or try using a HF mirror/proxy in your config.json: {err}\n{traceback.format_exc()}", type="ERROR", always=True)
                return False
            except Exception as err:
                ComfyLogger().log(message=f"Error downloading model: {err}\n{traceback.format_exc()}", type="ERROR", always=True)
                return False
            cls().download_complete_callback(model_name)
        return True
