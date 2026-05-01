import gc
import os
import traceback
from typing import List, Union
import requests
import torch
import timm
from unifiedefficientloader import UnifiedSafetensorsLoader
from tqdm import tqdm
import comfy.model_management as mm
import comfy.utils

from ..helpers.cache import CacheCleanupMethod, ComfyCache
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
    def __init__(self, model_basepath: str) -> None:
        self.model_basepath = model_basepath
        ComfyCache.set_max_size('model', 1)
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
        with UnifiedSafetensorsLoader(model_path, low_memory=True) as loader:
            sd = {key: loader.get_tensor(key) for key in loader.keys()}
        model.load_state_dict(sd)
        model.to(device="cpu", dtype=torch.float32)
        model.eval()
        ComfyCache.set(f'model.{model_name}', {
            "model": model,
            "version": version,
            "device": torch.device('cpu')
        })
        ComfyLogger().log(message=f"Model {model_name} loaded successfully on CPU", type="INFO", always=True)
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
        mm.soft_empty_cache()
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
        os.makedirs(cls().model_basepath, exist_ok=True)

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
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None
            with open(model_path, "wb") as f, tqdm(
                desc=model_name,
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
            ComfyLogger().log(
                message=f"Unable to download model. Download files manually or try using a HF mirror/proxy in your config.json: {err}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False
