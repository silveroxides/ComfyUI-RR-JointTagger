"""DINOv3 Tagger model manager — download, load, cache."""

from __future__ import annotations

import gc
import os
import traceback
from typing import Callable, List, Optional, Tuple

import requests
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

from .dinov3_backbone import DINOv3TaggerModel


class DINOv3ModelManager(metaclass=Singleton):
    """Singleton manager for DINOv3 Tagger model files.

    Handles downloading .safetensors from HuggingFace, loading into
    ``DINOv3TaggerModel``, and caching via ``ComfyCache`` under the
    ``model_dino`` namespace.
    """

    def __init__(
        self,
        model_basepath: str,
        download_progress_callback: Callable[[int, str], None],
        download_complete_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model_basepath = model_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        ComfyCache.set_max_size("model_dino", 1)
        ComfyCache.set_cachemethod("model_dino", CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush("model_dino")
        gc.collect()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        return (
            ComfyCache.get(f"model_dino.{model_name}") is not None
            and ComfyCache.get(f"model_dino.{model_name}.model") is not None
        )

    @classmethod
    def is_installed(cls, model_name: str) -> bool:
        return any(model_name + ".safetensors" in s for s in cls.list_installed())

    @classmethod
    def list_installed(cls) -> List[str]:
        model_path = os.path.abspath(cls().model_basepath)
        if not os.path.exists(model_path):
            ComfyLogger().log(
                f"Model path {model_path} does not exist, creating it",
                type="WARN", always=True,
            )
            os.makedirs(model_path, exist_ok=True)
            return []
        return [f for f in os.listdir(model_path) if f.endswith(".safetensors")]

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_name: str,
        num_tags: int,
        device: torch.device = torch.device("cpu"),
    ) -> bool:
        """Load a DINOv3 Tagger checkpoint into memory.

        Parameters
        ----------
        model_name:
            Config key (e.g. ``"tagger-proto"``).
        num_tags:
            Number of tags in the vocabulary — required to instantiate the
            projection head.
        device:
            Target device.
        """
        if cls.is_loaded(model_name):
            ComfyLogger().log(
                f"DINOv3 model {model_name} already loaded",
                type="WARNING", always=True,
            )
            return True

        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")
        if not os.path.exists(model_path):
            ComfyLogger().log(
                f"DINOv3 model {model_name} not found at {model_path}",
                type="ERROR", always=True,
            )
            return False

        ComfyLogger().log(
            f"Loading DINOv3 model {model_name} from {model_path} …",
            type="INFO", always=True,
        )

        try:
            sd = load_file(model_path, device="cpu")

            model = DINOv3TaggerModel(num_tags=num_tags)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            if missing:
                ComfyLogger().log(
                    f"DINOv3 missing keys ({len(missing)}): {missing[:5]}"
                    + ("…" if len(missing) > 5 else ""),
                    type="WARNING", always=True,
                )
            if unexpected:
                ComfyLogger().log(
                    f"DINOv3 unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                    + ("…" if len(unexpected) > 5 else ""),
                    type="WARNING", always=True,
                )

            # Backbone in bfloat16 (or float16 for older GPUs), projection stays float32
            use_dtype = torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] < 8:
                    use_dtype = torch.float16

            model.backbone = model.backbone.to(dtype=use_dtype)
            model = model.to(device)
            model.eval()

            ComfyCache.set(f"model_dino.{model_name}", {
                "model": model,
                "device": device,
                "dtype": use_dtype,
            })
            ComfyLogger().log(
                f"DINOv3 model {model_name} loaded on {device} ({use_dtype})",
                type="INFO", always=True,
            )
            return True

        except Exception as e:
            ComfyLogger().log(
                f"Error loading DINOv3 model {model_name}: {e}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @classmethod
    def download(cls, model_name: str) -> bool:
        os.makedirs(cls().model_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        url: str = ComfyExtensionConfig().get(property=f"models_dino.{model_name}.url")
        if not url:
            ComfyLogger().log(
                f"No URL found for DINOv3 model {model_name} in config",
                type="ERROR", always=True,
            )
            return False
        url = url.replace("{HF_ENDPOINT}", hf_endpoint)

        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")
        ComfyLogger().log(
            f"Downloading DINOv3 model {model_name} from {url}",
            type="INFO", always=True,
        )

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

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
                    if cls().download_progress_callback and total_size > 0:
                        cls().download_progress_callback(
                            int(downloaded / total_size * 100), model_name,
                        )

            if cls().download_complete_callback:
                cls().download_complete_callback(model_name)
            return True

        except Exception as err:
            ComfyLogger().log(
                f"Unable to download DINOv3 model: {err}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False
