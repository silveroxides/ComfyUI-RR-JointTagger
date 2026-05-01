"""DINOv3 Tagger model manager — download, load, cache."""

from __future__ import annotations

import gc
import hashlib
import os
import traceback
from typing import List, Optional, Tuple

import requests
import torch
from unifiedefficientloader import UnifiedSafetensorsLoader
from tqdm import tqdm
import comfy.utils

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

from .dinov3_backbone import DINOv3TaggerModel, build_head_from_checkpoint


class DINOv3ModelManager(metaclass=Singleton):
    """Singleton manager for DINOv3 Tagger model files.

    Handles downloading .safetensors from HuggingFace, loading into
    ``DINOv3TaggerModel``, and caching via ``ComfyCache`` under the
    ``model_dino`` namespace.
    """

    def __init__(self, model_basepath: str) -> None:
        self.model_basepath = model_basepath
        ComfyCache.set_max_size("model_dino", 1)
        ComfyCache.set_cachemethod("model_dino", CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush("model_dino")
        gc.collect()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_and_clean_state_dict(sd: dict) -> Tuple[dict, dict]:
        """Split full state dict into (backbone_sd, head_sd), stripping the
        ``backbone.`` prefix and applying the remaps needed to match
        ``DINOv3ViTH``'s parameter layout:

          1. ``backbone.model.layer.N.*`` → ``layer.N.*``
             (the checkpoint has an HF-style intermediate ``model`` wrapper
             that our flat backbone class does not)
          2. ``...layer_scale{1,2}.lambda1`` → ``...layer_scale{1,2}``
             (HF stores layer_scale as a sub-module with a ``lambda1``
             parameter; we use a plain ``nn.Parameter``)
          3. Drop any ``rope_embeddings`` buffers (recomputed on the fly)
        """
        backbone_sd: dict = {}
        head_sd: dict = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                nk = k[len("backbone."):]
                # Remap (1): strip intermediate "model." before "layer."
                if nk.startswith("model.layer."):
                    nk = nk[len("model."):]
                backbone_sd[nk] = v
            else:
                head_sd[k] = v

        # Remap (2): layer.N.layer_scale{1,2}.lambda1 → layer.N.layer_scale{1,2}
        for k in list(backbone_sd.keys()):
            if ".layer_scale" in k and k.endswith(".lambda1"):
                backbone_sd[k[:-len(".lambda1")]] = backbone_sd.pop(k)

        # Remap (3): drop rope buffers (recomputed on the fly)
        for k in list(backbone_sd.keys()):
            if "rope_embeddings" in k:
                backbone_sd.pop(k)

        return backbone_sd, head_sd

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        model_data = ComfyCache.get(f"model_dino.{model_name}")
        return (
            model_data is not None
            and isinstance(model_data, dict)
            and model_data.get("model") is not None
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
    # Update check helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_local_sha256(file_path: str) -> Optional[str]:
        """Compute the SHA-256 hex digest of a local file.

        Reads in 64 KiB chunks to keep memory usage low for large
        ``.safetensors`` files.

        Returns ``None`` if the file does not exist or an I/O error occurs.
        """
        try:
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65_536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as exc:
            ComfyLogger().log(
                f"Failed to compute SHA-256 for {file_path}: {exc}",
                type="WARNING", always=True,
            )
            return None

    @classmethod
    def fetch_remote_sha256(cls, model_name: str) -> Optional[str]:
        """Fetch the SHA-256 (``etag``) of the remote model from HuggingFace.

        Sends a lightweight request with the ``application/vnd.xet-fileinfo+json``
        accept header and ``Range: bytes=0-0`` so that HuggingFace returns a
        small JSON payload containing the ``etag`` field instead of streaming
        the full file.

        Returns the SHA-256 hex string, or ``None`` on any failure.
        """
        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config.get("huggingface_endpoint", "https://huggingface.co")
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        hf_endpoint = hf_endpoint.rstrip("/")

        url: str = ComfyExtensionConfig().get(
            property=f"models_dino.{model_name}.url",
        )
        if not url:
            ComfyLogger().log(
                f"No URL configured for DINOv3 model {model_name}",
                type="WARNING", always=True,
            )
            return None
        url = url.replace("{HF_ENDPOINT}", hf_endpoint)

        try:
            resp = requests.get(
                url,
                headers={
                    "Accept": "application/vnd.xet-fileinfo+json, */*",
                    "Range": "bytes=0-0",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            etag: str = data.get("etag", "")
            # The etag value is returned with surrounding quotes, strip them.
            return etag.strip('"') if etag else None
        except Exception as exc:
            ComfyLogger().log(
                f"Failed to fetch remote SHA-256 for {model_name}: {exc}",
                type="WARNING", always=True,
            )
            return None

    @classmethod
    def check_for_update(cls, model_name: str) -> bool:
        """Compare the local model SHA-256 against the remote repository.

        This check is guarded by a per-model ``update_checked`` flag in
        ``ComfyCache`` so that it only runs **once per ComfyUI session**
        (the cache is cleared on restart).

        Returns ``True`` if the model was re-downloaded, ``False`` otherwise.
        """
        # Once-per-session guard
        if ComfyCache.get(f"model_dino_update.{model_name}"):
            return False

        # Mark as checked regardless of outcome so we don't retry every run.
        ComfyCache.set(f"model_dino_update.{model_name}", True)

        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")
        if not os.path.exists(model_path):
            # Not installed yet — nothing to compare; download will happen later.
            return False

        ComfyLogger().log(
            f"Checking for DINOv3 model updates ({model_name}) …",
            type="INFO", always=True,
        )

        local_hash = cls.compute_local_sha256(model_path)
        remote_hash = cls.fetch_remote_sha256(model_name)

        if local_hash is None or remote_hash is None:
            ComfyLogger().log(
                "Unable to compare model hashes — skipping update check",
                type="WARNING", always=True,
            )
            return False

        if local_hash == remote_hash:
            ComfyLogger().log(
                f"DINOv3 model {model_name} is up-to-date (SHA-256 match)",
                type="INFO", always=True,
            )
            return False

        ComfyLogger().log(
            f"DINOv3 model {model_name} update detected "
            f"(local={local_hash[:12]}… remote={remote_hash[:12]}…) — re-downloading",
            type="INFO", always=True,
        )

        # Evict the loaded model from cache so it will be reloaded after download.
        if cls.is_loaded(model_name):
            ComfyCache.set(f"model_dino.{model_name}", None)

        return cls.download(model_name)

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
            # Load full checkpoint to CPU first
            with UnifiedSafetensorsLoader(model_path, low_memory=True) as loader:
                sd = {key: loader.get_tensor(key) for key in loader.keys()}

            # Split and clean state dict
            backbone_sd, head_sd = cls._split_and_clean_state_dict(sd)

            if not head_sd:
                raise RuntimeError(
                    "Checkpoint contains no non-backbone keys — cannot build head.")

            # Build model with dynamically-determined head architecture
            model = DINOv3TaggerModel()
            head_module, head_sd_remapped = build_head_from_checkpoint(
                head_sd, num_tags=num_tags,
            )
            model.head = head_module

            # Strict load both parts
            model.backbone.load_state_dict(backbone_sd, strict=True)
            model.head.load_state_dict(head_sd_remapped, strict=True)

            # Configure dtype: backbone in bf16/fp16, head stays fp32
            use_dtype = torch.bfloat16
            if device.type == "cuda":
                cap = torch.cuda.get_device_capability()
                if cap[0] < 8:
                    use_dtype = torch.float16

            model.backbone = model.backbone.to(dtype=use_dtype)
            model = model.to("cpu")
            model.eval()

            # Cache the model with all metadata in a single dict (stored on CPU)
            ComfyCache.set(f"model_dino.{model_name}", {
                "model": model,
                "device": torch.device("cpu"),
                "dtype": use_dtype,
            })

            ComfyLogger().log(
                f"DINOv3 model {model_name} loaded on CPU ({use_dtype})",
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
                f"Unable to download DINOv3 model: {err}\n{traceback.format_exc()}",
                type="ERROR", always=True,
            )
            return False
