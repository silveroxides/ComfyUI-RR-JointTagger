import gc
import os
from typing import Callable, List, Optional, Union, Tuple, Dict, Any
import requests
import torch
import timm
import safetensors.torch
from safetensors import safe_open
from torch.nn import Identity
from tqdm import tqdm

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

from .hydra_pool import HydraPool

def sdpa_attn_mask(
    patch_valid: torch.Tensor,
    num_prefix_tokens: int = 0,
    symmetric: bool = True,
    q_len: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    mask = patch_valid.unflatten(-1, (1, 1, -1))

    if num_prefix_tokens:
        mask = torch.cat((
            torch.ones(
                *mask.shape[:-1], num_prefix_tokens,
                device=patch_valid.device, dtype=torch.bool
            ), mask
        ), dim=-1)

    return mask

# Patch timm for NaFlex
timm.models.naflexvit.create_attention_mask = sdpa_attn_mask

class JtpModelV3Manager(metaclass=Singleton):
    """
    Manager for JTP-3 Hydra models.
    """
    def __init__(self, model_basepath: str, download_progress_callback: Callable[[int, int], None], download_complete_callback: Optional[Callable[[str], None]] = None) -> None:
        self.model_basepath = model_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        ComfyCache.set_max_size('model_v3', 1)
        ComfyCache.set_cachemethod('model_v3', CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush('model_v3')
        gc.collect()

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        return ComfyCache.get(f'model_v3.{model_name}') is not None and ComfyCache.get(f'model_v3.{model_name}.model') is not None

    @classmethod
    def load(cls, model_name: str, device: torch.device = torch.device('cpu')) -> Tuple[bool, List[str]]:
        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")

        if cls.is_loaded(model_name):
            ComfyLogger().log(message=f"Model {model_name} already loaded", type="WARNING", always=True)
            tags = ComfyCache.get(f'model_v3.{model_name}.tags')
            return True, tags

        if not os.path.exists(model_path):
            ComfyLogger().log(message=f"Model {model_name} not found in path: {model_path}", type="ERROR", always=True)
            return False, []

        ComfyLogger().log(message=f"Loading JTP-3 model {model_name} from {model_path}...", type="INFO", always=True)

        try:
            with safe_open(model_path, framework="pt", device="cpu") as file:
                metadata = file.metadata()
                state_dict = {key: file.get_tensor(key) for key in file.keys()}

            arch = metadata.get("modelspec.architecture", "")
            if not arch.startswith("naflexvit_so400m_patch16_siglip"):
                raise ValueError(f"Unrecognized model architecture: {arch}")

            tags = metadata["classifier.labels"].split("\n")

            model = timm.create_model(
                'naflexvit_so400m_patch16_siglip',
                pretrained=False, num_classes=0,
                pos_embed_interp_mode="bilinear",
                weight_init="skip", fix_init=False,
                device="cpu", dtype=torch.bfloat16,
            )

            variant = arch[31:]
            if variant == "": # vanilla
                model.reset_classifier(len(tags))
            elif variant == "+rr_slim":
                model.reset_classifier(len(tags))
                if "attn_pool.q.weight" not in state_dict:
                    model.attn_pool.q = Identity()
                if "head.bias" not in state_dict:
                    model.head.bias = None
            elif variant == "+rr_hydra":
                model.attn_pool = HydraPool.for_state(
                    state_dict, "attn_pool.",
                    device="cpu", dtype=torch.bfloat16
                )
                model.head = model.attn_pool.create_head()
                model.num_classes = len(tags)
                # q_normed is handled by HydraPool logic but we can set extra state if needed
                state_dict["attn_pool._extra_state"] = { "q_normed": True }
            else:
                raise ValueError(f"Unrecognized model variant: {variant}")

            model.eval().to(dtype=torch.bfloat16)

            # Load state dict
            msg = model.load_state_dict(state_dict, strict=False)

            # Move to device
            model.to(device=device)

            ComfyCache.set(f'model_v3.{model_name}', {
                "model": model,
                "tags": tags,
                "device": device
            })
            ComfyLogger().log(message=f"Model {model_name} loaded successfully", type="INFO", always=True)
            return True, tags

        except Exception as e:
            ComfyLogger().log(message=f"Error loading model {model_name}: {e}", type="ERROR", always=True)
            return False, []

    @classmethod
    def is_installed(cls, model_name: str) -> bool:
        return any(model_name + ".safetensors" in s for s in cls.list_installed())

    @classmethod
    def list_installed(cls) -> List[str]:
        model_path = os.path.abspath(cls().model_basepath)
        if not os.path.exists(model_path):
            ComfyLogger().log(message=f"Model path {model_path} does not exist, it is being created", type="WARN", always=True)
            os.makedirs(os.path.abspath(model_path))
            return []
        models = list(filter(lambda x: x.endswith(".safetensors"), os.listdir(model_path)))
        return models

    @classmethod
    def download(cls, model_name: str) -> bool:
        # Create model directory if it doesn't exist
        os.makedirs(cls().model_basepath, exist_ok=True)

        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config["huggingface_endpoint"]
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        if hf_endpoint.endswith("/"):
            hf_endpoint = hf_endpoint.rstrip("/")

        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")

        # Use models_v3 config structure
        url: str = ComfyExtensionConfig().get(property=f"models_v3.{model_name}.url")
        if not url:
             ComfyLogger().log(message=f"No URL found for model {model_name} in config", type="ERROR", always=True)
             return False

        url = url.replace("{HF_ENDPOINT}", hf_endpoint)

        ComfyLogger().log(message=f"Downloading model {model_name} from {url}", type="INFO", always=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            with open(model_path, 'wb') as file, tqdm(
                desc=model_name,
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
                        cls().download_progress_callback(int(downloaded / total_size * 100), model_name)

            cls().download_complete_callback(model_name)
            return True
        except Exception as err:
            ComfyLogger().log(message=f"Unable to download model: {err}", type="ERROR", always=True)
            return False
