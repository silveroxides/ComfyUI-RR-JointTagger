import hashlib
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from math import ceil
from typing import Any, Tuple, Union, Optional, Dict
from einops import rearrange

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.metaclasses import Singleton
from ..helpers.logger import ComfyLogger

class JtpImageV3Manager(metaclass=Singleton):
    def __init__(self) -> None:
        ComfyCache.set_max_size('image_v3', 1) # Adjust as needed
        ComfyCache.set_cachemethod('image_v3', CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush('image_v3')
        import gc
        gc.collect()

    @staticmethod
    def get_image_size_for_seq(
        image_hw: tuple[int, int],
        patch_size: int = 16,
        max_seq_len: int = 1024,
        max_ratio: float = 1.0,
        eps: float = 1e-5,
    ) -> tuple[int, int]:
        """Determine image size for sequence length constraint."""

        assert max_ratio >= 1.0
        assert eps * 2 < max_ratio

        h, w = image_hw
        max_py = int(max((h * max_ratio) // patch_size, 1))
        max_px = int(max((w * max_ratio) // patch_size, 1))

        if (max_py * max_px) <= max_seq_len:
            return max_py * patch_size, max_px * patch_size

        def patchify(ratio: float) -> tuple[int, int]:
            return (
                min(int(ceil((h * ratio) / patch_size)), max_py),
                min(int(ceil((w * ratio) / patch_size)), max_px)
            )

        py, px = patchify(eps)
        if (py * px) > max_seq_len:
            # Fallback or error? For safety we can clamp to 1x1 patch or raise
            # But the original code raises ValueError. We'll try to be robust.
            # raise ValueError(f"Image of size {w}x{h} is too large.")
            return patch_size, patch_size

        ratio = eps
        while (max_ratio - ratio) >= eps:
            mid = (ratio + max_ratio) / 2.0

            mpy, mpx = patchify(mid)
            seq_len = mpy * mpx

            if seq_len > max_seq_len:
                max_ratio = mid
                continue

            ratio = mid
            py = mpy
            px = mpx

            if seq_len == max_seq_len:
                break

        assert py >= 1 and px >= 1
        return py * patch_size, px * patch_size

    @staticmethod
    def process_image(img: Image.Image, patch_size: int, max_seq_len: int) -> Image.Image:
        def compute_resize(wh: tuple[int, int]) -> tuple[int, int]:
            h, w = JtpImageV3Manager.get_image_size_for_seq((wh[1], wh[0]), patch_size, max_seq_len)
            return w, h
        
        # Simple resize using LANCZOS, avoiding full PIL.ImageCms complexity for now as ComfyUI handles color spaces usually
        w, h = compute_resize(img.size)
        if (w, h) != img.size:
            img = img.resize((w, h), resample=Image.Resampling.LANCZOS)
        
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        return img

    @staticmethod
    def put_srgb_patch(
        img: Image.Image,
        patch_data: torch.Tensor,
        patch_coord: torch.Tensor,
        patch_valid: torch.Tensor,
        patch_size: int
    ) -> None:
        if img.mode not in ("RGB", "RGBA", "RGBa"):
             img = img.convert("RGB")

        # Ensure numpy array is HWC
        img_arr = np.asarray(img)
        if img_arr.shape[-1] == 4:
            img_arr = img_arr[:, :, :3]
            
        patches = rearrange(
            img_arr,
            "(h p1) (w p2) c -> h w (p1 p2 c)",
            p1=patch_size, p2=patch_size
        )

        coords = np.stack(np.meshgrid(
            np.arange(patches.shape[0], dtype=np.int16),
            np.arange(patches.shape[1], dtype=np.int16),
            indexing="ij"
        ), axis=-1)

        coords = rearrange(coords, "h w c -> (h w) c")
        patches = rearrange(patches, "h w p -> (h w) p")
        n = patches.shape[0]

        np.copyto(patch_data[:n].numpy(), patches, casting="no")
        np.copyto(patch_coord[:n].numpy(), coords, casting="no")
        patch_valid[:n] = True

    @classmethod
    def patchify_image(cls, img: Image.Image, patch_size: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = torch.zeros(max_seq_len, patch_size * patch_size * 3, device="cpu", dtype=torch.uint8)
        patch_coords = torch.zeros(max_seq_len, 2, device="cpu", dtype=torch.int16)
        patch_valid = torch.zeros(max_seq_len, device="cpu", dtype=torch.bool)

        cls.put_srgb_patch(img, patches, patch_coords, patch_valid, patch_size)
        return patches, patch_coords, patch_valid

    @classmethod
    def load(cls, image: Union[Path, np.ndarray, Image.Image, None], device: torch.device, seqlen: int = 1024, params_key: str = None) -> Union[Tuple[Any, ...], Tuple[str, Dict[str, Any]], None]:
        """
        Load an image into memory, preprocessing for NaFlex.
        Returns (patches, patch_coords, patch_valid) tensors OR cached output.
        """
        patch_size = 16 # Fixed for JTP-3
        
        image_key = None
        pil_image = None
        
        if image is not None and isinstance(image, Path):
            image_key = str(image)
            if os.path.exists(image_key):
                pil_image = Image.open(image_key).convert("RGB")
        elif image is not None and isinstance(image, np.ndarray):
            image_key = hashlib.sha256(image.tobytes()).hexdigest()
            pil_image = Image.fromarray(image).convert("RGB")
        elif image is not None and isinstance(image, Image.Image):
            # For PIL image, we need a hash
            img_arr = np.array(image.convert("RGB"))
            image_key = hashlib.sha256(img_arr.tobytes()).hexdigest()
            pil_image = image.convert("RGB")
            
        if image_key is None or pil_image is None:
            ComfyLogger().log("No valid image provided", "ERROR", True)
            return None

        # Check cache
        cache_key = f'image_v3.{image_key}'
        cached_output = ComfyCache.get(f'{cache_key}.output')
        
        if params_key and cached_output and isinstance(cached_output, dict) and params_key in cached_output:
            ComfyLogger().log(f"Returning cached result for {image_key}", "DEBUG", True)
            return cached_output[params_key]

        cached_input = ComfyCache.get(f'{cache_key}.input')
        if cached_input is not None:
             # Check if cached input matches seqlen requirements?
             # NaFlex depends on seqlen for resizing.
             # If seqlen changes, we might need to re-process.
             # So we should include seqlen in the input cache key or check it.
             # Let's store input as dict by seqlen
             if seqlen in cached_input:
                 tensors = cached_input[seqlen]
                 return tensors[0].to(device), tensors[1].to(device), tensors[2].to(device)

        # Process image
        processed = cls.process_image(pil_image, patch_size, seqlen)
        patches, coords, valid = cls.patchify_image(processed, patch_size, seqlen)
        
        # Store in cache
        if cached_input is None:
            cached_input = {}
        
        cached_input[seqlen] = (patches, coords, valid)
        
        ComfyCache.set(f'{cache_key}.input', cached_input)
        if ComfyCache.get(f'{cache_key}.output') is None:
             ComfyCache.set(f'{cache_key}.output', {}) # Initialize output dict
             
        return patches.to(device), coords.to(device), valid.to(device)

    @classmethod
    def commit_cache(cls, image_key: str, output: Any, params_key: str) -> bool:
        if not image_key: return False
        cache_key = f'image_v3.{image_key}'
        
        current_output = ComfyCache.get(f'{cache_key}.output')
        if current_output is None:
            current_output = {}
            
        current_output[params_key] = output
        ComfyCache.set(f'{cache_key}.output', current_output)
        return True

    @staticmethod
    def unpatchify(seq: torch.Tensor, coords: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Scatter valid patches from (seqlen, ...) to (H, W, ...).
        """
        valid_coords = coords[0, valid[0]]  # (n_valid, 2)
        valid_patches = seq[valid[0]]  # (n_valid, ...)

        if valid_coords.numel() == 0:
             return seq.new_zeros((1, 1) + seq.shape[1:])

        h = int(valid_coords[:, 0].max().item()) + 1
        w = int(valid_coords[:, 1].max().item()) + 1

        output_shape = (h, w) + seq.shape[1:]
        output = seq.new_zeros(output_shape)

        output[valid_coords[:, 0], valid_coords[:, 1]] = valid_patches
        return output

    @staticmethod
    def cam_composite(image: Image.Image, cam: np.ndarray) -> Image.Image:
        """
        Overlays CAM on image and returns a PIL image.
        """
        cam_abs = np.abs(cam)
        cam_scale = cam_abs.max() + 1e-5

        cam_rgba = np.dstack((
            (cam < 0).astype(np.float32),
            (cam > 0).astype(np.float32),
            np.zeros_like(cam, dtype=np.float32),
            cam_abs * (0.5 / cam_scale),
        ))  # Shape: (H, W, 4)

        cam_pil = Image.fromarray((cam_rgba * 255).astype(np.uint8))
        cam_pil = cam_pil.resize(image.size, resample=Image.Resampling.NEAREST)

        image = Image.blend(
            image.convert('RGBA'),
            image.convert('L').convert('RGBA'),
            0.33
        )

        image = Image.alpha_composite(image, cam_pil)
        return image.convert("RGB")
