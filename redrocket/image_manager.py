import hashlib
import os
from pathlib import Path
import time
from PIL import Image
from typing import Any, Dict, Tuple, Union
import numpy
import torch
from enum import Enum
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from ..helpers.cache import CacheCleanupMethod, ComfyCache

from ..helpers.metaclasses import Singleton


class Fit(torch.nn.Module):
    """
    Resize an image to fit within the given bounds while maintaining aspect ratio
    """
    def __init__(self, bounds: Union[Tuple[int, int], int], interpolation: InterpolationMode = InterpolationMode.LANCZOS, grow: bool = True, pad: Union[float, None] = None) -> None:
        super().__init__()
        self.bounds = (bounds, bounds) if isinstance(bounds, int) else bounds
        self.interpolation = interpolation
        self.grow = grow
        self.pad = pad

    def forward(self, img) -> Any:
        wimg, himg = img.size
        hbound, wbound = self.bounds
        hscale = hbound / himg
        wscale = wbound / wimg
        if not self.grow:
            hscale = min(hscale, 1.0)
            wscale = min(wscale, 1.0)
        scale = min(hscale, wscale)
        if scale == 1.0:
            return img
        hnew = min(round(himg * scale), hbound)
        wnew = min(round(wimg * scale), wbound)
        img = TF.resize(img, (hnew, wnew), self.interpolation)
        if self.pad is None:
            return img
        hpad = hbound - hnew
        wpad = wbound - wnew
        tpad = hpad // 2
        bpad = hpad - tpad
        lpad = wpad // 2
        rpad = wpad - lpad
        return TF.pad(img, (lpad, tpad, rpad, bpad), self.pad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"bounds={self.bounds}, " +
            f"interpolation={self.interpolation.value}, " +
            f"grow={self.grow}, " +
            f"pad={self.pad})"
        )


class CompositeAlpha(torch.nn.Module):
    """
    Composite an image with an alpha channel onto a background color
    """
    def __init__(self, background: Union[Tuple[float, float, float], float]) -> None:
        super().__init__()
        self.background = (background, background, background) if isinstance(
            background, float) else background
        self.background = torch.tensor(
            self.background).unsqueeze(1).unsqueeze(2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-3] == 3:
            return img
        alpha = img[..., 3, None, :, :]
        img[..., :3, :, :] *= alpha
        background = self.background.expand(-1, img.shape[-2], img.shape[-1])
        if background.ndim == 1:
            background = background[:, None, None]
        elif background.ndim == 2:
            background = background[None, :, :]
        img[..., :3, :, :] += (1.0 - alpha) * background
        return img[..., :3, :, :]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"background={self.background})"
        )


class JtpImageManager(metaclass=Singleton):
    def __init__(self, cache_maxsize: int = 1, cache_method: CacheCleanupMethod = CacheCleanupMethod.ROUND_ROBIN) -> None:
        ComfyCache.set_max_size('image', cache_maxsize)
        ComfyCache.set_cachemethod('image', CacheCleanupMethod[cache_method.name])
  
    def __del__(self) -> None:
        ComfyCache.flush('image')
        import gc
        gc.collect()

    @classmethod
    def is_cached(cls, image_name: str) -> bool:
        """
        Check if an image is loaded into memory
        """
        return ComfyCache.get(f'image.{image_name}') is not None and ComfyCache.get(f'image.{image_name}.input') is not None

    @classmethod
    def is_done(cls, image_name: str) -> bool:
        """
        Check if an image is done processing
        """
        return cls.is_cached(image_name) and ComfyCache.get(f'image.{image_name}.output') is not None
    
    @classmethod
    def get_transform(cls, width: int, height: int, interpolation, grow, pad, background: Tuple[int, int, int]) -> transforms.Compose:
        """
        Perform transformations on an image
        """
        return transforms.Compose([
            Fit(bounds=(width, height), interpolation=interpolation, grow=grow, pad=pad),
            transforms.ToTensor(),
            CompositeAlpha(background=background),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.CenterCrop(size=(width, height)),
        ])
  
    @classmethod
    async def load(cls, image: Union[Path, numpy.ndarray, Image.Image, None], device: torch.device) -> Union[Tuple[str, Dict[str, Any]], torch.Tensor, None]:
        """
        Load an image into memory
  
        - Return None if no image is provided or there was an error loading it
        - Return a loaded tensor if we need to perform inference on it
        - Return a tuple containing tags and tag:score dict if we are already through with it.
        """
        from ..helpers.logger import ComfyLogger
        if image is not None and isinstance(image, Path):
            # Image is a path to an image, so load it with PIL
            image = str(image)
            if cls.is_done(image):
                ComfyLogger().log(f"Image {image} already processed, using from cache", "WARNING", True)
                return ComfyCache.get(f'image.{image}.output')
            if cls.is_cached(image):
                ComfyLogger().log(f"Image {image} already loaded, using from cache", "WARNING", True)
                ComfyCache.set(f'image.{image}.used_timestamp', time.time())
                image_input = ComfyCache.get(f'image.{image}.input')
            else:
                if not os.path.exists(image):
                    ComfyLogger.log(f"Image {image} not found in path: {image}", "ERROR", True)
                    return None
                ComfyCache.set(f'image.{image}', {
                    "input": Image.open(image).convert("RGBA"),
                    "timestamp": time.time(),
                    "used_timestamp": time.time(),
                    "device": device,
                    "output": None
                })
                ComfyLogger().log(f"Image: {image} loaded into cache", "DEBUG", True)
                image_input = ComfyCache.get(f'image.{image}.input')
        elif image is not None and isinstance(image, numpy.ndarray):
            # Image is a numpy array, so convert it to a PIL image
            image_hash = hashlib.sha256(image.tobytes(), usedforsecurity=False).hexdigest()
            if cls.is_done(image_hash):
                ComfyLogger().log(f"Image {image} already processed, using from cache", "WARNING", True)
                return ComfyCache.get(f'image.{image_hash}.output')
            if cls.is_cached(image_hash):
                ComfyLogger().log(f"Image {image_hash} already loaded, using from cache", "WARNING", True)
                ComfyCache.set(f'image.{image_hash}.used_timestamp', time.time())
                image_input = ComfyCache.get(f'image.{image_hash}.input')
            else:
                ComfyCache.set(f'image.{image_hash}', {
                    "input": Image.fromarray(image).convert("RGBA"),
                    "timestamp": time.time(),
                    "used_timestamp": time.time(),
                    "device": device,
                    "output": None
                })
                ComfyLogger().log(f"Image {image_hash} loaded into cache", "DEBUG", True)
                image_input = ComfyCache.get(f'image.{image_hash}.input')
        elif image is not None and isinstance(image, Image.Image):
            # Image is a PIL image, we need to sha256 hash it
            img = numpy.array(image.convert("RGBA"), dtype=numpy.uint8)
            image_hash = hashlib.sha256(img.tobytes(), usedforsecurity=False).hexdigest()
            if cls.is_done(image_hash):
                ComfyLogger().log(f"Image {image} already processed, using from cache", "WARNING", True)
                return ComfyCache.get(f'image.{image_hash}.output')
            if cls.is_cached(image_hash):
                ComfyLogger().log(f"Image {image_hash} already loaded, using from cache", "WARNING", True)
                ComfyCache.set(f'image.{image_hash}.used_timestamp', time.time())
                image_input = ComfyCache.get(f'image.{image_hash}.input')
            else:
                ComfyCache.set(f'image.{image_hash}', {
					"input": image.convert("RGBA"),
					"timestamp": time.time(),
					"used_timestamp": time.time(),
					"device": device,
					"output": None
				})
                ComfyLogger().log(f"Image {image_hash} loaded into cache", "DEBUG", True)
                image_input = image
        else:
            ComfyLogger().log("No image provided to load", "ERROR", True)
            return None
        # If we reach this codepath, we have to perform inference on the image
        transform = cls().get_transform(
            width=384,
            height=384,
            interpolation=InterpolationMode.LANCZOS,
            grow=True,
            pad=None,
            background=0.5,
        )
        tensor = transform(image_input).unsqueeze(0).to(device.type)
        if torch.cuda.is_available() is True and device.type == "cuda":
            tensor.cuda()
            if torch.cuda.get_device_capability()[0] >= 7:
                tensor = tensor.to(dtype=torch.float16, memory_format=torch.channels_last)
                ComfyLogger().log("Image loaded to GPU with mixed precision", "INFO", True)
            else:
                ComfyLogger().log("Image loaded to older GPU without mixed precision", "WARNING", True)
        return tensor

    @classmethod
    def unload_image(cls, image: Union[Path, numpy.ndarray, Image.Image, None]) -> bool:
        """
        Unload an image from memory
        """
        if image is not None and isinstance(image, Path):
            image = str(image)
            if cls.is_cached(image):
                ComfyCache.flush(f'image.{image}')
                return True
        elif image is not None and isinstance(image, numpy.ndarray):
            image_hash = hashlib.sha256(image.tobytes(), usedforsecurity=False).hexdigest()
            if cls.is_cached(image_hash):
                ComfyCache.flush(f'image.{image_hash}')
                return True
        elif image is not None and isinstance(image, Image.Image):
            img = numpy.array(image.convert("RGBA"), dtype=numpy.uint8)
            image_hash = hashlib.sha256(img.tobytes(), usedforsecurity=False).hexdigest()
            if cls.is_cached(image_hash):
                ComfyCache.flush(f'image.{image_hash}')
                return True
        return False

    @classmethod
    async def commit_cache(cls, image, output: Tuple[str, Dict[str, Any]]) -> bool:
        """
        Commit the output of an image to the cache
        """
        from ..helpers.logger import ComfyLogger
        if image is not None and isinstance(image, Path) and output is not None:
            image = str(image)
            if cls.is_cached(image):
                ComfyCache.set(f'image.{image}.output', output)
                return True
        elif image is not None and isinstance(image, numpy.ndarray) and output is not None:
            image_hash = hashlib.sha256(image.tobytes(), usedforsecurity=False).hexdigest()
            if cls.is_cached(image_hash):
                ComfyCache.set(f'image.{image_hash}.output', output)
                return True
        elif image is not None and isinstance(image, Image.Image) and output is not None:
            img = numpy.array(image.convert("RGBA"), dtype=numpy.uint8)
            image_hash = hashlib.sha256(img.tobytes(), usedforsecurity=False).hexdigest()
            if cls.is_cached(image_hash):
                ComfyCache.set(f'image.{image_hash}.output', output)
                return True
        else:
            ComfyLogger().log("Nothing to commit to cache", "ERROR", True)
        return False