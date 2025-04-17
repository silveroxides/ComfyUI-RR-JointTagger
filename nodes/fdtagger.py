from enum import Enum
import torch
import numpy as np
import os
from PIL import Image
from aiohttp import web
from typing import List, Optional, Union, Tuple, Dict, Any

import comfy.utils
from ..helpers.cache import CacheCleanupMethod
from ..redrocket.image_manager import JtpImageManager
from server import PromptServer

from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager
from ..redrocket.classifier import JtpInference
from ..helpers.extension import ComfyExtension
from ..helpers.config import ComfyExtensionConfig
from ..helpers.multithreading import ComfyThreading

    
class ModelDevice(Enum):
    CPU = "cpu"
    GPU = "cuda"
    
    def to_torch_device(self) -> torch.device:
        return torch.device(self.value)

async def classify_tags(image: Image.Image, model_name: str, tags_name: str, device: torch.device, steps: float = 0.35, threshold: float = 0.35, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False) -> Tuple[str, Dict[str, float]]:
    """
    Classify e621 tags for an image using RedRocket JTP Vision Transformer model
    """
    tag_string, tag_scores = await JtpInference().run_classifier(model_name=model_name, device=device, tags_name=tags_name, image=image, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
    return tag_string, tag_scores


async def download_progress_callback(perc: int, file_name: str, client_id: Optional[str] = None, node: Optional[str] = None, api_endpoint: Optional[str] = None) -> None:
    """
    Callback function for download progress updates
    """
    from ..helpers.nodes import ComfyNode
    if client_id is None:
        client_id = ComfyExtension().client_id()
    if api_endpoint is None:
        api_endpoint = ComfyExtensionConfig().get(property="api_endpoint")
    if api_endpoint is None:
        raise ValueError("API endpoint is not set")
    message: str = ""
    if perc < 100:
        message = "[{0}%] Downloading {1}...".format(perc, file_name)
    else:
        message = "Download {0} complete!".format(file_name)
    ComfyNode().update_node_status(client_id=client_id, node=node, api_endpoint=api_endpoint, text=message, progress=perc)


async def download_complete_callback(file_name: str, client_id: Optional[str] = None, node: Optional[str] = None, api_endpoint: Optional[str] = None) -> None:
    """
    Callback function for download completion updates
    """
    from ..helpers.nodes import ComfyNode
    from ..helpers.extension import ComfyExtension
    if client_id is None:
        client_id = ComfyExtension().client_id()
    if client_id is None:
        raise ValueError("Client ID is not set")
    if api_endpoint is None:
        api_endpoint = ComfyExtensionConfig().get(property="api_endpoint")
    if api_endpoint is None:
        raise ValueError("API endpoint is not set")
    ComfyNode().update_node_status(client_id=client_id, node=node, api_endpoint=api_endpoint)


api_endpoint: str = ComfyExtensionConfig().get(property="api_endpoint")
@PromptServer.instance.routes.get(f"/{api_endpoint}/fdtagger/tag")
async def get_tags(request: web.Request) -> web.Response:
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)
    type: str = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)
    target_dir: str = ComfyExtension().comfy_dir(type)
    image_path: str = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)
    if not os.path.isfile(image_path):
        return web.Response(status=404)
    image: np.ndarray = np.array(Image.open(image_path).convert("RGBA"))
    models: List[str] = JtpModelManager().list_installed()
    default: str = ComfyExtensionConfig().get()["settings"]["model"]
    model: str = default if default in models else models[0]
    steps: int = int(request.query.get("steps", ComfyExtensionConfig().get()["settings"]["steps"]))
    threshold: float = float(request.query.get("threshold", ComfyExtensionConfig().get()["settings"]["threshold"]))
    exclude_tags: str = request.query.get("exclude_tags", ComfyExtensionConfig().get()["settings"]["exclude_tags"])
    replace_underscore: bool = request.query.get("replace_underscore", ComfyExtensionConfig().get()["settings"]["replace_underscore"]) == "true"
    trailing_comma: bool = request.query.get("trailing_comma", ComfyExtensionConfig().get()["settings"]["trailing_comma"]) == "true"
    device: ModelDevice = ModelDevice(request.query.get("device", "cpu"))
    client_id: str = request.rel_url.query.get("clientId", None)
    node: str = request.rel_url.query.get("node", None)
    return web.json_response(await classify_tags(image=image, model_name=model, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma, client_id=client_id, node=node))


class FDTagger():
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        models: List[str] =  [v["name"] for k, v in ComfyExtensionConfig().get(property="models").items()]
        device: List[str] = ["cpu", "cuda"]
        return {"required": {
            "image": ("IMAGE", ),
            "device": (device, {"default": ComfyExtensionConfig().get(property="fdtagger_settings.device")}),
            "model": (models, {"default": ComfyExtensionConfig().get(property="fdtagger_settings.model")}),
            "steps": ("INT", {"default": ComfyExtensionConfig().get(property="fdtagger_settings.steps"), "min": 1, "max": 500, "display": "slider"}),
            "threshold": ("FLOAT", {"default": ComfyExtensionConfig().get(property="fdtagger_settings.threshold"), "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            "replace_underscore": ("BOOLEAN", {"default": ComfyExtensionConfig().get(property="fdtagger_settings.replace_underscore")}),
            "trailing_comma": ("BOOLEAN", {"default": ComfyExtensionConfig().get(property="fdtagger_settings.trailing_comma")}),
            "exclude_tags": ("STRING", {"default": ComfyExtensionConfig().get(property="fdtagger_settings.exclude_tags"), "multiline": True}),
        }}

    RETURN_TYPES: Tuple[str] = ("STRING", "TAGSCORES",)
    RETURN_NAMES: Tuple[str] = ("tags", "scores",)
    OUTPUT_IS_LIST: Tuple[bool] = (True, True,)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "🐺 Furry Diffusion"

    def tag(self, image: Image.Image, device: str, model: str, steps: int, threshold: float, exclude_tags: str = "", replace_underscore: bool = False, trailing_comma: bool = False) -> Dict[str, Any]:
        model_name = ComfyExtensionConfig().get_model_from_name(model)
        tags_name = ComfyExtensionConfig().get_tags_from_name(model)
        device_type = ModelDevice(device)
        tensor: np.ndarray = image * 255
        tensor = np.array(tensor, dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags: List[str] = []
        scores: List[Dict[str, float]] = []
        for i in range(tensor.shape[0]):
            img: Image.Image = Image.fromarray(tensor[i]).convert("RGBA")
            tags_t, scores_t = ComfyThreading().wait_for_async(lambda: classify_tags(image=img, model_name=model_name, tags_name=tags_name, device=device_type.to_torch_device(), threshold=threshold,
                        exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma))
            tags.append(tags_t)
            scores.append(scores_t)
            pbar.update(1)
        return {"ui": {"tags": tags, "scores": scores}, "result": (tags, scores,)}

JtpModelManager(model_basepath=ComfyExtension().extension_dir("models", mkdir=True), download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpTagManager(tags_basepath=ComfyExtension().extension_dir("tags", mkdir=True), download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpImageManager(cache_maxsize=ComfyExtensionConfig().get(property="image_cache_maxsize"), cache_method=CacheCleanupMethod(ComfyExtensionConfig().get(property="image_cache_method")))
JtpInference(device=ComfyExtensionConfig().get(property="device"))

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "FDTagger|furrydiffusion": FDTagger,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "FDTagger|furrydiffusion": "FurryDiffusion Tagger 🐺",
}
