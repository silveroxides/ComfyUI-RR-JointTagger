from enum import Enum
import torch
import numpy as np
import os
from PIL import Image
from aiohttp import web
from typing import List, Optional, Union, Tuple, Dict, Any
import re
import comfy.utils
import comfy.model_management
from ..redrocket.image_manager import JtpImageManager
from server import PromptServer
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator

from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager
from ..redrocket.classifier import JtpInference
from ..helpers.extension import ComfyExtension
from ..helpers.config import ComfyExtensionConfig
import folder_paths

model_basepath = os.path.join(folder_paths.models_dir, "RedRocket")
tags_basepath = os.path.join(model_basepath, "tags")

class ModelDevice(Enum):
    CPU = "cpu"
    GPU = "cuda"
    
    def to_torch_device(self) -> torch.device:
        return torch.device(self.value)

def classify_tags(image: Image.Image, model_name: str, tags_name: str, device: comfy.model_management.get_torch_device(), steps: float = 0.35, threshold: float = 0.35, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False) -> Tuple[str, Dict[str, float]]:
    """
    Classify e621 tags for an image using RedRocket JTP Vision Transformer model
    """
    tag_string, tag_scores = JtpInference().run_classifier(model_name=model_name, device=device, tags_name=tags_name, image=image, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
    return tag_string, tag_scores


def download_progress_callback(perc: int, file_name: str, client_id: Optional[str] = None, node: Optional[str] = None, api_endpoint: Optional[str] = None) -> None:
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


def download_complete_callback(file_name: str, client_id: Optional[str] = None, node: Optional[str] = None, api_endpoint: Optional[str] = None) -> None:
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
@PromptServer.instance.routes.get(f"/{api_endpoint}/rrtagger/tag")
def get_tags(request: web.Request) -> web.Response:
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
    return web.json_response(classify_tags(image=image, model_name=model, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma, client_id=client_id, node=node))

def normalize_tag(tag: str) -> str:
  """
  Normalizes a single tag by:
  1. Stripping leading/trailing whitespace.
  2. Replacing internal spaces with underscores.
  """
  # Strip whitespace from the ends first
  stripped_tag = tag.strip()
  # Replace all occurrences of one or more spaces with a single underscore
  # This handles cases like "digital   media" -> "digital_media"
  normalized = re.sub(r'\s+', '_', stripped_tag)
  return normalized

def filter_tags(input_tags_str: str, remove_tags_str: str) -> str:
  """
  Filters tags from an input string based on a removal string,
  handling various formatting inconsistencies.

  Args:
    input_tags_str: A string containing a comma-separated list of input tags.
                    Tags can have spaces or underscores, varying spacing
                    around commas, and special characters.
    remove_tags_str: A string containing a comma-separated list of tags to
                     be removed. Follows the same potential formatting
                     variations as input_tags_str.

  Returns:
    A string containing the filtered tags. Remaining tags will have
    internal spaces replaced with underscores, and tags will be
    separated by a comma and a space (", ").
    Returns an empty string if no tags remain or the input string is empty.
  """
  # Handle empty input string immediately
  if not input_tags_str:
    return ""

  # 1. Process Input Tags:
  #    - Split the string by commas.
  #    - For each potential tag, normalize it (strip whitespace, replace internal spaces with underscores).
  #    - Filter out any empty strings that might result from splitting (e.g., ",," or trailing comma).
  input_tags_list = [
      normalize_tag(tag) for tag in input_tags_str.split(',') if tag.strip()
  ]

  # 2. Process Tags to Remove:
  #    - Perform the same normalization process as for input tags.
  #    - Store them in a set for efficient O(1) average-case lookup during filtering.
  if not remove_tags_str:
      remove_tags_set = set()
  else:
      remove_tags_set = {
          normalize_tag(tag) for tag in remove_tags_str.split(',') if tag.strip()
      }

  # 3. Filter the Input Tags:
  #    - Create a new list containing only those normalized input tags
  #      that are NOT found in the normalized set of tags to remove.
  filtered_tags = [
      tag for tag in input_tags_list if tag not in remove_tags_set
  ]

  # 4. Format the Output String:
  #    - Join the filtered tags with ", " as the separator.
  return ", ".join(filtered_tags)

class RRJointTagger(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        models: List[str] =  [v["name"] for k, v in ComfyExtensionConfig().get(property="models").items()]
        return {"required": {
            "image": (IO.IMAGE, ),
            "model": (models, {"default": ComfyExtensionConfig().get(property="rrtagger_settings.model")}),
            "steps": (IO.INT, {"default": 255, "min": 1, "max": 500, "display": "slider"}),
            "threshold": (IO.FLOAT, {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            "replace_underscore": (IO.BOOLEAN, {"default": False}),
            "trailing_comma": (IO.BOOLEAN, {"default": False}),
            "exclude_tags": (IO.STRING, {"multiline": True, "tooltip": "Tags to exclude from output."}),
            "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES: Tuple[str] = (IO.STRING, IO.STRING,)
    RETURN_NAMES: Tuple[str] = ("tags", "scores",)
    OUTPUT_IS_LIST: Tuple[bool] = (True, True,)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "🐺 Furry Diffusion"

    def tag(self, image: Image.Image, model: str, steps: int, threshold: float, seed: int, exclude_tags: str = "", replace_underscore: bool = False, trailing_comma: bool = False) -> Dict[str, Any]:
        model_name = ComfyExtensionConfig().get_model_from_name(model)
        tags_name = ComfyExtensionConfig().get_tags_from_name(model)
        device_type = comfy.model_management.get_torch_device()
        
        # Set the seed for reproducibility if needed, though inference is deterministic
        torch.manual_seed(seed)
        
        tensor: np.ndarray = image * 255
        tensor = np.array(tensor, dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags: List[str] = []
        scores: List[Dict[str, float]] = []
        for i in range(tensor.shape[0]):
            img: Image.Image = Image.fromarray(tensor[i]).convert("RGBA")
            tags_t, scores_t = classify_tags(image=img, model_name=model_name, tags_name=tags_name, device=comfy.model_management.get_torch_device(), threshold=threshold,
                        exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
            print(f"tags_t={tags_t}")
            final_tags = filter_tags(tags_t, exclude_tags)
            tags.append(final_tags)
            print(f"tags={tags}")
            scores.append(scores_t)
            pbar.update(1)
            print(f"exclude_tags={exclude_tags}")
            # tags = [tag for tag in tags if tag[0] not in remove]
        return {"ui": {"tags": tags, "scores": scores}, "result": (tags, scores,)}

JtpModelManager(model_basepath=model_basepath, download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpTagManager(tags_basepath=tags_basepath, download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpInference(device=comfy.model_management.get_torch_device())

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "RRJointTagger|redrocket": RRJointTagger,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "RRJointTagger|redrocket": "RedRocket Tagger 🐺",
}
