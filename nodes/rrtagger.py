import torch
import numpy as np
import os
from PIL import Image
from typing import List, Optional, Union, Tuple, Dict, Any
import re
import comfy.utils
import comfy.model_management
from ..redrocket.image_manager import JtpImageManager
from comfy.comfy_types import IO, ComfyNodeABC

from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager
from ..redrocket.classifier import JtpInference
from ..helpers.extension import ComfyExtension
from ..helpers.config import ComfyExtensionConfig
import folder_paths

model_basepath = os.path.join(folder_paths.models_dir, "RedRocket")
tags_basepath = os.path.join(model_basepath, "tags")

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
            "implications_mode": (["inherit", "constrain", "remove", "constrain-remove", "off"], {"default": "off", "tooltip": "How to handle implied tags (e.g. if 'cat' is present, 'feline' is implied). Requires JTP-3 metadata CSV."}),
            "exclude_tags": (IO.STRING, {"multiline": True, "tooltip": "Comma-separated tags to exclude. Supports * wildcards: human* (starts with), *human (ends with), *human* (contains), human*top (starts/ends)."}),
            "exclude_categories": (IO.STRING, {"multiline": True, "tooltip": "Comma separated categories to exclude (e.g. copyright, character, species, meta, lore)."}),
            "prefix": (IO.STRING, {"default": "", "tooltip": "Text to prepend to the tags output."}),
            "replace_underscore": (IO.BOOLEAN, {"default": True, "tooltip": "Replace underscores with spaces in tags."}),
            "trailing_comma": (IO.BOOLEAN, {"default": False, "tooltip": "Add a trailing comma to the tag string."}),
            "keep_model_loaded": (IO.BOOLEAN, {"default": False, "tooltip": "If True, keep the model in RAM between runs for faster repeat inference. If False, fully unload after each run to free RAM."}),
            "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES: Tuple[str] = (IO.STRING, IO.STRING,)
    RETURN_NAMES: Tuple[str] = ("tags", "scores",)
    OUTPUT_IS_LIST: Tuple[bool] = (True, True,)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "🐺 Furry Diffusion"

    def tag(self, image: Image.Image, model: str, steps: int, threshold: float, seed: int,
            exclude_tags: str = "", exclude_categories: str = "", prefix: str = "",
            replace_underscore: bool = True, trailing_comma: bool = False,
            keep_model_loaded: bool = False,
            implications_mode: str = "off") -> Dict[str, Any]:
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
            tags_t, scores_t = JtpInference.run_classifier(
                model_name=model_name,
                tags_name=tags_name,
                device=comfy.model_management.get_torch_device(),
                image=img,
                steps=steps,
                threshold=threshold,
                exclude_tags=exclude_tags,
                replace_underscore=replace_underscore,
                trailing_comma=trailing_comma,
                implications_mode=implications_mode,
                exclude_categories=exclude_categories,
                prefix=prefix,
                keep_model_loaded=keep_model_loaded,
            )
            tags.append(tags_t)
            scores.append(scores_t)
            pbar.update(1)
        return {"ui": {"tags": tags, "scores": scores}, "result": (tags, scores,)}

JtpModelManager(model_basepath=model_basepath)
JtpTagManager(tags_basepath=tags_basepath)

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "RRJointTagger|redrocket": RRJointTagger,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "RRJointTagger|redrocket": "RedRocket Tagger 🐺",
}
