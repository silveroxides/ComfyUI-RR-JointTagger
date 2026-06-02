import csv
import random
import os
from typing import Tuple, Dict, Any
from comfy.comfy_types import IO, ComfyNodeABC
import folder_paths




model_basepath = os.path.join(folder_paths.models_dir, "RedRocket")
tags_basepath = os.path.join(model_basepath, "tags")
pruner_tag_list_path = os.path.join(tags_basepath, "tags-2024-05-05.csv")

def download_pruner_tag_list():
    if not os.path.exists(pruner_tag_list_path):
        import requests
        import json
        extension_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        config_path = os.path.join(extension_path, "config.json")

        if not os.path.exists(config_path):
            print(f"ComfyUI-RR-JointTagger: config.json not found at {config_path}")
            return False

        with open(config_path, "r") as f:
            config = json.load(f)

        endpoint = config.get("huggingface_endpoint", "https://huggingface.co")
        tag_list_url = config["pruner"]["tag_list_url"].format(HF_ENDPOINT=endpoint)

        try:
            response = requests.get(tag_list_url)
            response.raise_for_status()

            os.makedirs(os.path.dirname(pruner_tag_list_path), exist_ok=True)
            with open(pruner_tag_list_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"ComfyUI-RR-JointTagger: Error downloading tag list: {e}")
            return False
    return False


category_id_to_str = {
        "0": "general",
        # 3 copyright
        "4": "character",
        "5": "species",
        "7": "meta",
        "8": "lore",
        "1": "artist",
    }

def create_tag_sets(path_to_tag_list_csv):
    species_tags = set()
    allowed_tags = set()
    if not os.path.exists(path_to_tag_list_csv):
        print(f"ComfyUI-RR-JointTagger: Tag list CSV not found at {path_to_tag_list_csv}")
        return species_tags, allowed_tags
    with open(path_to_tag_list_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        name_index = header.index("name")
        category_index = header.index("category")
        post_count_index = header.index("post_count")
        for row in reader:
            if int(row[post_count_index]) > 20:
                category = row[category_index]
                name = row[name_index]
                if category == "5":
                    species_tags.add(name)
                    allowed_tags.add(name)
                elif category == "0":
                    allowed_tags.add(name)
                elif category == "7":
                    allowed_tags.add(name)

    species_tags = species_tags
    allowed_tags = allowed_tags
    return species_tags, allowed_tags

def prune_not_allowed_tags(raw_tags, allowed_tags):
    this_allowed_tags = set()
    for tag in raw_tags:
        if tag in allowed_tags:
            this_allowed_tags.add(tag)
    return this_allowed_tags

def find_and_format_species_tags(tag_set, species_tags):
    this_specie_tags = []
    for tag in tag_set:
        if tag in species_tags:
            this_specie_tags.append(tag)

    formatted_tags = f"species: {' '.join([t for t in this_specie_tags])}\n"
    return formatted_tags, this_specie_tags

def prompt_construction_pipeline_florence2(tags, length, species_tags, allowed_tags):
    if type(tags) is str:
        tags = tags.split(" ")
    random.shuffle(tags)
    tags = prune_not_allowed_tags(tags, allowed_tags)
    formatted_species_tags, this_specie_tags = find_and_format_species_tags(tags, species_tags)
    non_species_tags = [t for t in tags if t not in this_specie_tags]
    prompt = f"{' '.join(non_species_tags)}\n{formatted_species_tags}\nlength: {length}\n\nSTYLE1 FURRY CAPTION:"
    return prompt


class RRTagPruner(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tags": (IO.STRING, {"multiline": True, "tooltip": "A space or comma separated list of tags with underscores."}),
                "length": (IO.INT, {"tooltip": "The length of the caption."}),
            }
        }


    RETURN_TYPES = (IO.STRING, )
    FUNCTION = "prune_tags"
    OUTPUT_NODE = True
    CATEGORY = "🐺 Furry Diffusion"

    def prune_tags(self, tags: str, length: int) -> Tuple[str, str]:
        if ", " in tags:
            tags = tags.replace(", ", " ")
        elif "," in tags:
            tags = tags.replace(",", " ")
        download_pruner_tag_list()
        path_to_tag_list_csv = pruner_tag_list_path
        species_tags, allowed_tags = create_tag_sets(path_to_tag_list_csv)
        prompt = prompt_construction_pipeline_florence2(tags, length, species_tags, allowed_tags)
        return prompt, ""


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "RRTagPruner|redrocket": RRTagPruner,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "RRTagPruner|redrocket": "RedRocket Tag Pruner 🐺",
}
