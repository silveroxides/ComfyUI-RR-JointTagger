import csv
import gc
import os
import random
import traceback
from typing import Dict, List, Set, Tuple, Union

from ..helpers.cache import CacheCleanupMethod, ComfyCache
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton

class Furrence2CaptionManager(metaclass=Singleton):
    """
    The Furrence2 Caption Manager handles loading tag constraints (Pruner) and formatting
    tags into the expected prompt string for Furrence-2.
    """
    def __init__(self, tags_basepath: str) -> None:
        self.tags_basepath = tags_basepath
        ComfyCache.set_max_size('cap_furrence2', 1)
        ComfyCache.set_cachemethod('cap_furrence2', CacheCleanupMethod.ROUND_ROBIN)

    def __del__(self) -> None:
        ComfyCache.flush('cap_furrence2')
        gc.collect()

    @classmethod
    def is_loaded(cls, tags_name: str) -> bool:
        return ComfyCache.get(f'cap_furrence2.{tags_name}') is not None

    @classmethod
    def load(cls, tags_name: str) -> bool:
        # Resolve path - we expect tags_name to be something like 'tags-2024-05-05'
        # The file is saved inside the model directory by ModelManager
        from .model_furrence2_manager import Furrence2ModelManager
        
        tags_path = os.path.join(cls().tags_basepath, "furrence2-large", f"{tags_name}.csv")
        
        if cls.is_loaded(tags_name):
            return True
            
        if not os.path.exists(tags_path):
            ComfyLogger().log(f"Furrence2 tags CSV not found at {tags_path}. Checking model installation...", "INFO", True)
            # Try to trigger download via ModelManager if missing
            if not Furrence2ModelManager.is_installed("furrence2-large"):
                if not Furrence2ModelManager.download("furrence2-large"):
                    return False
            
            # Re-check after download
            if not os.path.exists(tags_path):
                ComfyLogger().log(f"Furrence2 tags CSV still missing after download attempt: {tags_path}", "ERROR", True)
                return False

        try:
            species_tags = set()
            allowed_tags = set()
            with open(tags_path, "r", encoding="utf-8") as f:
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

            ComfyCache.set(f'cap_furrence2.{tags_name}', {
                "species_tags": species_tags,
                "allowed_tags": allowed_tags
            })
            ComfyLogger().log(f"Loaded Furrence2 tags constraints for {tags_name}", "INFO", True)
            return True
        except Exception as err:
            ComfyLogger().log(f"Error loading Furrence2 tags: {err}\n{traceback.format_exc()}", "ERROR", True)
            return False

    @classmethod
    def format_prompt(cls, tags_name: str, tags: Union[str, List[str]], length: int) -> str:
        """
        Prunes disallowed tags and constructs the Florence-2 prompt.
        """
        if not cls.is_loaded(tags_name):
            cls.load(tags_name)

        data = ComfyCache.get(f'cap_furrence2.{tags_name}')
        if not data:
            ComfyLogger().log(f"Cannot format prompt: constraints for {tags_name} not loaded.", "ERROR", True)
            return ""

        allowed_tags: Set[str] = data["allowed_tags"]
        species_tags: Set[str] = data["species_tags"]

        if isinstance(tags, str):
            # The input might be comma-separated or space-separated based on the tagger node output.
            # Usually the tagger node outputs comma-separated.
            tags = [t.strip().replace(" ", "_") for t in tags.replace(",", " ").split() if t.strip()]

        # Replace underscores back to spaces if needed, but the pruner expects whatever is in the CSV.
        # The CSV has spaces for names usually, let's normalize to spaces just in case,
        # actually, the pruner uses CSV name which might be raw.
        # If the input tags came from RRJointTagger, they could be space or underscore depending on `replace_underscore`.
        # Assuming the CSV has spaces for tags or underscores... Let's just match as is, but handle both if needed.
        # In references/app.py: tag names are matched exactly.

        random.shuffle(tags)

        # _prune_not_allowed_tags
        this_allowed_tags = []
        for tag in tags:
            tag_space = tag.replace("_", " ")
            if tag_space in allowed_tags:
                this_allowed_tags.append(tag_space)
            elif tag in allowed_tags:
                this_allowed_tags.append(tag)

        # _find_and_format_species_tags
        this_specie_tags = []
        for tag in this_allowed_tags:
            if tag in species_tags:
                this_specie_tags.append(tag)

        formatted_species_tags = f"species: {' '.join(this_specie_tags)}\n"

        non_species_tags = [t for t in this_allowed_tags if t not in this_specie_tags]
        prompt = f"{' '.join(non_species_tags)}\n{formatted_species_tags}\nlength: {length}\n\nSTYLE1 FURRY CAPTION:"
        return prompt
