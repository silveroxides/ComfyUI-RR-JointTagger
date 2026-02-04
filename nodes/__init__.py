from .rrtagger import NODE_CLASS_MAPPINGS as MAPPINGS_V1, NODE_DISPLAY_NAME_MAPPINGS as NAMES_V1
from .rrtagger_v3 import Jtp3HydraTagger

NODE_CLASS_MAPPINGS = {
    **MAPPINGS_V1,
    "Jtp3HydraTagger|redrocket": Jtp3HydraTagger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NAMES_V1,
    "Jtp3HydraTagger|redrocket": "JTP-3 Hydra Tagger 🐺"
}
