from .rrtagger import NODE_CLASS_MAPPINGS as MAPPINGS_V1, NODE_DISPLAY_NAME_MAPPINGS as NAMES_V1
from .rrtagger_v3 import Jtp3HydraTagger
from .rrtagger_dino import DINOv3Tagger, DINOv3CategoryConfig

NODE_CLASS_MAPPINGS = {
    **MAPPINGS_V1,
    "Jtp3HydraTagger|redrocket": Jtp3HydraTagger,
    "DINOv3Tagger|redrocket": DINOv3Tagger,
    "DINOv3CategoryConfig|redrocket": DINOv3CategoryConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NAMES_V1,
    "Jtp3HydraTagger|redrocket": "JTP-3 Hydra Tagger 🐺",
    "DINOv3Tagger|redrocket": "DINOv3 Tagger 🐺",
    "DINOv3CategoryConfig|redrocket": "DINOv3 Category Config 🐺",
}
