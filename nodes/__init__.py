from .rrtagger import NODE_CLASS_MAPPINGS as MAPPINGS_V1, NODE_DISPLAY_NAME_MAPPINGS as NAMES_V1
from .rrtagger_v3 import Jtp3HydraTagger, JTP3CategoryConfig
from .rrtagger_dino import DINOv3Tagger, DINOv3CategoryConfig

NODE_CLASS_MAPPINGS = {
    **MAPPINGS_V1,
    "Jtp3HydraTagger|redrocket": Jtp3HydraTagger,
    "JTP3CategoryConfig|redrocket": JTP3CategoryConfig,
    "DINOv3Tagger|redrocket": DINOv3Tagger,
    "DINOv3CategoryConfig|redrocket": DINOv3CategoryConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NAMES_V1,
    "Jtp3HydraTagger|redrocket": "JTP-3 Hydra Tagger 🐺",
    "JTP3CategoryConfig|redrocket": "JTP-3 Category Config 🐺",
    "DINOv3Tagger|redrocket": "DINOv3 Taggerine by Lodestone 🐺",
    "DINOv3CategoryConfig|redrocket": "DINOv3 Taggerine Category Config 🐺",
}
