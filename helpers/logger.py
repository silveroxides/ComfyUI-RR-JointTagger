import logging
import os
from typing import Optional

from .metaclasses import Singleton



class ComfyLogger(metaclass=Singleton):
    """
    A simple logger class for comfy extensions that logs to stdout and a file.
    """
    def __init__(self) -> None:
        from .config import ComfyExtensionConfig
        from .extension import ComfyExtension
        config = ComfyExtensionConfig().get()
        name = config["name"]
        self.logger = logging.getLogger(f"{name}")
        self.logger.setLevel(self.log_level())

        # Stream Handler for console output
        self.logger.addHandler(logging.StreamHandler())

        # File Handler for logging to a file
        if self.is_logging_enabled():
            try:
                log_dir = ComfyExtension.extension_dir("logs")
                # Ensure the logs directory exists
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                log_file = os.path.join(log_dir, "ComfyUI-RR-JointTagger.log")
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to setup file logging: {e}")

        self.logger.propagate = False

    @classmethod
    def log(cls, message: str, type: Optional[str] = None, always: bool = False) -> None:
        if not always or not cls().is_logging_enabled():
            return
        if type is not None:
            type_match = {
                "info": cls().logger.info,
                "warning": cls().logger.warning,
                'warn': cls().logger.warning,   # alias for dev laziness
                "error": cls().logger.error,
                "debug": cls().logger.debug
            }
            log_fn = type_match.get(type.lower(), cls().logger.info)
        else:
            log_fn = cls().logger.info
        log_fn(message)

    @classmethod
    def log_level(cls) -> int:
        from .config import ComfyExtensionConfig
        config = ComfyExtensionConfig().get()
        if "loglevel" not in config:
            return logging.INFO
        return getattr(logging, config["loglevel"].upper())

    @classmethod
    def is_logging_enabled(cls) -> bool:
        from .config import ComfyExtensionConfig
        config = ComfyExtensionConfig().get()
        if "logging" not in config:
            return False
        return config["logging"]