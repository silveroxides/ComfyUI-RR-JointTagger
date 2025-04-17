import logging
from typing import Optional
from colorama import init as colorama_init
from colorama import Fore, Back, Style

from .metaclasses import Singleton


class CustomFormatter(logging.Formatter):
    """
    A custom formatter for logging messages.
    """
    def format(self, record: logging.LogRecord) -> str:
        color = {
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT
        }
        record.levelname = f"{color[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        return f"🐺 [{record.levelname}] {record.getMessage()}"

class ComfyLogger(metaclass=Singleton):
    """
    A simple logger class for comfy extensions that logs to stdout.
    """
    def __init__(self) -> None:
        from .config import ComfyExtensionConfig
        name = ComfyExtensionConfig().get()["name"]
        self.logger = logging.getLogger(f"{name}")
        self.logger.setLevel(self.log_level())
        self.logger.addHandler(logging.StreamHandler())
        self.logger.handlers[0].setFormatter(CustomFormatter())
        self.logger.propagate = False
        colorama_init(autoreset=True, strip=False, convert=True, wrap=True)
    
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