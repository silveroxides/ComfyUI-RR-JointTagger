import os
import inspect
from typing import List, Optional, Union
import comfy.utils
from server import PromptServer

from .metaclasses import Singleton


class ComfyExtension(metaclass=Singleton):
    """
    A singleton class to provide utility functions for a comfy extension.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def name(cls) -> str:
        """
        Get the name of the extension
        """
        from .config import ComfyExtensionConfig
        return str(ComfyExtensionConfig().get()["name"]).lower().replace(" ", "_").replace('/', ".")

    @classmethod
    def extension_dir(cls, subpath: Optional[str] = None, mkdir: bool = False) -> str:
        """
        Get the directory the extension is installed in
        """
        dir = os.path.dirname(__file__).partition("helpers")[0]
        if subpath is not None:
            dir = os.path.join(dir, subpath)
        dir = os.path.abspath(dir)
        if mkdir and not os.path.exists(dir):
            from .logger import ComfyLogger
            ComfyLogger().log(f"Directory {dir} does not exist, it is being created", type="WARNING", always=True)
            os.makedirs(dir)
        return dir

    @classmethod
    def comfy_dir(cls, subpath: Optional[str] = None) -> str:
        """
        Get the directory the extension is installed in
        """
        dir = os.path.dirname(inspect.getfile(PromptServer))
        if subpath is not None:
            dir = os.path.join(dir, subpath)
        dir = os.path.abspath(dir)
        return dir
