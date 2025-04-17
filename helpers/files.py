import os

from .metaclasses import Singleton


class ComfyFiles(metaclass=Singleton):
    def __init__(self) -> None:
        pass

    @classmethod
    def link_item(cls, src: str, dst: str) -> bool:
        """
        Helper method to create a symlink or junction from source to destination
        """
        from .logger import ComfyLogger
        src = os.path.abspath(src)
        dst = os.path.abspath(dst)
        if os.name == "nt":
            try:
                import _winapi
                _winapi.CreateJunction(src, dst)
                ComfyLogger().log(f"Created junction from {src} to {dst}")
                return True
            except Exception as e:
                ComfyLogger().log(f"Failed to create junction from {src} to {dst}\nException: {e}", type="ERROR", always=True)
                return False
        else:
            try:
                os.symlink(src, dst)
                ComfyLogger().log(f"Created symlink from {src} to {dst}")
                return True
            except Exception as e:
                ComfyLogger.log(f"Failed to create symlink from {src} to {dst}\nException: {e}", type="ERROR", always=True)
                return False

    @classmethod
    def is_symlink(cls, path: str) -> bool:
        """
        Helper method to check if a path is a symlink or junction
        """
        from .logger import ComfyLogger
        try:
            if os.path.islink(path):
                return True
            else:
                return False
        except Exception as e:
            ComfyLogger().log(f"Failed to read link from {path}\nException: {e}", type="ERROR", always=True)
            return False
