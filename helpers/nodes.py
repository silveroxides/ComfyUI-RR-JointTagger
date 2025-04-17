from importlib.util import spec_from_file_location, module_from_spec, spec_from_loader
import inspect
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

from server import PromptServer
from .metaclasses import Singleton


class ComfyNode(metaclass=Singleton):
    """
    A singleton class to provide additional node utility functions for a comfy extension.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def update_node_status(cls, client_id: Optional[str], node: Union[str, None], api_endpoint: Union[str, None], text: Optional[str] = None, progress: Optional[float] = None) -> None:
        """
		Update the status of a node in the Comfy UI
        """
        from .extension import ComfyExtension
        from .config import ComfyExtensionConfig
        if client_id is None:
            client_id = ComfyExtension().client_id()
        if client_id is None:
            raise ValueError("Client ID is not set")
        if api_endpoint is None:
            api_endpoint = ComfyExtensionConfig().get(property="api_endpoint")
        if api_endpoint is None:
            raise ValueError("API endpoint is not set")
        PromptServer.instance.send_sync(f"{api_endpoint}/update_status", {
            "node": node,
            "progress": progress,
            "text": text
        }, client_id)

    @classmethod
    async def update_node_status_async(cls, client_id: Optional[str], node: Union[str, None], api_endpoint: Union[str, None], text: Optional[str] = None, progress: Optional[float] = None) -> None:
        """
		Update the status of a node in the Comfy UI asynchronously
        """
        from .extension import ComfyExtension
        from .config import ComfyExtensionConfig
        if client_id is None:
            client_id = ComfyExtension().client_id()
        if client_id is None:
            raise ValueError("Client ID is not set")
        if api_endpoint is None:
            api_endpoint = ComfyExtensionConfig().get(property="api_endpoint")
        if api_endpoint is None:
            raise ValueError("API endpoint is not set")
        await PromptServer.instance.send(f"{api_endpoint}/update_status", {
            "node": node,
            "progress": progress,
            "text": text
		}, client_id)
    
    @classmethod
    def get_module_vars(cls, module_path: str) -> Tuple[str, Dict[str, Any]]:
        """
		Get the declared variables in a module
        """
        module_dir, module_file = os.path.split(module_path)
        module_name, _ = os.path.splitext(module_file)
        abs_module_dir = os.path.abspath(module_dir)
        sys.path.insert(0, abs_module_dir)
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        package_parts = module_dir.split(os.sep)
        module.__package__ = '.'.join(package_parts[-2:])
        try:
            spec.loader.exec_module(module)
            module_vars = {name: value for name, value in vars(module).items() if not name.startswith('__') and not inspect.ismodule(value) and not inspect.isclass(value) and not inspect.isfunction(value)}
        finally:
            del sys.modules[module_name]
            sys.path = [p for p in sys.path if not p.startswith(abs_module_dir)]
        return module_name, module_vars

    @classmethod
    def get_node_vars(cls) -> Dict[str, Any]:
        """
        Search for Comfy UI related variables in the source files located in the `nodes` directory of the project, and return the results combined.
        """
        from .extension import ComfyExtension
        from .logger import ComfyLogger
        source_path = ComfyExtension().extension_dir("nodes", mkdir=False)
        vars = {}
        for file in os.listdir(source_path):
            if file.endswith(".py") and not file.startswith("__"):
                module_name, module_vars = cls.get_module_vars(os.path.join(source_path, file))
                for name, obj in module_vars.items():
                    if name.isupper() and name.isidentifier():
                        if isinstance(obj, str):
                            ComfyLogger().log(f"Loaded str({name})", type="DEBUG", always=True)
                            if name not in vars:
                                vars.update({name: obj})
                            else:
                                vars.update({name: [o for o in obj if obj not in vars[name]]})
                        elif isinstance(obj, dict):
                            ComfyLogger().log(f"Loaded dict({name}) in {module_name}", type="DEBUG", always=True)
                            for key, value in obj.items():
                                if name not in vars:
                                    vars.update({name: {key: value}})
                                else:
                                    vars[name].update({key: value})	
        return vars
