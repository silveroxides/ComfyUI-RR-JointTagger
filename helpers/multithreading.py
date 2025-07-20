# import asyncio
from typing import Any, Callable, List, Optional

from .metaclasses import Singleton


# class ComfyThreading(metaclass=Singleton):
#     """
#     A singleton class to provide threading functionality for a comfy extension.
#     """
#     def __init__(self) -> None:
#         pass
# 
#     @classmethod
#     def wait_for_async(cls, async_fn: Callable[[], Any], loop: Optional[asyncio.AbstractEventLoop] = None) -> Any:
#         res: List[Any] = []
#         async def run_async() -> None:
#             r = await async_fn()
#             res.append(r)
#         if loop is None:
#             try:
#                 loop = asyncio.get_event_loop()
#             except:
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)
#         loop.run_until_complete(run_async())
#         return res[0]

