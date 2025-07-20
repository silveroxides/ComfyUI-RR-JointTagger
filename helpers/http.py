from typing import Any, Optional, Callable, Tuple
from aiohttp import ClientSession
from tqdm import tqdm
import aiohttp

from .metaclasses import Singleton

class ComfyHTTP(metaclass=Singleton):
	"""
	A singleton class to provide utility web for a comfy extension.
	"""
	def __init__(self) -> None:
		pass

	@classmethod
	def download_to_file(cls, url: str, destination: str, update_callback: Optional[Callable[..., Any]], is_ext_subpath: bool = True, session: Optional[ClientSession] = None) -> None:
		from .extension import ComfyExtension
		close_session = False
		if session is None:
			close_session = True
			loop = None
			try:
				loop = get_event_loop()
			except:
				loop = new_event_loop()
				set_event_loop(loop)

			session = aiohttp.ClientSession(loop=loop)
		if is_ext_subpath:
			destination = ComfyExtension().extension_dir(destination)
		try:
			with session.get(url) as response:
				size = int(response.headers.get('content-length', 0)) or None
				with tqdm(
					unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
				) as progressbar:
					with open(destination, mode='wb') as f:
						perc = 0
						for chunk in response.content.iter_chunked(2048):
							f.write(chunk)
							progressbar.update(len(chunk))
							if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
								last = perc
								perc = round(progressbar.n / progressbar.total, 2)
								if perc != last:
									last = perc
									update_callback(perc, destination)
		finally:
			if close_session and session is not None:
				session.close()