import gc
import time
from typing import Any, Dict, Optional, Union
from enum import Enum

from ..helpers.metaclasses import Singleton


class CacheCleanupMethod(Enum):
    ROUND_ROBIN = 'round_robin'
    LEAST_USED = 'least_used'
    MOST_OFTEN_USED = 'most_often_used'


class ComfyCache(metaclass=Singleton):
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Union[int, float, CacheCleanupMethod, Optional[int]]]] = {}

    def __del__(self) -> None:
        self.data.clear()
        self.metadata.clear()
        del self.data
        del self.metadata
        gc.collect()

    @classmethod
    def cache_pop(cls, property: str) -> Any:
        method = cls.get_cachemethod(property)
        if method == CacheCleanupMethod.ROUND_ROBIN:
            return cls.cache_pop_roundrobin(property)
        elif method == CacheCleanupMethod.LEAST_USED:
            return cls.cache_pop_least_used(property)
        elif method == CacheCleanupMethod.MOST_OFTEN_USED:
            return cls.cache_pop_most_often_used(property)
        else:
            raise ValueError("Unsupported cache cleanup method")

    @classmethod
    def cache_pop_roundrobin(cls, property: str) -> Any:
        instance = cls()
        if property not in instance.metadata:
            return None
        oldest_property = min(instance.metadata[property].items(), key=lambda item: item[1]['timestamp'])[0]
        value = instance.data[property].pop(oldest_property)
        instance.metadata[property].pop(oldest_property)
        if not instance.data[property]:
            del instance.data[property]
            del instance.metadata[property]
        return value

    @classmethod
    def cache_pop_least_used(cls, property: str) -> Any:
        instance = cls()
        if property not in instance.metadata:
            return None
        least_used_property = min(instance.metadata[property].items(), key=lambda item: item[1]['usage'])[0]
        value = instance.data[property].pop(least_used_property)
        instance.metadata[property].pop(least_used_property)
        if not instance.data[property]:
            del instance.data[property]
            del instance.metadata[property]
        return value

    @classmethod
    def cache_pop_most_often_used(cls, property: str) -> Any:
        instance = cls()
        if property not in instance.metadata:
            return None
        most_often_used_property = max(instance.metadata[property].items(), key=lambda item: item[1]['usage'])[0]
        value = instance.data[property].pop(most_often_used_property)
        instance.metadata[property].pop(most_often_used_property)
        if not instance.data[property]:
            del instance.data[property]
            del instance.metadata[property]
        return value

    @classmethod
    def get(cls, property: Optional[str] = None) -> Any:
        instance = cls()
        cache = instance.data
        metadata = instance.metadata

        if property is not None:
            keys = property.split(".")
            for key in keys:
                if key not in cache:
                    return None
                cache = cache[key]
                if property not in metadata:
                    metadata[property] = {}
                if key not in metadata[property]:
                    metadata[property][key] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
                metadata[property][key]['timestamp'] = time.time()
                metadata[property][key]['usage'] += 1
            return cache

        return cache

    @classmethod
    def set(cls, property: Union[str, None], value: Any) -> None:
        instance = cls()
        if property is not None:
            keys = property.split(".")
            cache = instance.data
            metadata = instance.metadata

            for key in keys[:-1]:
                if key not in cache:
                    cache[key] = {}
                cache = cache[key]
                if property not in metadata:
                    metadata[property] = {}
                if key not in metadata[property]:
                    metadata[property][key] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
                metadata[property][key]['timestamp'] = time.time()
                metadata[property][key]['usage'] += 1

            if property not in metadata:
                metadata[property] = {}
            if keys[-1] not in metadata[property]:
                metadata[property][keys[-1]] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
            metadata[property][keys[-1]]['timestamp'] = time.time()
            metadata[property][keys[-1]]['usage'] += 1

            max_size = metadata[property].get(keys[-1], {}).get('max_size')
            if max_size is not None and len(cache) >= max_size:
                cls.cache_pop(property)

            cache[keys[-1]] = value
        else:
            instance.data = value
            instance.metadata = {}

        return

    @classmethod
    def set_cachemethod(cls, property: str, method: CacheCleanupMethod) -> None:
        instance = cls()
        keys = property.split(".")
        cache = instance.data
        metadata = instance.metadata

        for key in keys[:-1]:
            if key not in cache:
                cache[key] = {}
            cache = cache[key]
            if property not in metadata:
                metadata[property] = {}
            if key not in metadata[property]:
                metadata[property][key] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
            metadata[property][key]['timestamp'] = time.time()
            metadata[property][key]['usage'] += 1

        if property not in metadata:
            metadata[property] = {}
        if keys[-1] not in metadata[property]:
            metadata[property][keys[-1]] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
        metadata[property][keys[-1]]['cachemethod'] = method
        return

    @classmethod
    def get_cachemethod(cls, property: str) -> CacheCleanupMethod:
        instance = cls()
        keys = property.split(".")
        cache = instance.data
        metadata = instance.metadata

        for key in keys:
            if key not in cache:
                return CacheCleanupMethod.ROUND_ROBIN  # Default method if not set
            cache = cache[key]

        if property in metadata and keys[-1] in metadata[property]:
            return metadata[property][keys[-1]]['cachemethod']
        return CacheCleanupMethod.ROUND_ROBIN

    @classmethod
    def set_max_size(cls, property: str, max_size: int) -> None:
        instance = cls()
        keys = property.split(".")
        cache = instance.data
        metadata = instance.metadata

        for key in keys[:-1]:
            if key not in cache:
                cache[key] = {}
            cache = cache[key]
            if property not in metadata:
                metadata[property] = {}
            if key not in metadata[property]:
                metadata[property][key] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
            metadata[property][key]['timestamp'] = time.time()
            metadata[property][key]['usage'] += 1

        if property not in metadata:
            metadata[property] = {}
        if keys[-1] not in metadata[property]:
            metadata[property][keys[-1]] = {'timestamp': time.time(), 'usage': 0, 'cachemethod': CacheCleanupMethod.ROUND_ROBIN, 'max_size': None}
        metadata[property][keys[-1]]['max_size'] = max_size
        return

    @classmethod
    def get_max_size(cls, property: str) -> Optional[int]:
        instance = cls()
        keys = property.split(".")
        cache = instance.data
        metadata = instance.metadata

        for key in keys:
            if key not in cache:
                return None
            cache = cache[key]

        if property in metadata and keys[-1] in metadata[property]:
            return metadata[property][keys[-1]].get('max_size')
        return None

    @classmethod
    def flush(cls, property: str) -> None:
        instance = cls()
        keys = property.split(".")
        cache = instance.data
        metadata = instance.metadata

        for key in keys[:-1]:
            if key not in cache:
                return
            cache = cache[key]
            metadata = metadata.get(property, {})
            metadata = metadata.get(key, {})

        if keys[-1] in cache:
            del cache[keys[-1]]
            if keys[-1] in metadata:
                del metadata[keys[-1]]
