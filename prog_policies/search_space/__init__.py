from __future__ import annotations

from .base_space import BaseSearchSpace
from .latent_space import LatentSpace
from .latent_space2 import LatentSpace2
from .programmatic_space import ProgrammaticSpace

def get_search_space_cls(search_space_cls_name: str) -> type[BaseSearchSpace]:
    search_space_cls = globals().get(search_space_cls_name)
    assert issubclass(search_space_cls, BaseSearchSpace)
    return search_space_cls
