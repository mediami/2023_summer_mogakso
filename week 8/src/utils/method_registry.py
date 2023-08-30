import copy
from typing import Any, Callable, Dict

_METHODS_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    _METHODS_REGISTRY[fn.__name__] = fn
    return fn


def create_method(method_name):
    fn = _METHODS_REGISTRY.get(method_name, None)
    if fn:
        return fn
    else:
        raise NotImplementedError(f'{method_name} method is not implemented.')


def get_method_dict():
    method = copy.deepcopy(_METHODS_REGISTRY)
    method.pop('erf', None)
    method.pop('sah', None)
    return method
