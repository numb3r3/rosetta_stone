import importlib
import itertools
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional, Union

import py


def import_path(p: Union[str, py.path.local, Path]) -> ModuleType:
    """Imports and returns a module from the given path, which can be a file (a
    module) or a directory (a package)."""
    path = Path(str(p))

    if not path.exists():
        raise ImportError(path)

    pkg_path = resolve_package_path(path)
    if pkg_path is not None:
        pkg_root = pkg_path.parent
        names = list(path.with_suffix('').relative_to(pkg_root).parts)
        if names[-1] == '__init__':
            names.pop()
        module_name = '.'.join(names)
    else:
        pkg_root = path.parent
        module_name = path.stem

    # change sys.path permanently
    if str(pkg_root) != sys.path[0]:
        sys.path.insert(0, str(pkg_root))

    importlib.import_module(module_name)
    mod = sys.modules[module_name]

    return mod


def resolve_package_path(path: Path) -> Optional[Path]:
    """Return the Python package path by looking for the last directory upwards
    which still contains an __init__.py.

    Return None if it can not be determined.
    """
    result = None
    for parent in itertools.chain((path, ), path.parents):
        if parent.is_dir():
            if not parent.joinpath('__init__.py').is_file():
                break
            if not parent.name.isidentifier():
                break
            result = parent
    return result
