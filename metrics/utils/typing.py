from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Union

PathTypes = Union[str, Path, PosixPath]
path_classes = (str, Path, PosixPath)
# list of components, consisting vertices: list of list of int or float,
# closed: bool, knotvector: list of int, ...
Poly2D = List[Dict[str, Union[int, float, str, bool, List[int], List[List[
    Union[int, float]]]]]]
