import inspect
import json
import os
import os.path as osp
import re
import shutil
import string
import sys
import types
from importlib import import_module
from pathlib import Path, PosixPath
from typing import Tuple, Union
from .typing import PathTypes, path_classes


def load_json(path):
    """The function of loading json file

    Arguments:
        path {str} -- The path of the json file

    Returns:
        list, dict -- The obj stored in the json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump_json(data, path):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def json_serializing(config: Union[list, dict]):
    """Try to make items in list or dict json serializable
    For now, it's mainly for config part (especially lambda function)

    Args:
        config (Union[list, dict]): config

    Returns:
        Union[list, dict]: json serializable config
    """
    # TODO: deal with more non-serializable types
    __default_key__ = 'default'
    if not isinstance(config, dict):
        config = {__default_key__: config}

    output = {}
    for k, v in config.items():
        if isinstance(k, tuple):
            k = json_serializing(k)
        if isinstance(v, types.LambdaType):
            lambda_func = inspect.getsource(v).strip()
            if lambda_func[-1] in string.punctuation:
                lambda_func = lambda_func[:-1]
            index = re.search(r'(lambda)\s', lambda_func)
            output.update({k: lambda_func[index.start():]})
        elif isinstance(v, dict):
            output.update({k: json_serializing(v)})
        elif isinstance(v, PosixPath):
            output.update({k: str(v)})
        elif isinstance(v, list):
            output.update({k: [json_serializing(vv) for vv in v]})
        else:
            output.update({k: v})

    return output if __default_key__ not in output else output[__default_key__]


def path_converting(path: Union[str, tuple]) -> str:
    """reshape path tuple (from bddhelper) to str

    Args:
        path (Union[str, tuple]): path

    Returns:
        str: transformed path
    """
    # TODO: ret_path doesn't contain data-root now. data-root part will be implemented in the near future.
    assert isinstance(
        path, (tuple, str)
    ), f'Expected frame_id be a tuple which contains folders and filename, or a str'
    ret_path = path if isinstance(path, str) else os.path.join(*path)
    return ret_path


def copy(src: Union[str, Path], dst: Union[str, Path], *args, **kwargs):
    """
    copy file or directory
    """
    src = str(src)
    dst = str(dst)
    is_file = osp.isfile(src)
    if is_file:
        return shutil.copy(src, dst, *args, **kwargs)
    else:
        return shutil.copytree(src, dst, *args, **kwargs)


def copy_and_parse_py_config(path: PathTypes,
                             tmp_file_path: PathTypes) -> Tuple[dict, dict]:
    """Copy .py config file to the other file[
        suppose with random file name
    ] and parse the contents into dict

    Args:
        path ([type]): [description]
        tmp_file_path ([type]): [description]

    Returns:
        data, json-serializable-data
    """
    copy(path, tmp_file_path)
    path = Path(tmp_file_path)
    assert path.exists() and path.suffix == '.py'
    module_name = path.stem
    sys.path.insert(0, str(path.parent))
    module = import_module(module_name)
    sys.path.pop(0)
    data = {k: v for k, v in module.__dict__.items() if not k.startswith('_')}
    del sys.modules[module_name]
    json_serialized_data = json_serializing(data)
    return data, json_serialized_data
