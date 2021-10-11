import json
import commentjson
import os
import pickle
from pathlib import Path, PosixPath


def read_commentjson(path):
    with open(path, mode="r") as f:
        return commentjson.load(f)


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


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_text(path):
    with open(path) as f:
        return [l.replace("\n", "") for l in f.readlines()]


def write_pkl(data, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)