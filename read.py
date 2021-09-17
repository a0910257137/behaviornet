from itertools import count
import json
import numpy as np
from pprint import pprint


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


annos = load_json("/aidata/anders/objects/hico/annos/train.json")
tmp_cates = []
for anno in annos['frame_list']:
    for lb in anno['labels']:
        tmp_cates += [lb['category']]
items, counts = np.unique(tmp_cates, return_counts=True)