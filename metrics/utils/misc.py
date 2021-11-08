from collections import Counter
from functools import reduce
from typing import Any, Dict, List, Union

from bddhelper.objects import BaseLabelObject


def get_value_by_key(item: Union[dict, BaseLabelObject],
                     key: Union[dict, str]) -> Any:
    """get value of bdd object or dict by given key or nested key

    Args:
        item (Union[dict, BaseLabelObject]): [description]
        key (Union[dict, str]): [description]

    Returns:
        str: [description]
    """
    assert isinstance(item, (
        dict,
        BaseLabelObject)), f'Expected dict or BaseLabelObject, got {type(item)}'
    assert isinstance(key,
                      (str, dict)), f'Expected str or dict, got {type(key)}'
    if isinstance(key, dict):
        assert len(key) == 1, 'Only one key is allowed'
        k, v = next(iter(key.items()))
        next_item = item[k] if isinstance(item, dict) else getattr(item, k)
        return next_item if not isinstance(v, dict) else get_value_by_key(
            next_item, v)
    else:
        return item[key] if isinstance(item, dict) else getattr(item, key)


def set_value_by_key(item: Union[dict, BaseLabelObject], key: dict):
    """set value by given key (only one key allowed)

    target = {'a': 1, 'b':{'b1':1, 'b2':2}}
    key = {'b':{'b1':2}}
    set_value_by_key(target, key)
    
    target: {'a': 1, 'b': {'b1': 2, 'b2': 2}} 
    """

    assert isinstance(item, (dict, BaseLabelObject)) and isinstance(key, dict)
    assert len(key) == 1, 'Only one key is allowed'
    k = next(iter(key))
    v = key[k]
    if isinstance(item, BaseLabelObject):
        if k == 'attributes':
            item.update_attributes(v)
        elif k == 'shape':
            item.set_shape(v)
        else:
            raise KeyError(
                f'If there is a bdd object, the valid key should be one of ["attributes", "shape"]. Got {k}'
            )
    else:
        if isinstance(v, dict):
            set_value_by_key(item[k], v)
        else:
            item[k] = v


def set_last_value(data: dict, value: Any):
    """set the `value` as the last key's value
    
    data = {'a':{'b':1}}
    value = 5 
    set_last_value(data, value) 
    
    data: {'a':{'b':5}}
    
    """
    assert isinstance(data, dict)
    assert len(data) == 1, 'Only one key is allowed'
    for k, v in data.items():
        if isinstance(v, dict):
            set_last_value(data[k], value)
        else:
            data[k] = value


def sum_over_list_of_dict(data: List[Dict[str, Union[int, float]]],
                          sort: bool = True):
    """sum up list of dict
    data = [{a:1, b:1, c:1}, {a:3, b:3, c:4, d:5}]

    Args:
        data (List[Dict[str, Union[int, float]]]): list of dict
        
    """
    assert isinstance(data, (tuple, list))
    assert all(isinstance(v, (int, float)) for d in data for v in d.values())
    sum_dict = reduce(lambda a, b: dict(Counter(a) + Counter(b)), data)
    if sort:
        sum_dict = dict(sorted(sum_dict.items()))
    return sum_dict
