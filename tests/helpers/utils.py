import math
from numbers import Number
from typing import Any, Dict, List, Union


def is_close(ret: Number, expected: Number, rel_tol: float = 1e-4, abs_tol: float = 0.0):
    return math.isclose(ret, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def all_close(
    datum_1: Dict[str, Union[Number, List[Number]]],
    datum_2: Dict[str, Union[Number, List[Number]]],
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-4,
):
    if set(datum_1.keys()) != set(datum_2.keys()):
        return False
    for key, value1 in datum_1.items():
        if isinstance(value1, list):
            if not all(math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol) for v1, v2 in zip(value1, datum_2[key])):
                return False
        else:
            if not math.isclose(value1, datum_2[key], rel_tol=rel_tol, abs_tol=abs_tol):
                return False
    return True


def in_zero_one(ret: Union[Number, Dict[str, Number]]):
    if isinstance(ret, Number):
        return ret >= 0 and ret <= 1
    else:
        return all(v >= 0 and v <= 1 for v in ret.values())


def list_of_dicts_to_dict_of_lists(data: List[Dict[str, Any]]):
    # Initialize the result dictionary
    result = {}
    for item in data:
        for key, value in item.items():
            # If the key doesn't exist in the result dictionary, initialize it with a list
            if key not in result:
                result[key] = []
            # Append the value to the list associated with the key
            result[key].append(value)
    return result
