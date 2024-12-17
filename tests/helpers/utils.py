import math
from numbers import Number
from typing import Any, Dict, List, Union

from continuous_eval.metrics.base import Arg, Field
from continuous_eval.utils.types import check_type


def is_close(
    ret: Number, expected: Number, rel_tol: float = 1e-4, abs_tol: float = 0.0
):
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
            if not all(
                math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol)
                for v1, v2 in zip(value1, datum_2[key])
            ):
                return False
        else:
            if not math.isclose(
                value1, datum_2[key], rel_tol=rel_tol, abs_tol=abs_tol
            ):
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


def validate_schema(schema: Dict[str, Field], data: Dict[str, Any]):
    for key, value in schema.items():
        if key not in data:
            raise ValueError(f"Key {key} not found in data")
        if not check_type(data[key], value.type_hint):
            raise ValueError(
                f"Value {data[key]} for key {key} is not of type {value.type_hint}"
            )
        if value.limits is not None:
            assert (
                value.limits[0] <= data[key] <= value.limits[1]
            ), f"Value {data[key]} for key {key} is not in limits {value.limits}"
    return True


def validate_args(args: Dict[str, Arg]):
    assert isinstance(args, dict), "args must be a dictionary"
    for key, value in args.items():
        assert isinstance(
            key, str
        ), f"All keys in args must be strings, got {type(key)}"
        assert isinstance(
            value, Arg
        ), f"All values in args must be of type Field, got {type(value)}"
    return True
