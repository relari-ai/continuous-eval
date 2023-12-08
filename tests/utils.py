import math
from numbers import Number
from typing import Dict


def all_close(
    datum_1: Dict[str, Number],
    datum_2: Dict[str, Number],
    rel_tol: float = 1e-4,
    abs_tol: float = 0.0,
):
    if set(datum_1.keys()) != set(datum_2.keys()):
        return False
    for key, value1 in datum_1.items():
        if not math.isclose(value1, datum_2[key], rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


def in_zero_one(ret):
    return all(0 <= v <= 1 for v in ret.values())
