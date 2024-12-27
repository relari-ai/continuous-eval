def all_sets_equal(d: dict) -> bool:
    return len(set(map(frozenset, d.values()))) == 1
