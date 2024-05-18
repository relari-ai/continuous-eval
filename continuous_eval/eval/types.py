from typing import Any, Dict, TypedDict

UID = str


def args_to_dict(*args, **kwargs):
    arg_dict = dict(kwargs)
    for index, value in enumerate(args):
        arg_dict[f'_arg{index}'] = value
    return arg_dict


class ToolCall(TypedDict):
    name: str
    kwargs: Dict[str, Any]
