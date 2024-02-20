from typing import Any, Dict, TypedDict

UUID = str


class ToolCall(TypedDict):
    name: str
    kwargs: Dict[str, Any]
