import builtins
import typing
from typing import Any, Type, get_origin

from pydantic import ValidationError, create_model


def str_to_type_hint(type_str: str):
    # Utility to safely get type from typing module or builtins
    def get_type(type_name):
        if hasattr(builtins, type_name):
            return getattr(builtins, type_name)
        elif hasattr(typing, type_name):
            return getattr(typing, type_name)
        else:
            raise ValueError(
                f"Type {type_name} not found in builtins or typing module"
            )

    # Base case: simple type without parameters
    if "[" not in type_str:
        return get_type(type_str)
    # Recursive case: generic type with parameters
    base_type_str, params_str = type_str.split("[", 1)
    params_str = params_str.rstrip("]")  # Remove the closing bracket
    # Recursively convert parameter strings back to type hints
    params = [
        str_to_type_hint(param.strip()) for param in params_str.split(",")
    ]
    base_type = get_type(base_type_str.strip())
    # Reconstruct the generic type with its parameters
    return base_type[tuple(params)]


def type_hint_to_str(type_hint: Type):
    if hasattr(type_hint, "__origin__"):  # Check if it's a generic type
        # Get the base type name (e.g., 'List' or 'Dict')
        base = type_hint.__origin__.__name__.title()
        # Recursively process the arguments (e.g., the contents of List, Dict, etc.)
        args = ", ".join(type_hint_to_str(arg) for arg in type_hint.__args__)
        return f"{base}[{args}]"
    elif hasattr(type_hint, "__name__"):
        return type_hint.__name__
    else:
        return type_hint if isinstance(type_hint, str) else repr(type_hint)


def instantiate_type(type_hint: Type):
    origin = get_origin(type_hint)
    # If the origin is None, it means type_hint is not a generic type
    # and we assume type_hint itself is directly instantiable
    if origin is None:
        origin = type_hint
    try:
        # This only works for types without required arguments in their __init__.
        instance = origin()
    except TypeError:
        # If instantiation fails, return an error message or raise a custom exception
        instance = None
    return instance


def check_type(var: Any, type_hint: Any) -> bool:
    """
    Checks if 'var' matches the 'type_hint'.

    Args:
    var (Any): The variable to check.
    type_hint (Any): The type hint (from the typing module) against which to check the variable.

    Returns:
    bool: True if 'var' matches the 'type_hint', False otherwise.
    """
    # Dynamically create a Pydantic model with one field 'data' of the provided type hint
    DynamicModel = create_model("DynamicModel", data=(type_hint, ...))
    try:
        # Create an instance of the model with 'var' as the data to validate the type
        DynamicModel(data=var)
        return True
    except ValidationError:
        return False
