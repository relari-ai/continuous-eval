from typing import Type, get_origin

# def infer_type_hint(variable: Any) -> str:
#     """Infers the type hint of a variable."""
#     if isinstance(variable, list):
#         # Infer the type of list elements
#         if variable:
#             element_types = {infer_type_hint(elem) for elem in variable}
#             # Assuming all elements are of the same type for simplicity
#             if len(element_types) == 1:
#                 element_type = element_types.pop()
#             else:
#                 element_type = 'Any'
#         else:
#             element_type = 'Any'
#         return f'List[{element_type}]'
#     elif isinstance(variable, dict):
#         # Infer the type of keys and values
#         key_types = {infer_type_hint(key) for key in variable.keys()}
#         value_types = {infer_type_hint(value) for value in variable.values()}
#         key_type = key_types.pop() if len(key_types) == 1 else 'Any'
#         value_type = value_types.pop() if len(value_types) == 1 else 'Any'
#         return f'Dict[{key_type}, {value_type}]'
#     else:
#         # Direct type mapping for non-collection types
#         return type(variable).__name__


def type_hint_to_str(type_hint: Type):
    if hasattr(type_hint, '__origin__'):  # Check if it's a generic type
        # Get the base type name (e.g., 'List' or 'Dict')
        base = type_hint.__origin__.__name__.title()
        # Recursively process the arguments (e.g., the contents of List, Dict, etc.)
        args = ", ".join(type_hint_to_str(arg) for arg in type_hint.__args__)
        return f"{base}[{args}]"
    elif hasattr(type_hint, '__name__'):
        return type_hint.__name__
    else:
        return repr(type_hint)


def instantiate_type(type_hint: Type):
    origin = get_origin(type_hint)
    # If the origin is None, it means type_hint is not a generic type
    # and we assume type_hint itself is directly instantiable
    if origin is None:
        origin = type_hint
    try:
        # This only works for types without required arguments in their __init__.
        instance = origin()
    except TypeError as e:
        # If instantiation fails, return an error message or raise a custom exception
        instance = None
    return instance
