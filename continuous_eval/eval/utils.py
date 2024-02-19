from typing import Type


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
