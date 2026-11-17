"""Optional workspace composition limits, independent of scientific acceptance."""
from pathlib import Path

from ..discovery.common import load


def load_scope(root):
    root = Path(root).resolve()
    path = root/'research-material-scope.json'
    if not path.exists():
        return None
    if not path.resolve().is_relative_to(root) or path.stat().st_size > 16384:
        raise ValueError('Material scope must be a bounded local workspace file')
    scope = load(path)
    validate_scope(scope)
    return scope


def validate_scope(scope):
    if (not isinstance(scope, dict) or set(scope) != {'format', 'allowed_atomic_numbers',
            'required_atomic_numbers', 'max_atoms'} or scope['format'] != 'REMAG-MATERIAL-SCOPE/1'):
        raise ValueError('Invalid workspace material scope')
    allowed, required = scope['allowed_atomic_numbers'], scope['required_atomic_numbers']
    for numbers in (allowed, required):
        if (not isinstance(numbers, list) or len(numbers) > 118 or
                any(type(z) is not int or not 1 <= z <= 118 for z in numbers) or
                len(set(numbers)) != len(numbers)):
            raise ValueError('Material scope requires distinct integer atomic numbers')
    if (not allowed or not set(required) <= set(allowed) or type(scope['max_atoms']) is not int or
            not max(1, len(required)) <= scope['max_atoms'] <= 256):
        raise ValueError('Inconsistent material scope')


def check_scope(export, scope):
    """Call after exact export verification and before candidate/bundle writes."""
    if scope is None:
        return
    validate_scope(scope)
    numbers = export['geometry']['atomic_numbers']
    if (not set(numbers) <= set(scope['allowed_atomic_numbers']) or
            not set(scope['required_atomic_numbers']) <= set(numbers) or
            len(numbers) > scope['max_atoms']):
        raise ValueError('Structure is outside the active workspace material scope')
