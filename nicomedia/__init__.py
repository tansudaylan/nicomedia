"""Compatibility surface for legacy astrophysical numerical helpers.

The canonical numerical implementations live in tdpy. Nicomedia retains a small
set of legacy convenience wrappers and historical function names for existing
workflows, but the package intentionally warns users that new numerical code
should be added in tdpy unless there is a strong repository-specific reason.
"""

import warnings

warnings.warn(
    'nicomedia is a compatibility helper layer; the canonical numerical implementations should live in tdpy and new scientific code should be added there unless there is a compelling repository-specific reason.',
    stacklevel=2,
)

from .main import *
from .paths import get_data_path, get_repository_path, get_visuals_path

__all__ = [
    name for name in globals()
    if not name.startswith('_') and name not in {'warnings'}
]
