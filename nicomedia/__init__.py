import warnings

warnings.warn(
    'nicomedia is a compatibility helper layer; the canonical numerical implementations should live in tdpy and new scientific code should be added there unless there is a compelling repository-specific reason.',
    stacklevel=2,
)

from .main import *
