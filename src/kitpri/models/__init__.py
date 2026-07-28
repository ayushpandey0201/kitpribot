"""Model package — importing this module registers all builders."""

from kitpri.models.registry import MODEL_REGISTRY, register_model, build_model
from kitpri.models import mobilenet as _mobilenet  # noqa: F401  (registers timm builders)
from kitpri.models import ast as _ast              # noqa: F401  (registers AST)

__all__ = ["MODEL_REGISTRY", "register_model", "build_model"]
