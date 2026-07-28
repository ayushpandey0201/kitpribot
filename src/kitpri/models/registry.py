"""
Model registry — THE swap point.

Adding a new architecture requires only:
  1. a builder function decorated with @register_model("name")
  2. a YAML file in configs/models/
No edits to training, evaluation, or inference code.
"""

MODEL_REGISTRY: dict = {}


def register_model(name: str):
    def deco(fn):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


def build_model(model_cfg):
    name = model_cfg.get("name") if hasattr(model_cfg, "get") else model_cfg.name
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](model_cfg)
