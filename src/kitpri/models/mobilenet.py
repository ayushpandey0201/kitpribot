"""
timm-based CNN builders: MobileNetV2 student (the shipped v4 model) and
EfficientNet-B0 (future comparison runs).

The verified student checkpoint loads with strict=True into
timm.create_model('mobilenetv2_100', num_classes=1) — keys begin `conv_stem.*`.
"""

from kitpri.models.registry import register_model


def _timm_model(arch: str, cfg):
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            f"Model '{arch}' requires timm. Install with: pip install 'kitpri[timm]'"
        ) from e
    return timm.create_model(
        arch,
        pretrained=bool(cfg.get("pretrained", False)),
        num_classes=int(cfg.get("num_classes", 1)),
    )


@register_model("mobilenetv2_100")
def mobilenetv2_100(cfg):
    return _timm_model("mobilenetv2_100", cfg)


@register_model("efficientnet_b0")
def efficientnet_b0(cfg):
    return _timm_model("efficientnet_b0", cfg)
