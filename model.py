"""
AST model definition — matches v6 Kaggle checkpoint exactly.
"""

import torch
import torch.nn as nn
import timm


class ASTModel(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = False):
        super().__init__()

        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        embed_dim = self.backbone.num_features  # 384

        # head.0=LayerNorm(384), head.1=GELU, head.2=Linear(384,1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),         # head.0
            nn.GELU(),                       # head.1
            nn.Linear(embed_dim, num_classes),# head.2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def load_model(ckpt_path: str, device: torch.device) -> ASTModel:
    model = ASTModel(num_classes=1, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[model] Loaded checkpoint from {ckpt_path} → {device}")
    return model