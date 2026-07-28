"""
AST teacher — HuggingFace ASTForAudioClassification wrapped to the KitPri
interface: input (B, 3, n_mels, T), output (B, 1) logit.

transformers is imported lazily: the teacher only runs on Kaggle (training),
never on the edge target, and is an optional dependency ('kitpri[ast]').

Verified checkpoint facts (results/kitpri_v4_ast_diagnostic/best_model.pt):
  * 86.2 M params, keys begin `audio_spectrogram_transformer.*`
  * dict keys: model_state / epoch / val_f1
  * ~329 MiB — exceeds GitHub's 100 MB limit, never commit.

NOTE for the Kaggle port: the exact input adaptation the notebook used
(3-channel mel image -> AST's expected spectrogram layout) must be copied
from the notebook when training/ is populated. This wrapper feeds the mel
without ImageNet-style preprocessing, matching the verified feature pipeline;
if the ported notebook code disagrees, STOP and reconcile explicitly.
"""

import torch.nn as nn

from kitpri.models.registry import register_model


@register_model("ast")
def ast_teacher(cfg):
    try:
        from transformers import ASTConfig, ASTForAudioClassification
    except ImportError as e:
        raise ImportError(
            "Model 'ast' requires transformers. Install with: pip install 'kitpri[ast]'. "
            "(The AST teacher is only used for training on Kaggle.)"
        ) from e

    num_classes = int(cfg.get("num_classes", 1))
    pretrained_id = cfg.get("pretrained_id")
    if pretrained_id:
        model = ASTForAudioClassification.from_pretrained(
            pretrained_id, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        model = ASTForAudioClassification(ASTConfig(num_labels=num_classes))

    class _ASTWrapper(nn.Module):
        """Adapts (B, 3, n_mels, T) mel images to AST and returns raw logits (B, 1)."""

        def __init__(self, ast):
            super().__init__()
            self.ast = ast

        def forward(self, x):
            # AST expects (B, time, freq) single-channel input values.
            mono = x.mean(dim=1)               # (B, n_mels, T)
            mono = mono.transpose(1, 2)        # (B, T, n_mels)
            return self.ast(input_values=mono).logits

    return _ASTWrapper(model)
