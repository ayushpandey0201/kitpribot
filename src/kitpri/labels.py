"""
Label convention — defined ONCE. Do not redefine elsewhere.

Verified facts (do not flip):
  * 1 = cooking, 0 = noncooking.
  * Confirmed against metadata CSVs: every label==1 row points under
    audio_32k/cooking/ (0 exceptions in the 450-row test split), and
    test_predictions.csv rows prefixed "c_" carry true_label=1.
  * All models emit a single logit; sigmoid(logit) = P(cooking).
  * Classification rule is `probability >= threshold` (>=, not >).

The historical "bot classifies everything as Cooking" bug was NOT a label
flip (diagnosed 2026-07-28: the retired v6 model had correct direction but
poor separation). Centralizing the convention here keeps that class of bug
impossible to reintroduce regardless.
"""

LABEL_COOKING = 1
LABEL_NONCOOKING = 0

LABEL_NAMES = {LABEL_COOKING: "Cooking", LABEL_NONCOOKING: "Not Cooking"}


def label_from_probability(prob: float, threshold: float) -> int:
    """P(cooking) -> integer label. `>=` so prob exactly at threshold is cooking."""
    return LABEL_COOKING if prob >= threshold else LABEL_NONCOOKING
