"""Model architectures.

Two scenarios, two models, one contract: a batch of tensors in,
``(batch, n_classes)`` **logits** out. No ``softmax`` here: turning logits into
probabilities is a decision of the uncertainty method, not of the architecture,
and doing it inside the model would make temperature scaling impossible on the
original logits.

The models do not train, do not compute metrics and do not know the dataset.
They receive widths and return an ``nn.Module``.
"""

from __future__ import annotations

from uq.models.cnn import CNNConfig, SmallCNN, build_cnn
from uq.models.mlp import MLP, MLPConfig, build_mlp

__all__ = [
    "MLP",
    "CNNConfig",
    "MLPConfig",
    "SmallCNN",
    "build_cnn",
    "build_mlp",
]
