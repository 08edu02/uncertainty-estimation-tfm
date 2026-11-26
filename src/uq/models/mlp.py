"""Multi-layer perceptron for the tabular scenario (credit default).

Two properties matter here more than capacity:

* the output is always ``n_classes`` logits, exactly as in the image scenario,
  so the loss, the probability extraction and the temperature scaling are the
  same code in both scenarios;
* there is exactly **one** ``dropout`` module, right before the output layer.
  That is what turns MC Dropout into a perturbation of the representation the
  classifier reads, and it keeps the explanation honest: "we randomly switch off
  20 % of the inputs of the classifier and average".

There is no ``BatchNorm``: its per-batch statistics would mix the batch into the
prediction and contaminate the uncertainty that MC Dropout is supposed to
measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    """Shape of the network.

    Attributes
    ----------
    input_dim:
        Number of features the model receives, after encoding. The credit
        dataset has 23 published columns, but the categorical ones are encoded
        later, so the caller decides the final width.
    hidden_dims:
        Width of each hidden layer, from the input towards the output.
    dropout:
        Dropout probability, applied once before the output layer.
    n_classes:
        Number of output logits.
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    n_classes: int = 2


class MLP(nn.Module):
    """Fully connected network with a single dropout before the output."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        _check_config(config)

        layers: list[nn.Module] = []
        previous_width = config.input_dim
        for width in config.hidden_dims:
            layers.append(nn.Linear(previous_width, width))
            layers.append(nn.ReLU())
            previous_width = width
        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(previous_width, config.n_classes))

        self.config = config
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_classes)`` logits for ``(batch, input_dim)`` features."""
        return self.layers(features)


def build_mlp(config: MLPConfig) -> MLP:
    """Build the model described by ``config``."""
    return MLP(config)


def _check_config(config: MLPConfig) -> None:
    """Reject a configuration that would otherwise fail later and obscurely."""
    if config.input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {config.input_dim}")
    if not config.hidden_dims:
        raise ValueError("hidden_dims must name at least one hidden layer")
    if any(width < 1 for width in config.hidden_dims):
        raise ValueError(f"every hidden width must be positive, got {config.hidden_dims}")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}")
    if config.n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {config.n_classes}")
