"""Small convolutional network for the 28x28 medical images.

Trained from scratch: with 4 185 development images, a pretrained ResNet-18
would be disproportionate and would add decisions that then have to be defended
(which layers to freeze, on which split to fine-tune). Two convolutions and one
dense layer are enough to separate pneumonia from normal, and the goal of this
work is not accuracy but comparing how well calibrated the confidence is.

Like the MLP, it returns ``n_classes`` logits and has **one** ``dropout``
module, right before the output layer, so that MC Dropout means the same thing
in both scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    """Shape of the network.

    Attributes
    ----------
    channels:
        Number of filters of each convolution. Each one is followed by a 2x2
        max-pool, so the spatial size is divided by 4 in total.
    hidden_dim:
        Width of the dense layer before the output.
    dropout:
        Dropout probability, applied once before the output layer.
    n_classes:
        Number of output logits.
    image_size:
        Side of the square input. MedMNIST+ ships 28 for this project.
    """

    channels: tuple[int, int] = (16, 32)
    hidden_dim: int = 64
    dropout: float = 0.2
    n_classes: int = 2
    image_size: int = 28


class SmallCNN(nn.Module):
    """Two convolutions, one hidden dense layer and a single dropout."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        _check_config(config)

        first_channels, second_channels = config.channels
        reduced_size = config.image_size // 4
        flattened_size = second_channels * reduced_size * reduced_size

        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(flattened_size, config.hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, config.n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_classes)`` logits for ``(batch, 1, size, size)`` images."""
        hidden = self.dropout(self.features(images))
        return self.output(hidden)


def build_cnn(config: CNNConfig | None = None) -> SmallCNN:
    """Build the model described by ``config``, or the default 28x28 one."""
    return SmallCNN(config or CNNConfig())


def _check_config(config: CNNConfig) -> None:
    """Reject a configuration that would otherwise fail later and obscurely."""
    if len(config.channels) != 2:
        raise ValueError(f"channels must name exactly two convolutions, got {config.channels}")
    if any(count < 1 for count in config.channels):
        raise ValueError(f"every channel count must be positive, got {config.channels}")
    if config.hidden_dim < 1:
        raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}")
    if config.n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {config.n_classes}")
    if config.image_size < 8 or config.image_size % 4:
        raise ValueError(
            f"image_size must be a multiple of 4 and at least 8, got {config.image_size}"
        )
