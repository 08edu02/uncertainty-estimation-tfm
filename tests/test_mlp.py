"""Tests for the tabular network: shapes, dropout placement and parameter count."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from uq.models.mlp import MLPConfig, build_mlp


def make_model(input_dim: int = 23) -> nn.Module:
    return build_mlp(MLPConfig(input_dim=input_dim))


def test_output_shape_is_one_logit_per_class() -> None:
    model = make_model(input_dim=23)
    features = torch.zeros((7, 23))

    assert model(features).shape == (7, 2)


def test_parameter_count_matches_the_hand_computed_value() -> None:
    # 23*128 + 128 = 3072, 128*64 + 64 = 8256, 64*2 + 2 = 130.
    model = make_model(input_dim=23)

    assert sum(parameter.numel() for parameter in model.parameters()) == 3072 + 8256 + 130


def test_there_is_exactly_one_dropout_module() -> None:
    model = make_model()

    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]

    assert len(dropouts) == 1


def test_the_dropout_is_immediately_before_the_output_layer() -> None:
    model = make_model()
    layers = list(model.layers)

    assert isinstance(layers[-2], nn.Dropout)
    assert isinstance(layers[-1], nn.Linear)
    assert layers[-2].p == model.config.dropout


def test_eval_mode_is_deterministic() -> None:
    model = make_model().eval()
    features = torch.ones((4, 23))

    first = model(features)
    second = model(features)

    assert torch.equal(first, second)


def test_training_mode_makes_dropout_active() -> None:
    model = make_model().train()
    features = torch.ones((64, 23))

    first = model(features)
    second = model(features)

    assert not torch.equal(first, second)


def test_zero_dropout_behaves_the_same_in_both_modes() -> None:
    model = build_mlp(MLPConfig(input_dim=5, dropout=0.0))
    features = torch.randn((8, 5))

    model.eval()
    without_training = model(features)
    model.train()
    with_training = model(features)

    assert torch.equal(without_training, with_training)


def test_backward_reaches_every_parameter() -> None:
    model = make_model(input_dim=4)
    features = torch.randn((6, 4))
    logits = model(features)

    logits.sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_the_configuration_is_kept_for_rebuilding_the_model() -> None:
    config = MLPConfig(input_dim=10, hidden_dims=(32,), dropout=0.1)

    model = build_mlp(config)

    assert model.config == config


def test_an_empty_hidden_stack_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one hidden layer"):
        build_mlp(MLPConfig(input_dim=5, hidden_dims=()))


def test_a_non_positive_input_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="input_dim must be positive"):
        build_mlp(MLPConfig(input_dim=0))


def test_a_hidden_width_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="hidden width must be positive"):
        build_mlp(MLPConfig(input_dim=5, hidden_dims=(16, 0)))


def test_a_dropout_of_one_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        build_mlp(MLPConfig(input_dim=5, dropout=1.0))


def test_a_single_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_classes must be at least 2"):
        build_mlp(MLPConfig(input_dim=5, n_classes=1))
