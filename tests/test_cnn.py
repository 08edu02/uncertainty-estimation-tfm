"""Tests for the image network: shapes, dropout placement and parameter count."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from uq.models.cnn import CNNConfig, build_cnn


def make_model(**overrides: object) -> nn.Module:
    return build_cnn(CNNConfig(**overrides))  # type: ignore[arg-type]


def test_output_shape_is_one_logit_per_class() -> None:
    model = make_model()
    images = torch.zeros((5, 1, 28, 28))

    assert model(images).shape == (5, 2)


def test_parameter_count_matches_the_hand_computed_value() -> None:
    # conv1: 1*16*3*3 + 16 = 160;  conv2: 16*32*3*3 + 32 = 4640;
    # dense: 32*7*7 = 1568 inputs, 1568*64 + 64 = 100416;  output: 64*2 + 2 = 130.
    model = make_model()

    expected = 160 + 4640 + 100416 + 130
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_the_spatial_size_shrinks_by_four_over_two_poolings() -> None:
    model = make_model()
    images = torch.zeros((2, 1, 28, 28))

    flattened = model.features[:7](images)

    assert flattened.shape == (2, 32 * 7 * 7)


def test_there_is_exactly_one_dropout_module() -> None:
    model = make_model()

    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]

    assert len(dropouts) == 1
    assert model.dropout is dropouts[0]


def test_the_dropout_sits_between_the_features_and_the_output() -> None:
    model = make_model()

    assert isinstance(model.features, nn.Sequential)
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.output, nn.Linear)
    assert model.dropout.p == model.config.dropout


def test_eval_mode_is_deterministic() -> None:
    model = make_model().eval()
    images = torch.ones((4, 1, 28, 28))

    first = model(images)
    second = model(images)

    assert torch.equal(first, second)


def test_training_mode_makes_dropout_active() -> None:
    model = make_model().train()
    images = torch.ones((64, 1, 28, 28))

    first = model(images)
    second = model(images)

    assert not torch.equal(first, second)


def test_zero_dropout_behaves_the_same_in_both_modes() -> None:
    model = make_model(dropout=0.0)
    images = torch.randn((8, 1, 28, 28))

    model.eval()
    without_training = model(images)
    model.train()
    with_training = model(images)

    assert torch.equal(without_training, with_training)


def test_backward_reaches_every_parameter() -> None:
    model = make_model()
    images = torch.randn((3, 1, 28, 28))
    logits = model(images)

    logits.sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_the_configuration_is_kept_for_rebuilding_the_model() -> None:
    config = CNNConfig(hidden_dim=32, dropout=0.1)

    model = build_cnn(config)

    assert model.config == config


def test_the_default_configuration_matches_the_project_decision() -> None:
    model = build_cnn()

    assert model.config == CNNConfig(
        channels=(16, 32), hidden_dim=64, dropout=0.2, n_classes=2, image_size=28
    )


def test_an_image_size_that_is_not_a_multiple_of_four_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        build_cnn(CNNConfig(image_size=30))


def test_more_than_two_convolutions_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly two convolutions"):
        build_cnn(CNNConfig(channels=(8, 16, 32)))  # type: ignore[arg-type]


def test_a_dropout_of_one_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        build_cnn(CNNConfig(dropout=1.0))
