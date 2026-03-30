"""Tests for spatial patch extraction."""

import torch
import pytest
from src.field.spatial import extract_patches, build_activation_map


class TestExtractPatches:
    def test_output_shape(self):
        image = torch.randn(1, 1, 28, 28)
        patches = extract_patches(image, patch_size=5, stride=1)
        assert patches.shape == (576, 25)

    def test_output_shape_stride2(self):
        image = torch.randn(1, 1, 28, 28)
        patches = extract_patches(image, patch_size=5, stride=2)
        assert patches.shape == (144, 25)

    def test_patch_values_correct(self):
        image = torch.zeros(1, 1, 28, 28)
        image[0, 0, :5, :5] = 1.0
        patches = extract_patches(image, patch_size=5, stride=1)
        assert patches[0].sum() == 25.0
        assert patches[1].sum() < 25.0

    def test_multichannel(self):
        fmap = torch.randn(1, 4, 24, 24)
        patches = extract_patches(fmap, patch_size=5, stride=1)
        assert patches.shape[0] == 20 * 20
        assert patches.shape[1] == 4 * 25

    def test_gpu_if_available(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.randn(1, 1, 28, 28, device=device)
        patches = extract_patches(image, patch_size=5, stride=1)
        assert patches.device.type == device.type


class TestBuildActivationMap:
    def test_output_shape(self):
        positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1],
                                  [0, 0], [0, 1], [1, 0], [1, 1],
                                  [2, 0], [2, 1]])
        activations = torch.ones(10)
        amap = build_activation_map(positions, activations, n_features=4, grid_h=3, grid_w=2)
        assert amap.shape == (1, 4, 3, 2)

    def test_max_pooling_per_position(self):
        positions = torch.tensor([[0, 0], [0, 0], [0, 0]])
        activations = torch.tensor([0.9, 0.5, 0.1])
        amap = build_activation_map(positions, activations, n_features=2, grid_h=1, grid_w=1)
        assert amap[0, 0, 0, 0] == 0.9
        assert amap[0, 1, 0, 0] == 0.5
