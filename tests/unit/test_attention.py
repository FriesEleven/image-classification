import torch

from image_classification.models.attention import CBAM, SEBlock


def test_attention_blocks_preserve_shape():
    inputs = torch.randn(2, 16, 8, 8)
    assert CBAM(16)(inputs).shape == inputs.shape
    assert SEBlock(16)(inputs).shape == inputs.shape
