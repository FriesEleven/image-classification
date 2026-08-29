import torch

from image_classification.models.attention import CBAM, CrossStageGuidedCBAM, SEBlock


def test_attention_blocks_preserve_shape():
    inputs = torch.randn(2, 16, 8, 8)
    assert CBAM(16)(inputs).shape == inputs.shape
    assert SEBlock(16)(inputs).shape == inputs.shape


def test_cross_stage_guided_cbam_preserves_shape_and_uses_guidance():
    block = CrossStageGuidedCBAM(channels=16, guide_channels=8).eval()
    inputs = torch.randn(2, 16, 8, 8)
    zero_guidance = torch.zeros(2, 8)
    positive_guidance = torch.ones(2, 8)
    with torch.no_grad():
        for parameter in block.channel_attention.guide_projection.parameters():
            parameter.fill_(0.1)
        without_guidance = block(inputs, zero_guidance)
        with_guidance = block(inputs, positive_guidance)

    assert without_guidance.shape == inputs.shape
    assert with_guidance.shape == inputs.shape
    assert not torch.allclose(without_guidance, with_guidance)
