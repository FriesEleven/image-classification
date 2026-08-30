import torch

from image_classification.models.attention import (
    CBAM,
    CrossStageChannelAttention,
    CrossStageGuidedCBAM,
    SEBlock,
)


def test_attention_blocks_preserve_shape():
    inputs = torch.randn(2, 16, 8, 8)
    assert CBAM(16)(inputs).shape == inputs.shape
    assert SEBlock(16)(inputs).shape == inputs.shape


def test_cross_stage_guided_cbam_preserves_shape():
    block = CrossStageGuidedCBAM(channels=16, guide_channels=8).eval()
    inputs = torch.randn(2, 16, 8, 8)
    guidance = torch.randn(2, 8)

    assert block(inputs, guidance).shape == inputs.shape


def test_cross_stage_gate_starts_as_deep_only_attention():
    attention = CrossStageChannelAttention(channels=16, guide_channels=8).eval()
    inputs = torch.randn(2, 16, 8, 8)
    first_guidance = torch.randn(2, 8)
    second_guidance = torch.randn(2, 8)

    with torch.no_grad():
        first_gate = attention(inputs, first_guidance)
        second_gate = attention(inputs, second_guidance)
        deep_logits, _raw_guidance, bounded_guidance, gated_guidance = (
            attention.attention_logits(inputs, first_guidance)
        )

    assert attention.guidance_scale.item() == 0.0
    assert torch.count_nonzero(gated_guidance) == 0
    assert bounded_guidance.abs().max().item() <= 1.0
    assert torch.allclose(first_gate, torch.sigmoid(deep_logits))
    assert torch.allclose(first_gate, second_gate)


def test_cross_stage_projection_is_bounded_before_scaling():
    attention = CrossStageChannelAttention(channels=16, guide_channels=8).eval()
    inputs = torch.randn(2, 16, 8, 8)
    guidance = torch.randn(2, 8)

    with torch.no_grad():
        for parameter in attention.guide_projection.parameters():
            parameter.mul_(100)
        attention.guidance_scale.fill_(100)
        _deep, raw_guidance, bounded_guidance, gated_guidance = (
            attention.attention_logits(inputs, guidance)
        )

    assert raw_guidance.abs().max().item() > 1.0
    assert bounded_guidance.abs().max().item() <= 1.0
    assert gated_guidance.abs().max().item() <= 1.0


def test_cross_stage_scale_then_projection_receive_gradients():
    attention = CrossStageChannelAttention(channels=16, guide_channels=8).train()
    optimizer = torch.optim.SGD(attention.parameters(), lr=0.1)
    inputs = torch.randn(2, 16, 8, 8)
    guidance = torch.randn(2, 8)

    attention(inputs, guidance).sum().backward()
    assert attention.guidance_scale.grad is not None
    assert attention.guidance_scale.grad.abs().item() > 0
    optimizer.step()
    optimizer.zero_grad()

    attention(inputs, guidance).sum().backward()
    projection_gradients = [
        parameter.grad for parameter in attention.guide_projection.parameters()
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in projection_gradients
    )
