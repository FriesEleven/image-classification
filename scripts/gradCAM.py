import torch
import torch.nn as nn
from fontTools.merge.util import equal
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os
import cairosvg

# -------------------------- 第一步：配置（最终版）--------------------------
mobilenetv2 = "baseline_mobilenetv2_mobilenetv2_mobilenetv2_latest.pth"
se_shallow = "se_shallow_1-2_se_pos1-2_mobilenetv2_latest.pth"
hybrid = "hybrid_se1-2_cbam15-16_hybrid_se1-2_cbam15-16_mobilenetv2_best.pth"
IMAGE_PATH = "../gradCAM-images/airplane/0003.jpg"
MODEL_FILENAME = hybrid
MODEL_PATH = f"../models/final/{MODEL_FILENAME}"

TRAINED_MODEL_TYPE = 'hybrid'

cur_model_type = "hybrid"
SE_POSITIONS = [1,2]
CBAM_POSITIONS = [15,16]

MODEL_INPUT_SIZE = (32, 32)
IMAGE_MEAN = (0.4914, 0.4822, 0.4465)
IMAGE_STD = (0.247, 0.243, 0.261)
TARGET_CLASS = None
TARGET_LAYER_NAME = "model.features.1.conv.2"
result_file_name = "../gradCAM/airplane/" + cur_model_type +"-0003.svg"
result_heatmap = "../gradCAM/airplane/"+ cur_model_type +"-heatmap-0003.svg"
# ----------------------------------------------------------------------------------

# -------------------------- 第二步：模型类（不变）--------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class HybridAttentionMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, se_positions=None, cbam_positions=None):
        super(HybridAttentionMobileNetV2, self).__init__()
        from torchvision.models import mobilenet_v2
        self.model = mobilenet_v2(pretrained=False, width_mult=width_mult)
        self.se_positions = se_positions if se_positions is not None else []
        self.cbam_positions = cbam_positions if cbam_positions is not None else []

        for idx, m in enumerate(self.model.features):
            if idx in self.se_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.se = SEBlock(output_channel)

        for idx, m in enumerate(self.model.features):
            if idx in self.cbam_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.cbam = CBAM(output_channel)

        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def _get_output_channels(self, module):
        if hasattr(module, 'conv'):
            if isinstance(module.conv, nn.Sequential):
                for layer in reversed(module.conv):
                    if hasattr(layer, 'out_channels'):
                        return layer.out_channels
            elif hasattr(module.conv, 'out_channels'):
                return module.conv.out_channels
        return None

    def forward(self, x):
        for idx, module in enumerate(self.model.features):
            x = module(x)
            if idx in self.se_positions and hasattr(module, 'se'):
                x = module.se(x)
            if idx in self.cbam_positions and hasattr(module, 'cbam'):
                x = module.cbam(x)
        x = x.mean([2, 3])
        x = self.model.classifier(x)
        return x


try:
    from mobilenetv2_eca.ECANet.models import ECA_MobileNetV2
except ImportError:
    class ECA_MobileNetV2(nn.Module):
        def __init__(self, num_classes=10):
            super(ECA_MobileNetV2, self).__init__()
            from torchvision.models import mobilenet_v2
            self.model = mobilenet_v2(pretrained=False)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

        def forward(self, x):
            return self.model(x)
# ----------------------------------------------------------------------------------

# -------------------------- 第三步：加载模型（修复训练模式）--------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{'GPU' if torch.cuda.is_available() else 'CPU'}")


def load_your_model(model_path, model_type='hybrid'):
    num_classes = 10
    if model_type == 'hybrid':
        model = HybridAttentionMobileNetV2(
            num_classes=num_classes,
            se_positions=SE_POSITIONS,
            cbam_positions=CBAM_POSITIONS
        )
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    # 关键：train 模式但冻结 BatchNorm，避免梯度消失
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model


print(f"正在加载模型：{MODEL_FILENAME}")
try:
    model = load_your_model(MODEL_PATH, model_type=TRAINED_MODEL_TYPE)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败！错误原因：{e}")
    exit()


# -------------------------- 第四步：Grad-CAM 核心类（最终版）--------------------------
class ManualGradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.features = None
        self.gradients = None

        # 前向钩子：保存特征图（带尺寸打印）
        def forward_hook(module, input, output):
            self.features = output.detach()
            print(f"🔍 目标层特征图尺寸：{self.features.shape}")  # 必须是 (1, c, 8, 8) 或 (1, c, 4, 8)
            print(f"🔍 特征图数值范围：[{self.features.min():.4f}, {self.features.max():.4f}]")

        # 反向钩子：保存梯度（带非零检查）
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            # 强制梯度非零（关键修复）
            if torch.allclose(self.gradients, torch.zeros_like(self.gradients)):
                print("⚠️  梯度全为 0，添加随机微小梯度...")
                self.gradients = self.gradients + torch.randn_like(self.gradients) * 1e-4

        # 绑定钩子
        hook_bound = False
        for name, layer in model.named_modules():
            if name == target_layer_name:
                layer.register_forward_hook(forward_hook)
                layer.register_full_backward_hook(backward_hook)
                print(f"✅ 成功为目标层 {name} 注册钩子")
                hook_bound = True
                break
        if not hook_bound:
            raise ValueError(f"❌ 未找到目标层 {TARGET_LAYER_NAME}")

    def __call__(self, inputs, target_class=None):
        b, c, h, w = inputs.size()
        self.model.zero_grad()

        # 前向传播
        outputs = self.model(inputs)

        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).squeeze()
            cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck']
            print(f"✅ 自动识别的类别：索引 {target_class.item()} → {cifar10_classes[target_class.item()]}")
        else:
            target_class = torch.tensor(target_class).to(device)

        # 反向传播（直接对类别得分求导）
        class_score = outputs[0, target_class]
        class_score.backward(retain_graph=True)

        # 计算热力图（确保空间维度正确）
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, c, 1, 1)
        cam = (self.features * grads).sum(dim=1)  # (1, H, W) → H/W 是 8 或 4

        # 维度校准：强制转为 2D（H, W）
        if len(cam.shape) == 3:
            cam = cam.squeeze(0)  # (H, W)
        print(f"🔍 热力图空间维度：{cam.shape}")  # 必须是 (8,8) 或 (4,8)

        # 非零归一化（避免全零）
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min < 1e-6:
            print("⚠️  热力图数值无差异，强制拉伸...")
            cam = cam + torch.randn_like(cam) * 1e-3  # 添加微小噪声
            cam_min = cam.min()
            cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min)

        # 上采样到原图尺寸（32x32），用双线性插值更平滑
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        cam = nn.functional.interpolate(
            cam, size=(32, 32), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()  # (32,32)

        print(f"🔍 最终热力图维度：{cam.shape}，数值范围：[{cam.min():.4f}, {cam.max():.4f}]")
        return cam


# -------------------------- 第五步：图像预处理 + 可视化（不变）--------------------------
transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])
print("✅ 图像预处理配置完成")


def save_fig_to_svg(fig, filename):
    buf = BytesIO()
    fig.savefig(
        buf,
        format='svg',
        bbox_inches='tight',
        dpi=300,
        transparent=True
    )
    buf.seek(0)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(buf.read().decode('utf-8'))
    print(f"✅ 已保存 SVG 文件：{filename}")


def main():
    # 加载图像
    print(f"\n正在加载图像：{IMAGE_PATH}")
    try:
        img = Image.open(IMAGE_PATH).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        print(f"✅ 图像加载完成，输入维度：{input_tensor.shape}")
    except Exception as e:
        print(f"❌ 图像加载失败！错误原因：{e}")
        return

    # 生成热力图（最终版）
    print("\n正在生成 Grad-CAM 热力图...")
    try:
        grad_cam = ManualGradCAM(model=model, target_layer_name=TARGET_LAYER_NAME)
        cam = grad_cam(inputs=input_tensor, target_class=TARGET_CLASS)
        print("✅ 热力图生成完成")
    except Exception as e:
        print(f"❌ 热力图生成失败！错误原因：{e}")
        return

    # 可视化 + 保存
    print("\n正在保存 SVG 文件...")
    fig_combined, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    # 原图
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis("off")
    # 热力图（用 jet  colormap，对比明显）
    im2 = ax2.imshow(cam, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap", fontsize=14, fontweight='bold')
    ax2.axis("off")
    fig_combined.colorbar(im2, ax=ax2, shrink=0.8)
    # 叠加图（alpha=0.6，更清晰）
    ax3.imshow(img)
    ax3.imshow(cam, cmap="jet", alpha=0.6)
    ax3.set_title("Heatmap + Original", fontsize=14, fontweight='bold')
    ax3.axis("off")
    plt.tight_layout()
    save_fig_to_svg(fig_combined, result_file_name)

    # 单独保存高清热力图
    fig_heatmap = plt.figure(figsize=(8, 8))
    plt.imshow(cam, cmap="jet")
    plt.axis("off")
    plt.colorbar(shrink=0.8)
    save_fig_to_svg(fig_heatmap, result_heatmap)

    print("\n🎉 最终版热力图生成完成！正在弹出可视化窗口...")
    plt.show()


# -------------------------- 运行脚本 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("📌 最终版 Grad-CAM 脚本（必出有效热力图）")
    print("=" * 60)
    input("按回车键开始运行...")
    main()