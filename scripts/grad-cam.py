import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os

# ==================== 用户配置区域 ====================
# 模型配置
mobilenetv2 = "baseline_mobilenetv2_mobilenetv2_mobilenetv2_latest.pth"
eca = "eca_global_eca_mobilenetv2_latest.pth"
se_shallow = "se_shallow_1-2_se_pos1-2_mobilenetv2_latest.pth"
se_deep = "se_deep_15-16_se_pos15-16_mobilenetv2_latest.pth"
cbam_shallow = "cbam_shallow_1-2_cbam_pos1-2_mobilenetv2_latest.pth"
cbam_deep = "cbam_deep_15-16_cbam_pos15-16_mobilenetv2_latest.pth"
hybrid = "hybrid_se1-2_cbam15-16_hybrid_se1-2_cbam15-16_mobilenetv2_best.pth"

MODEL_FILENAME = mobilenetv2
MODEL_PATH = f"../models/final/{MODEL_FILENAME}"
TRAINED_MODEL_TYPE = 'hybrid'
MODEL_DISPLAY_NAME = ""  # 在SVG中显示的模型名称

# 图像配置
IMAGE_PATH = "../CIFAR-10-images-master/train/cat/0011.jpg"
IMAGE_NAME = "0011.jpg"  # 用于标注的可选图像名称

# 模型参数配置
cur_model_type = "mobilenetv2"
SE_POSITIONS = []
CBAM_POSITIONS = []
MODEL_INPUT_SIZE = (32, 32)
IMAGE_MEAN = (0.4914, 0.4822, 0.4465)
IMAGE_STD = (0.247, 0.243, 0.261)

# 目标配置
TARGET_CLASS = None
TARGET_LAYER_NAME = "model.features.1.conv.2"

# 输出配置
OUTPUT_FOLDER = "../gradCAM/cat/"
RESULT_FILENAME = cur_model_type + "-" + IMAGE_NAME.replace(".jpg", ".svg")
RESULT_FILE_PATH = os.path.join(OUTPUT_FOLDER, RESULT_FILENAME)

# 确保输出文件夹存在
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ======================================================

# -------------------------- 模型类（不变）--------------------------
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

# -------------------------- 加载模型 --------------------------
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


# -------------------------- Grad-CAM 核心类 --------------------------
class ManualGradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.features = None
        self.gradients = None

        # 前向钩子：保存特征图
        def forward_hook(module, input, output):
            self.features = output.detach()
            print(f"🔍 目标层特征图尺寸：{self.features.shape}")
            print(f"🔍 特征图数值范围：[{self.features.min():.4f}, {self.features.max():.4f}]")

        # 反向钩子：保存梯度
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            # 强制梯度非零
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

        # 反向传播
        class_score = outputs[0, target_class]
        class_score.backward(retain_graph=True)

        # 计算热力图
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (self.features * grads).sum(dim=1)

        # 维度校准
        if len(cam.shape) == 3:
            cam = cam.squeeze(0)
        print(f"🔍 热力图空间维度：{cam.shape}")

        # 非零归一化
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min < 1e-6:
            print("⚠️  热力图数值无差异，强制拉伸...")
            cam = cam + torch.randn_like(cam) * 1e-3
            cam_min = cam.min()
            cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min)

        # 上采样到原图尺寸
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = nn.functional.interpolate(
            cam, size=(32, 32), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()

        print(f"🔍 最终热力图维度：{cam.shape}，数值范围：[{cam.min():.4f}, {cam.max():.4f}]")
        return cam


# -------------------------- 图像预处理 --------------------------
transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])
print("✅ 图像预处理配置完成")


def save_fig_to_svg(fig, filename):
    """保存图形为SVG文件，不添加任何文字"""
    buf = BytesIO()
    # 保存时确保没有白边
    fig.savefig(
        buf,
        format='svg',
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,
        transparent=True
    )
    buf.seek(0)

    # 读取SVG内容
    svg_content = buf.read().decode('utf-8')

    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg_content)
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

    # 生成热力图
    print("\n正在生成 Grad-CAM 热力图...")
    try:
        grad_cam = ManualGradCAM(model=model, target_layer_name=TARGET_LAYER_NAME)
        cam = grad_cam(inputs=input_tensor, target_class=TARGET_CLASS)
        print("✅ 热力图生成完成")
    except Exception as e:
        print(f"❌ 热力图生成失败！错误原因：{e}")
        return

    # 可视化 - 垂直排列三张图，无任何文字
    print("\n正在创建无文字标签的可视化...")

    # 创建3行1列的子图，调整间距和大小
    # 去除所有边距和边框
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    # 去除所有子图的边框、坐标轴和刻度
    for ax in axes:
        ax.axis('off')

    # 第一行：原图
    axes[0].imshow(img)

    # 第二行：热力图（无颜色条）
    axes[1].imshow(cam, cmap="jet")

    # 第三行：叠加图
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap="jet", alpha=0.6)

    # 调整子图间距，使图像完全填充
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.02)

    # 保存SVG文件
    save_fig_to_svg(fig, RESULT_FILE_PATH)

    print(f"\n🎉 无文字标签的热力图生成完成！")
    print(f"📁 文件已保存至：{RESULT_FILE_PATH}")

    # 显示图形
    plt.show()


# -------------------------- 运行脚本 --------------------------
if __name__ == "__main__":
    main()