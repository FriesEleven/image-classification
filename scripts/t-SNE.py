import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# -------------------------- 1. 核心配置（用你的绝对路径）--------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_MODEL_PATH = REPO_ROOT / "artifacts/models/final/baseline_mobilenetv2_mobilenetv2_mobilenetv2_latest.pth"
HYBRID_MODEL_PATH = REPO_ROOT / "artifacts/models/final/hybrid_se1-2_cbam15-16_hybrid_se1-2_cbam15-16_mobilenetv2_best.pth"
NUM_SAMPLES = 2000
PERPLEXITY = 30
LEARNING_RATE = 200
RANDOM_SEED = 42
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
          '#00FFFF', '#880000', '#008800', '#000088', '#888800']
SAVE_PATH = REPO_ROOT / "results/visualizations/tsne_visualization.svg"


# -------------------------- 2. 注意力模块（无修改）--------------------------
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


# -------------------------- 3. 模型定义（关键修复 forward 逻辑）--------------------------
# 基线模型（无修改）
class BaseMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(BaseMobileNetV2, self).__init__()
        from torchvision.models import mobilenet_v2
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        # 正确提取全局池化特征（和训练逻辑一致）
        for module in self.model.features:
            x = module(x)
        global_pool = x.mean([2, 3])  # 全局平均池化
        logits = self.model.classifier(global_pool)
        return global_pool, logits


# 混合注意力模型（核心修复 forward 逻辑）
class HybridAttentionMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, se_positions=None, cbam_positions=None):
        super(HybridAttentionMobileNetV2, self).__init__()
        from torchvision.models import mobilenet_v2
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.se_positions = se_positions if se_positions is not None else []
        self.cbam_positions = cbam_positions if cbam_positions is not None else []

        # 为指定位置的模块添加 SE/CBAM（和训练时一致）
        for idx, m in enumerate(self.model.features):
            if idx in self.se_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.se = SEBlock(output_channel)
            if idx in self.cbam_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.cbam = CBAM(output_channel)

        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def _get_output_channels(self, module):
        # 正确获取模块的输出通道数（适配 MobileNetV2 的 InvertedResidual 结构）
        if hasattr(module, 'conv'):
            if isinstance(module.conv, nn.Sequential):
                for layer in reversed(module.conv):
                    if hasattr(layer, 'out_channels'):
                        return layer.out_channels
            elif hasattr(module.conv, 'out_channels'):
                return module.conv.out_channels
        # 适配 MobileNetV2 的第一层卷积
        elif isinstance(module, nn.Conv2d):
            return module.out_channels
        return None

    def forward(self, x):
        # 核心修复：和训练时完全一致的 forward 逻辑
        for idx, module in enumerate(self.model.features):
            # 第一步：执行当前 features 模块
            x = module(x)
            # 第二步：如果是 SE 位置，调用 SE 层
            if idx in self.se_positions and hasattr(module, 'se'):
                x = module.se(x)
            # 第三步：如果是 CBAM 位置，调用 CBAM 层
            if idx in self.cbam_positions and hasattr(module, 'cbam'):
                x = module.cbam(x)

        # 提取全局池化特征（用于 t-SNE）
        global_pool = x.mean([2, 3])
        # 分类器输出
        logits = self.model.classifier(global_pool)
        # 返回特征和预测结果
        return global_pool, logits


# -------------------------- 4. 加载模型（无修改）--------------------------
def load_model(model_type="baseline"):
    if model_type == "baseline":
        model = BaseMobileNetV2(num_classes=10)
        model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    elif model_type == "hybrid":
        model = HybridAttentionMobileNetV2(
            num_classes=10,
            se_positions=[1, 2],
            cbam_positions=[15, 16]
        )
        model.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        raise ValueError("model_type must be 'baseline' or 'hybrid'")

    model = model.to(DEVICE).eval()
    return model


# -------------------------- 5. 加载数据集（无修改）--------------------------
def load_cifar10():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(test_set)), NUM_SAMPLES)
    subset = Subset(test_set, indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
    return dataloader, [test_set.targets[i] for i in indices]


# -------------------------- 6. 提取特征（无修改）--------------------------
def extract_features(model, dataloader):
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            global_pool, _ = model(images)
            features.append(global_pool.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features


# -------------------------- 7. t-SNE 降维（无修改）--------------------------
def tsne_reduce(features):
    """用t-SNE将高维特征降维到2D"""
    tsne = TSNE(
        n_components=2,          # 降维到2D
        perplexity=PERPLEXITY,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_SEED,
        max_iter=1000,           # 关键修改：n_iter → max_iter
        verbose=1                # 打印进度
    )
    print(f"\n开始t-SNE降维（样本数：{features.shape[0]}，特征维度：{features.shape[1]}）...")
    tsne_results = tsne.fit_transform(features)
    return tsne_results


# -------------------------- 8. 可视化（无修改）--------------------------
def visualize_tsne(baseline_tsne, hybrid_tsne, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.set_title("Baseline MobileNetV2 (No Attention)", fontsize=16, fontweight='bold')
    for cls_idx in range(10):
        mask = np.array(labels) == cls_idx
        ax1.scatter(
            baseline_tsne[mask, 0], baseline_tsne[mask, 1],
            c=COLORS[cls_idx], label=CIFAR10_CLASSES[cls_idx],
            alpha=0.7, s=20, edgecolors='none'
        )
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax1.grid(alpha=0.3)

    ax2.set_title("Hybrid Attention MobileNetV2 (SE+CBAM)", fontsize=16, fontweight='bold')
    for cls_idx in range(10):
        mask = np.array(labels) == cls_idx
        ax2.scatter(
            hybrid_tsne[mask, 0], hybrid_tsne[mask, 1],
            c=COLORS[cls_idx], label=CIFAR10_CLASSES[cls_idx],
            alpha=0.7, s=20, edgecolors='none'
        )
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', format='svg')
    print(f"\n✅ t-SNE可视化结果已保存到：{SAVE_PATH}")
    plt.show()


# -------------------------- 9. 主函数（无修改）--------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("加载CIFAR-10测试集...")
    dataloader, labels = load_cifar10()
    print(f"✅ 加载完成，采样样本数：{len(labels)}")

    print("\n加载基线模型...")
    baseline_model = load_model("baseline")
    print("加载混合注意力模型...")
    hybrid_model = load_model("hybrid")
    print("✅ 模型加载完成")

    print("\n提取基线模型特征...")
    baseline_features = extract_features(baseline_model, dataloader)
    print("提取混合注意力模型特征...")
    hybrid_features = extract_features(hybrid_model, dataloader)
    print(f"✅ 特征提取完成，特征维度：{baseline_features.shape[1]}")

    baseline_tsne = tsne_reduce(baseline_features)
    hybrid_tsne = tsne_reduce(hybrid_features)

    visualize_tsne(baseline_tsne, hybrid_tsne, labels)


if __name__ == "__main__":
    main()
