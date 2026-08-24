import os
import sys
import csv
import json
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision.models.mobilenetv2 import InvertedResidual
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from mobilenetv2_eca.ECANet.models import ECA_MobileNetV2
# 导入标准的MobileNetV2
from torchvision.models import mobilenet_v2


# ANSI 转义序列用于设置文本颜色为绿色
COLOR = '\033[97m'
RESET = '\033[0m'


# 创建必要的目录结构
def create_directories():
    """创建实验所需的目录结构"""
    dirs = [
        'models/final',
        'models/checkpoints',
        'logs/csv',
        'logs/tensorboard',
        'results/metrics',
        'results/predictions',
        'results/visualizations',
        'configs'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


# 注意力模块定义 ----------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
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

# 基础MobileNetV2模型 -----------------------------------------
class BaseMobileNetV2(nn.Module):
    """基础的MobileNetV2模型"""

    def __init__(self, num_classes=10, width_mult=1.0):
        super(BaseMobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=False, width_mult=width_mult)
        # 修改分类器以适应CIFAR-10
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

class CBAMMobileNetV2(nn.Module):
    """MobileNetV2 + CBAM注意力（可选择插入位置）"""

    def __init__(self, num_classes=10, width_mult=1.0, aux_positions=None):
        super(CBAMMobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=False, width_mult=width_mult)
        self.aux_positions = aux_positions if aux_positions is not None else []

        print(f"{COLOR}[Model Init] CBAMMobileNetV2 initializing with aux_positions: {self.aux_positions}{RESET}")

        # 在指定位置添加CBAM模块
        for idx, m in enumerate(self.model.features):
            if idx in self.aux_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.cbam = CBAM(output_channel)
                    cbam_params = sum(p.numel() for p in m.cbam.parameters())
                    print(f"{COLOR}[Model Init] ✓ Added CBAM at layer {idx}, channels: {output_channel}, params: {cbam_params:,}{RESET}")
                else:
                    print(f"{COLOR}[Model Init] ⚠️ Cannot add CBAM at layer {idx} - no output channels{RESET}")

        # 修改分类器
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def _get_output_channels(self, module):
        """获取模块的输出通道数"""
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
            if idx in self.aux_positions and hasattr(module, 'cbam'):
                x = module.cbam(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.model.classifier(x)
        return x


class SEMobileNetV2(nn.Module):
    """MobileNetV2 + SE注意力（可选择插入位置）"""

    def __init__(self, num_classes=10, width_mult=1.0, aux_positions=None):
        super(SEMobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=False, width_mult=width_mult)
        self.aux_positions = aux_positions if aux_positions is not None else []

        print(f"{COLOR}[Model Init] SEMobileNetV2 initializing with aux_positions: {self.aux_positions}{RESET}")

        # 在指定位置添加SE模块
        for idx, m in enumerate(self.model.features):
            if idx in self.aux_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.se = SEBlock(output_channel)
                    se_params = sum(p.numel() for p in m.se.parameters())
                    print(f"{COLOR}[Model Init] ✓ Added SE at layer {idx}, channels: {output_channel}, params: {se_params:,}{RESET}")
                else:
                    print(f"{COLOR}[Model Init] ⚠️ Cannot add SE at layer {idx} - no output channels{RESET}")

        # 修改分类器
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def _get_output_channels(self, module):
        """获取模块的输出通道数"""
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
            if idx in self.aux_positions and hasattr(module, 'se'):
                x = module.se(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.model.classifier(x)
        return x


class HybridAttentionMobileNetV2(nn.Module):
    """混合注意力网络：浅层SE + 深层CBAM"""

    def __init__(self, num_classes=10, width_mult=1.0,
                 se_positions=None, cbam_positions=None):
        super(HybridAttentionMobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=False, width_mult=width_mult)
        self.se_positions = se_positions if se_positions is not None else []
        self.cbam_positions = cbam_positions if cbam_positions is not None else []

        print(f"{COLOR}[Model Init] HybridAttentionMobileNetV2 initializing{RESET}")
        print(f"{COLOR}  - SE positions: {self.se_positions}{RESET}")
        print(f"{COLOR}  - CBAM positions: {self.cbam_positions}{RESET}")

        # 在浅层添加SE模块（辅注意力）
        for idx, m in enumerate(self.model.features):
            if idx in self.se_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.se = SEBlock(output_channel)
                    se_params = sum(p.numel() for p in m.se.parameters())
                    print(f"{COLOR}[Model Init] ✓ Added SE (Aux) at layer {idx}, channels: {output_channel}, params: {se_params:,}{RESET}")

        # 在深层添加CBAM模块（主注意力）
        for idx, m in enumerate(self.model.features):
            if idx in self.cbam_positions:
                output_channel = self._get_output_channels(m)
                if output_channel is not None:
                    m.cbam = CBAM(output_channel)
                    cbam_params = sum(p.numel() for p in m.cbam.parameters())
                    print(
                        f"{COLOR}[Model Init] ✓ Added CBAM (Main) at layer {idx}, channels: {output_channel}, params: {cbam_params:,}{RESET}")

        # 修改分类器
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def _get_output_channels(self, module):
        """获取模块的输出通道数"""
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
            # 应用SE注意力（浅层）
            if idx in self.se_positions and hasattr(module, 'se'):
                x = module.se(x)
            # 应用CBAM注意力（深层）
            if idx in self.cbam_positions and hasattr(module, 'cbam'):
                x = module.cbam(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.model.classifier(x)
        return x


# 参数解析 ------------------------------------------------------
def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮次')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--amp', type=lambda x: x.lower() in ('true', '1'), default=True,
                        help='是否启用混合精度训练（True/False）')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='梯度累积步数（适配显存）')
    parser.add_argument('--model_type', type=str,
                        choices=['mobilenetv2', 'eca', 'cbam', 'se', 'hybrid'],
                        default='mobilenetv2',
                        help='模型类型：mobilenetv2（基线）, eca（全局ECA）, cbam（可配置CBAM）, se（可配置SE）, hybrid（混合注意力）')
    parser.add_argument('--aux_positions', type=str, default='',
                        help='注意力模块部署的位置索引，用逗号分隔，例如"2,4,7,14"')
    parser.add_argument('--se_positions', type=str, default='',
                        help='SE模块部署的位置索引（用于混合模型），用逗号分隔，例如"1,2"')
    parser.add_argument('--cbam_positions', type=str, default='',
                        help='CBAM模块部署的位置索引（用于混合模型），用逗号分隔，例如"15,16"')
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='实验名称，用于保存结果')
    return parser.parse_args()


# 数据加载 ------------------------------------------------------
def build_dataloaders(batch_size):
    """加载CIFAR-10数据集（自动下载）"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.RandomErasing(p=0.1)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 模型指标计算函数 ------------------------------------------------
def count_attention_modules(model, module_type):
    """统计特定类型注意力模块的数量"""
    count = 0
    for module in model.modules():
        if hasattr(module, module_type):
            count += 1
    return count


def get_attention_parameters(model, module_type):
    """获取特定类型注意力模块的参数数量"""
    params = 0
    for module in model.modules():
        if hasattr(module, module_type):
            attention_module = getattr(module, module_type)
            params += sum(p.numel() for p in attention_module.parameters())
    return params


def calculate_model_metrics_detailed(model, device, model_type, aux_positions=None, se_positions=None, cbam_positions=None):
    """详细计算模型的参数量和FLOPs - 修复ECA模型统计问题"""
    print(f"{COLOR}=== Calculating Model Metrics ==={RESET}")

    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 初始化参数计数器
    eca_params = 0
    cbam_params = 0
    se_params = 0
    classifier_params = 0
    backbone_params = 0

    # 特别处理ECA模型 - 检查导入的ECA_MobileNetV2结构
    if model_type == 'eca':
        print(f"{COLOR}Debug: Processing ECA model structure{RESET}")

        # 对于导入的ECA模型，需要特殊处理ECA模块的识别
        for name, module in model.named_modules():
            # 检查是否是ECA层 - 根据导入的ECA模型结构
            if 'eca' in name.lower() and hasattr(module, 'weight'):
                eca_params += sum(p.numel() for p in module.parameters())
                print(f"{COLOR}Found ECA module: {name}, params: {sum(p.numel() for p in module.parameters())}{RESET}")

            # 统计分类器参数
            if 'classifier' in name:
                classifier_params += sum(p.numel() for p in module.parameters())

        # 如果上面的方法没找到ECA参数，尝试另一种方法
        if eca_params == 0:
            print(f"{COLOR}Debug: Alternative ECA parameter search{RESET}")
            # 遍历所有模块，寻找可能的ECA结构
            for name, module in model.named_modules():
                # 检查一维卷积（ECA的典型特征）
                if isinstance(module, nn.Conv1d) and module.kernel_size[0] == 3:  # ECA通常使用kernel_size=3的1D卷积
                    eca_params += sum(p.numel() for p in module.parameters())
                    print(f"{COLOR}Found potential ECA conv1d: {name}, params: {sum(p.numel() for p in module.parameters())}{RESET}")

        # 基础网络参数 = 总参数 - ECA参数 - 分类器参数
        backbone_params = total_params - eca_params - classifier_params

    elif model_type in ['mobilenetv2', 'cbam', 'se', 'hybrid']:
        # 对于基于torchvision的模型
        for name, module in model.named_modules():
            # 统计注意力参数
            if hasattr(module, 'eca') and module.eca is not None:
                eca_params += sum(p.numel() for p in module.eca.parameters())
            if hasattr(module, 'cbam') and module.cbam is not None:
                cbam_params += sum(p.numel() for p in module.cbam.parameters())
            if hasattr(module, 'se') and module.se is not None:
                se_params += sum(p.numel() for p in module.se.parameters())
            # 统计分类器参数
            if 'classifier' in name:
                classifier_params += sum(p.numel() for p in module.parameters())

        # 基础网络参数 = 总参数 - 所有注意力参数 - 分类器参数
        backbone_params = total_params - eca_params - cbam_params - se_params - classifier_params

    # FLOPs估算
    base_flops = 91.0 * 1e6  # MobileNetV2在CIFAR-10的基础FLOPs

    # 根据模型类型和实际添加的模块计算FLOPs调整
    attention_flops_adjustment = 0

    if model_type == 'eca':
        # 如果找到了ECA参数，估算FLOPs调整
        if eca_params > 0:
            # 假设每个ECA模块增加约0.01M FLOPs
            # 统计实际的ECA模块数量（通过参数分布估算）
            estimated_eca_count = max(1, eca_params // 64)  # 简单估算
            attention_flops_adjustment = estimated_eca_count * 0.01 * 1e6
        else:
            # 如果没找到具体参数，使用默认值（ECA模型通常在多个层添加ECA）
            attention_flops_adjustment = 16 * 0.01 * 1e6  # 假设16个ECA模块

    elif model_type == 'cbam':
        num_attention_layers = len(aux_positions) if aux_positions else 0
        attention_flops_adjustment = num_attention_layers * 0.8 * 1e6

    elif model_type == 'se':
        num_attention_layers = len(aux_positions) if aux_positions else 0
        attention_flops_adjustment = num_attention_layers * 0.2 * 1e6

    elif model_type == 'hybrid':
        num_se_layers = len(se_positions) if se_positions else 0
        num_cbam_layers = len(cbam_positions) if cbam_positions else 0
        attention_flops_adjustment = num_se_layers * 0.2 * 1e6 + num_cbam_layers * 0.8 * 1e6

    total_flops = base_flops + attention_flops_adjustment

    # 主辅注意力参数分类
    if model_type == 'hybrid':
        main_attention_params = cbam_params
        aux_attention_params = se_params
    elif model_type == 'cbam':
        main_attention_params = cbam_params
        aux_attention_params = 0
    elif model_type == 'se':
        main_attention_params = se_params
        aux_attention_params = 0
    elif model_type == 'eca':
        main_attention_params = eca_params
        aux_attention_params = 0
    else:  # mobilenetv2
        main_attention_params = 0
        aux_attention_params = 0

    # 统计模块数量
    eca_count = 0
    cbam_count = 0
    se_count = 0

    if model_type == 'eca':
        # 对于ECA模型，通过参数存在性来估算数量
        if eca_params > 0:
            # 简单估算：每个ECA模块大约有64个参数（k=3的1D卷积）
            eca_count = max(1, eca_params // 64)
        else:
            # 如果没找到参数，使用默认值
            eca_count = 16  # ECA模型通常在多个bottleneck块后添加ECA
    else:
        # 对于其他模型，正常统计
        for module in model.modules():
            if hasattr(module, 'eca') and module.eca is not None:
                eca_count += 1
            if hasattr(module, 'cbam') and module.cbam is not None:
                cbam_count += 1
            if hasattr(module, 'se') and module.se is not None:
                se_count += 1

    metrics = {
        'parameters_total': total_params,
        'parameters_trainable': trainable_params,

        # 详细分解
        'parameters_backbone': backbone_params,
        'parameters_classifier': classifier_params,
        'parameters_eca': eca_params,
        'parameters_cbam': cbam_params,
        'parameters_se': se_params,

        # 主辅注意力分类
        'parameters_main_attention': main_attention_params,
        'parameters_aux_attention': aux_attention_params,

        # 注意力模块统计
        'num_eca_modules': eca_count,
        'num_cbam_modules': cbam_count,
        'num_se_modules': se_count,

        # FLOPs信息
        'flops_total': total_flops,
        'flops_base': base_flops,
        'flops_attention_adjustment': attention_flops_adjustment,

        # 模型配置信息
        'model_type': model_type,
        'aux_positions': aux_positions,
        'se_positions': se_positions,
        'cbam_positions': cbam_positions,
    }

    # 打印详细的调试信息
    print(f"{COLOR}Total parameters: {total_params:,}{RESET}")
    print(f"{COLOR}Backbone parameters: {backbone_params:,}{RESET}")
    print(f"{COLOR}Classifier parameters: {classifier_params:,}{RESET}")

    if model_type == 'eca':
        print(f"{COLOR}ECA parameters: {eca_params:,}{RESET}")
        print(f"{COLOR}ECA modules (estimated): {eca_count}{RESET}")
    elif model_type == 'cbam':
        print(f"{COLOR}CBAM parameters: {cbam_params:,}{RESET}")
        print(f"{COLOR}CBAM modules: {cbam_count}{RESET}")
    elif model_type == 'se':
        print(f"{COLOR}SE parameters: {se_params:,}{RESET}")
        print(f"{COLOR}SE modules: {se_count}{RESET}")
    elif model_type == 'hybrid':
        print(f"{COLOR}Main attention (CBAM) parameters: {main_attention_params:,}{RESET}")
        print(f"{COLOR}Aux attention (SE) parameters: {aux_attention_params:,}{RESET}")
        print(f"{COLOR}CBAM modules: {cbam_count}{RESET}")
        print(f"{COLOR}SE modules: {se_count}{RESET}")

    print(f"{COLOR}FLOPs: {total_flops:,.0f}{RESET}")
    print(f"{COLOR}FLOPs adjustment: {attention_flops_adjustment:,.0f}{RESET}")

    return metrics


def benchmark_inference_speed(model, device, input_size=(1, 3, 32, 32), num_runs=100):
    """基准测试推理速度"""
    original_mode = model.training
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # GPU预热
    if device.type == 'cuda':
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

    # 测量推理时间
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 恢复模型原始模式
    model.train(original_mode)

    latencies = np.array(latencies)
    benchmark_results = {
        'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'batch_size': input_size[0],
        'input_resolution': f"{input_size[2]}x{input_size[3]}",
        'inference_latency_mean': float(np.mean(latencies)),
        'inference_latency_std': float(np.std(latencies)),
        'throughput_fps': 1000 / float(np.mean(latencies))  # FPS
    }

    return benchmark_results


# 保存和加载工具函数 ------------------------------------------------
def save_experiment_config(args, model, device, save_path):
    """保存实验配置"""
    config = {
        'experiment_info': {
            'name': args.experiment_name,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
            'model_type': args.model_type,
            'aux_positions': args.aux_positions,
            'se_positions': args.se_positions,
            'cbam_positions': args.cbam_positions,
        },
        'model_architecture': {
            'type': args.model_type,
            'aux_positions': args.aux_positions,
            'se_positions': args.se_positions,
            'cbam_positions': args.cbam_positions,
            'num_classes': 10,
        },
        'dataset': {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'train_samples': 50000,
            'test_samples': 10000,
        },
        'hyperparameters': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'optimizer': 'AdamW',
            'weight_decay': 1e-4,
            'scheduler': 'OneCycleLR',
            'amp': args.amp,
            'accumulation_steps': args.accumulation_steps,
        },
        'training_info': {
            'device': str(device),
            'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"{COLOR}Experiment config saved to: {save_path}{RESET}")


def save_training_log(epoch, train_metrics, val_metrics, filename):
    """保存训练日志到CSV"""
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'learning_rate', 'train_loss', 'train_acc',
                             'val_loss', 'val_acc', 'train_precision', 'train_recall',
                             'train_f1', 'val_precision', 'val_recall', 'val_f1'])

        writer.writerow([
            epoch + 1,
            train_metrics.get('learning_rate', 0),
            train_metrics['loss'],
            train_metrics['accuracy'],
            val_metrics.get('loss', 0),
            val_metrics['accuracy'],
            train_metrics['precision'],
            train_metrics['recall'],
            train_metrics['f1'],
            val_metrics['precision'],
            val_metrics['recall'],
            val_metrics['f1']
        ])


def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, train_losses, val_accs, filename):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'train_loss_list': train_losses,
        'val_acc_list': val_accs
    }
    torch.save(checkpoint, filename)


# 数据保存函数 ----------------------------------------------------
def save_confusion_matrix_data(true_labels, predictions, experiment_id, phase='val'):
    """保存混淆矩阵数据到.npz文件"""
    cm = confusion_matrix(true_labels, predictions)

    cm_data_path = f"./results/predictions/{experiment_id}_{phase}_confusion_matrix.npz"
    np.savez(cm_data_path,
             confusion_matrix=cm,
             true_labels=true_labels,
             predictions=predictions)

    print(f"{COLOR}Confusion matrix data saved to: {cm_data_path}{RESET}")
    return cm


def save_roc_data(true_labels, probabilities, experiment_id, phase='val'):
    """保存ROC曲线数据到.npz文件"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    y_true_bin = label_binarize(true_labels, classes=range(len(class_names)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    roc_data_path = f"./results/predictions/{experiment_id}_{phase}_roc_data.npz"
    np.savez(roc_data_path,
             fpr=fpr,
             tpr=tpr,
             roc_auc=roc_auc,
             true_labels=true_labels,
             probabilities=probabilities,
             class_names=class_names)

    print(f"{COLOR}ROC data saved to: {roc_data_path}{RESET}")
    print(f"{COLOR}Micro-average AUC: {roc_auc['micro']:.4f}{RESET}")

    return roc_auc


# 训练流程 ------------------------------------------------------
def train():
    # 创建目录结构
    create_directories()

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 解析注意力位置
    aux_positions = [int(x) for x in args.aux_positions.split(',')] if args.aux_positions else []
    se_positions = [int(x) for x in args.se_positions.split(',')] if args.se_positions else []
    cbam_positions = [int(x) for x in args.cbam_positions.split(',')] if args.cbam_positions else []

    print(f"{COLOR}Using auxiliary positions: {aux_positions}{RESET}")
    print(f"{COLOR}Using SE positions: {se_positions}{RESET}")
    print(f"{COLOR}Using CBAM positions: {cbam_positions}{RESET}")

    # 根据模型类型选择模型
    if args.model_type == 'mobilenetv2':
        model = BaseMobileNetV2(num_classes=10).to(device)
        print(f"{COLOR}Using BaseMobileNetV2 (no attention){RESET}")
    elif args.model_type == 'eca':
        model = ECA_MobileNetV2(num_classes=10).to(device)
        print(f"{COLOR}Using ECAMobileNetV2 (global ECA){RESET}")
    elif args.model_type == 'cbam':
        model = CBAMMobileNetV2(num_classes=10, aux_positions=aux_positions).to(device)
        print(f"{COLOR}Using CBAMMobileNetV2 (CBAM at positions {aux_positions}){RESET}")
    elif args.model_type == 'se':
        model = SEMobileNetV2(num_classes=10, aux_positions=aux_positions).to(device)
        print(f"{COLOR}Using SEMobileNetV2 (SE at positions {aux_positions}){RESET}")
    elif args.model_type == 'hybrid':
        model = HybridAttentionMobileNetV2(
            num_classes=10,
            se_positions=se_positions,
            cbam_positions=cbam_positions
        ).to(device)
        print(f"{COLOR}Using HybridAttentionMobileNetV2 (SE at {se_positions}, CBAM at {cbam_positions}){RESET}")
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    # 生成实验相关的文件名
    if args.model_type == 'hybrid':
        se_str = '-'.join(map(str, se_positions))
        cbam_str = '-'.join(map(str, cbam_positions))
        experiment_id = f"{args.experiment_name}_hybrid_se{se_str}_cbam{cbam_str}"
    elif args.model_type in ['cbam', 'se'] and aux_positions:
        pos_str = '-'.join(map(str, aux_positions))
        experiment_id = f"{args.experiment_name}_{args.model_type}_pos{pos_str}"
    else:
        experiment_id = f"{args.experiment_name}_{args.model_type}"

    model_name = f"{experiment_id}_mobilenetv2"

    # 文件路径
    config_path = f"./configs/{experiment_id}_config.yaml"
    csv_log_path = f"./logs/csv/{experiment_id}_training_log.csv"
    final_model_path = f"./models/final/{model_name}_best.pth"
    checkpoint_path = f"./models/checkpoints/{model_name}_checkpoint_epoch_{{}}.pth"
    metrics_path = f"./results/metrics/{experiment_id}_metrics.json"
    benchmark_path = f"./results/metrics/{experiment_id}_benchmark.json"
    predictions_path = f"./results/predictions/{experiment_id}_predictions.npz"

    # 保存实验配置
    save_experiment_config(args, model, device, config_path)

    # 计算模型指标
    print(f"{COLOR}Calculating model metrics...{RESET}")
    model_metrics = calculate_model_metrics_detailed(
        model, device, args.model_type,
        aux_positions, se_positions, cbam_positions
    )
    with open(metrics_path, 'w') as f:
        json.dump(model_metrics, f, indent=2)

    # 基准测试
    print(f"{COLOR}Running inference benchmark...{RESET}")
    benchmark_results = benchmark_inference_speed(model, device)
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    # 初始化优化器、损失函数等
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=args.amp)
    tensorboard_writer = SummaryWriter(f"./logs/tensorboard/{experiment_id}")
    train_loader, test_loader = build_dataloaders(args.batch_size)

    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=100
    )

    # 训练过程记录
    train_losses = []
    val_accuracies = []

    # 用于保存最佳epoch的预测结果
    best_val_predictions = None
    best_val_probabilities = None
    best_val_labels = None
    best_epoch = 0

    print(f"{COLOR}Starting experiment: {experiment_id}{RESET}")
    print(f"{COLOR}Model metrics: {model_metrics}{RESET}")
    print(f"{COLOR}Benchmark results: {benchmark_results}{RESET}")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        step_count = 0

        with tqdm(train_loader, desc=f"{COLOR}Epoch {epoch + 1}/{args.epochs} [Train]{RESET}", unit="batch",
                  bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
            for step, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast('cuda', enabled=args.amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / args.accumulation_steps

                scaler.scale(loss).backward()
                step_count += 1
                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({"loss": f"{loss.item() * args.accumulation_steps:.4f}"})

                if step_count % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

        if step_count % args.accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        train_loss = train_loss / len(train_loader) * args.accumulation_steps
        train_losses.append(train_loss)
        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        train_precision = report["macro avg"]["precision"]
        train_recall = report["macro avg"]["recall"]
        train_f1 = report["macro avg"]["f1-score"]

        train_metrics = {
            'loss': train_loss,
            'accuracy': train_acc,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'learning_rate': scheduler.get_last_lr()[0]
        }

        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        val_loss = 0.0

        with torch.no_grad(), tqdm(test_loader, desc=f"{COLOR}Epoch {epoch + 1}/{args.epochs} [Val]{RESET}",
                                   unit="batch", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss /= len(test_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_accuracies.append(val_acc)
        report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
        val_precision = report["macro avg"]["precision"]
        val_recall = report["macro avg"]["recall"]
        val_f1 = report["macro avg"]["f1-score"]

        val_metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        }

        # 保存训练日志
        save_training_log(epoch, train_metrics, val_metrics, csv_log_path)

        # TensorBoard记录
        tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
        tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)
        tensorboard_writer.add_scalar('Accuracy/train', train_acc, epoch)
        tensorboard_writer.add_scalar('Accuracy/val', val_acc, epoch)
        tensorboard_writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        # 保存检查点（每10个epoch）
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_acc, train_losses, val_accuracies,
                            checkpoint_path.format(epoch + 1))

        # 保存最佳模型和预测结果
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_val_predictions = val_preds
            best_val_probabilities = np.array(val_probs)
            best_val_labels = val_labels

            torch.save(model.state_dict(), final_model_path)
            np.savez(predictions_path,
                     true_labels=val_labels,
                     predictions=val_preds,
                     probabilities=val_probs)

        # 保存最新模型
        torch.save(model.state_dict(), f"./models/final/{model_name}_latest.pth")

        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time

        # 打印结果
        print(f"{COLOR}\nEpoch [{epoch + 1}/{args.epochs}] - Time: {epoch_time:.2f}s{RESET}")
        print(f"{COLOR}Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"P: {train_precision:.4f} | R: {train_recall:.4f} | F1: {train_f1:.4f}{RESET}")
        print(f"{COLOR}Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P: {val_precision:.4f} | "
              f"R: {val_recall:.4f} | F1: {val_f1:.4f}{RESET}")
        print(f"{COLOR}Best Acc: {best_acc:.4f} (Epoch {best_epoch}){RESET}")

    # 训练结束后保存混淆矩阵和ROC数据
    print(f"{COLOR}\nSaving evaluation data...{RESET}")

    if best_val_predictions is not None and best_val_probabilities is not None:
        # 保存混淆矩阵数据
        cm = save_confusion_matrix_data(best_val_labels, best_val_predictions, experiment_id, 'val')

        # 保存ROC数据
        roc_auc = save_roc_data(best_val_labels, best_val_probabilities, experiment_id, 'val')

        print(f"{COLOR}Best model achieved:{RESET}")
        print(f"{COLOR}  - Accuracy: {best_acc:.4f}{RESET}")
        print(f"{COLOR}  - Micro-average AUC: {roc_auc['micro']:.4f}{RESET}")

    # 保存最终训练状态
    save_checkpoint(args.epochs - 1, model, optimizer, scheduler, best_acc, train_losses, val_accuracies,
                    f"./models/checkpoints/{model_name}_final.pth")

    tensorboard_writer.close()

    print(f"{COLOR}\nTraining completed!{RESET}")
    print(f"{COLOR}Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}{RESET}")
    print(f"{COLOR}All results saved to respective directories{RESET}")


if __name__ == "__main__":
    train()