import os
import csv
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 从自定义模块导入模型和ECA层
from mobilenetv2_eca.ECANet.models import ECA_MobileNetV2  # 主模型类
from mobilenetv2_eca.ECANet.models import eca_layer  # ECA层定义

# ANSI 转义序列用于设置文本颜色为绿色
COLOR = '\033[97m'
RESET = '\033[0m'
REPO_ROOT = Path(__file__).resolve().parents[1]


def get_args():
    """解析命令行参数（适配AMP布尔值）"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮次')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--amp', type=lambda x: x.lower() in ('true', '1'), default=True,
                        help='是否启用混合精度训练（True/False）')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='梯度累积步数（适配显存）')
    return parser.parse_args()


def build_dataloaders(batch_size):
    """加载CIFAR-10数据集（自动下载）"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    # Windows系统需设置num_workers=0
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECA_MobileNetV2(num_classes=10).to(device)
    # 创建output目录 新增
    os.makedirs("../output", exist_ok=True)

    # 初始化CSV文件 新增
    model_name = model.__class__.__name__
    csv_filename = f"output/{model_name}_result.csv"
    with open(csv_filename, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
                             "val_acc", "val_precision", "val_recall", "val_f1"])

    # 初始化模型、优化器、损失函数
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=args.amp)  # AMP依赖GradScaler
    tensorboard_writer = SummaryWriter()  # TensorBoard日志

    # 加载数据
    train_loader, test_loader = build_dataloaders(args.batch_size)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # ================== 训练阶段 ================== 修改部分
        model.train()
        optimizer.zero_grad()

        train_loss = 0.0
        all_preds = []
        all_labels = []

        # 添加进度条，设置 bar_format 并使用绿色 ANSI 转义序列
        with tqdm(train_loader, desc=f"{COLOR}Epoch {epoch + 1}/{args.epochs} [Train]{RESET}", unit="batch",
                  bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
            for step, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)

                # 混合精度前向
                with autocast('cuda', enabled=args.amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / args.accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                # 记录训练指标
                train_loss += loss.item() * args.accumulation_steps
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        train_precision = report["macro avg"]["precision"]
        train_recall = report["macro avg"]["recall"]
        train_f1 = report["macro avg"]["f1-score"]

        # ================== 验证阶段 ================== 修改部分
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad(), tqdm(test_loader, desc=f"{COLOR}Epoch {epoch + 1}/{args.epochs} [Val]{RESET}",
                                   unit="batch", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算验证指标
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
        val_precision = report["macro avg"]["precision"]
        val_recall = report["macro avg"]["recall"]
        val_f1 = report["macro avg"]["f1-score"]

        # ================== 保存结果 ================== 新增
        # 保存指标到CSV
        model_name = model.__class__.__name__
        csv_filename = f"output/{model_name}_result.csv"
        with open(csv_filename, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{train_precision:.4f}",
                f"{train_recall:.4f}",
                f"{train_f1:.4f}",
                f"{val_acc:.4f}",
                f"{val_precision:.4f}",
                f"{val_recall:.4f}",
                f"{val_f1:.4f}"
            ])

        # 保存混淆矩阵数据（每个epoch单独保存）
        np.savez(f"output/confusion_matrix_epoch{epoch + 1}.npz",
                 true_labels=val_labels,
                 predictions=val_preds)

        # 打印结果，使用绿色 ANSI 转义序列
        print(f"{COLOR}\nEpoch [{epoch + 1}/{args.epochs}]{RESET}")
        print(f"{COLOR}Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"P: {train_precision:.4f} | R: {train_recall:.4f} | F1: {train_f1:.4f}{RESET}")
        print(f"{COLOR}Val Acc: {val_acc:.4f} | P: {val_precision:.4f} | "
              f"R: {val_recall:.4f} | F1: {val_f1:.4f}{RESET}")

        # 验证精度
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_correct += (outputs.argmax(1) == labels).sum().item()

        acc = 100 * total_correct / len(test_loader.dataset)
        print(f"{COLOR}Epoch [{epoch + 1}/{args.epochs}] | Val Acc: {acc:.2f}%{RESET}")

        # 记录TensorBoard指标
        tensorboard_writer.add_scalar('Accuracy/val', acc, epoch)

        # 保存模型... (保留原有代码)
        if acc > best_acc:
            best_acc = acc
            model_dir = REPO_ROOT / "artifacts/models"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "best_model.pth")
        model_dir = REPO_ROOT / "artifacts/models"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "last_model.pth")
    tensorboard_writer.close()


if __name__ == "__main__":
    train()
