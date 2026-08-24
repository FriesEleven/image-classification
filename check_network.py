from train import ECA_MobileNetV2
from train import ECA_CBAM_MobileNetV2
from train import ECA_SE_MobileNetV2
import torch
import pandas as pd
import os
def diagnose_training_issues():
    """诊断训练过程可能的问题"""
    import glob
    import matplotlib.pyplot as plt

    print("🔍 DIAGNOSING POTENTIAL ISSUES")
    print("=" * 50)

    # 检查训练曲线
    training_logs = glob.glob("./logs/csv/*training_log.csv")

    for log_file in training_logs:
        df = pd.read_csv(log_file)
        exp_name = os.path.basename(log_file).replace('_training_log.csv', '')

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        if 'val_loss' in df.columns and df['val_loss'].sum() > 0:
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title(f'{exp_name} - Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
        plt.title(f'{exp_name} - Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'./results/visualizations/{exp_name}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 检查收敛情况
        final_train_acc = df['train_acc'].iloc[-1]
        final_val_acc = df['val_acc'].iloc[-1]
        max_val_acc = df['val_acc'].max()

        print(f"\n{exp_name}:")
        print(f"  Final Train Acc: {final_train_acc:.3f}")
        print(f"  Final Val Acc: {final_val_acc:.3f}")
        print(f"  Best Val Acc: {max_val_acc:.3f}")
        print(f"  Overfitting: {final_train_acc - final_val_acc:.3f}")

    print("\n📊 PERFORMANCE SUMMARY:")
    print("Baseline should be ~92-94% on CIFAR-10 with MobileNetV2")
    print("Current baseline: 81.14% - indicates potential training issues")


diagnose_training_issues()


def verify_model_implementation():
    """验证模型实现是否正确"""
    print("\n🔧 VERIFYING MODEL IMPLEMENTATION")
    print("=" * 50)

    # 测试基础模型
    baseline_model = ECA_MobileNetV2(num_classes=10)
    total_params = sum(p.numel() for p in baseline_model.parameters())

    print(f"Baseline Model Parameters: {total_params:,}")
    print(f"Expected: ~2,300,000 parameters")

    # 检查注意力模块是否正确集成
    test_model = ECA_CBAM_MobileNetV2(num_classes=10, aux_positions=[15, 16])
    cbam_count = 0
    for idx, module in enumerate(test_model.features):
        if hasattr(module, 'cbam'):
            cbam_count += 1
            print(f"Found CBAM at layer {idx}")

    print(f"Total CBAM modules: {cbam_count}")

    # 验证前向传播
    try:
        dummy_input = torch.randn(2, 3, 32, 32)
        output = test_model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")


verify_model_implementation()