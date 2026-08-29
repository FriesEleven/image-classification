# MobileNetV2 Attention Experiments

本项目在 CIFAR-10/100 上比较 MobileNetV2、ECA、CBAM、SE 及混合注意力模型，重点分析注意力模块类型和插入位置对精度、参数量、计算量与推理延迟的影响。

## 项目结构

- `src/image_classification/`：模型、数据加载、训练、评估和基准测试源码。
- `configs/experiments/`：可直接运行的实验配置。
- `scripts/`：训练、批量实验、分析、可视化和诊断入口。
- `tests/`：单元测试、集成冒烟测试及小型测试素材。
- `third_party/eca_net/`：保留许可证和说明的原始 ECA-Net 实现。
- `assets/samples/gradcam/`：Grad-CAM 示例输入图片。
- `artifacts/runs/`：按实验保存的模型、日志、预测和运行配置，不提交 Git。
- `reports/`：论文需要版本化的汇总表格和最终图片。

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,visualization,dev]'
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

## 运行实验

从配置运行：

```bash
python scripts/train.py --config configs/experiments/baseline.yaml
```

CIFAR-100 baseline：

```bash
python scripts/train.py --config configs/experiments/baseline_cifar100.yaml
```

两个数据集都从官方 50,000 张训练图像中按类别分层划分 45,000 张训练集和 5,000 张验证集；官方 10,000 张测试集只在验证集选出的最佳 checkpoint 上评估一次。`dataset`、`validation_size` 和 `seed` 均记录在运行配置中。

旧命令仍然兼容：

```bash
python train.py --model_type hybrid --se_positions 1,2 --cbam_positions 15,16 \
  --epochs 200 --batch_size 64 --experiment_name hybrid_se1-2_cbam15-16
```

查看批量实验但不启动训练：

```bash
python scripts/run_experiments.py --dry-run
```

正式运行全部实验：

```bash
python scripts/run_experiments.py
```

在 Linux GPU 服务器上一行启动 CIFAR-10/100 baseline 的三个随机种子实验：

```bash
bash scripts/launch_baselines.sh
```

该命令会在后台依次运行两个数据集的 seeds 42/43/44，断开 SSH 后仍会继续。终端会打印日志路径；后台日志只记录每个 epoch 的汇总、异常和最终结果，不记录逐 batch 进度条。批次级环境、配置、状态和最终结果记录在 `artifacts/sweeps/<launch_id>/manifest.json`。再次执行时，配置一致且已有 `summary.json` 的实验会自动跳过。

## 输出约定

每次实验写入独立目录：

```text
artifacts/runs/<experiment_id>/
├── config.yaml
├── metrics.json
├── benchmark.json
├── summary.json
├── checkpoints/
├── logs/
└── predictions/
```

原始产物默认忽略。需要放入论文或长期保留的表格和图片，应由分析脚本生成到 `reports/`。

## 验证

```bash
python -m pytest
python -m compileall -q src scripts tests train.py
python scripts/diagnostics/check_data.py
```

CPU 环境可运行模型前向和配置测试；完整训练会自动下载所选 CIFAR 数据集。论文与实验规划见 `docs/`。
