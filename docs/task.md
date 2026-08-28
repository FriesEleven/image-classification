基于你的实验设计，我来提供完整的训练示例。这些实验涵盖了从基线到各种注意力机制的组合，非常适合你的"主-辅"式注意力研究。

## 完整的实验训练命令

### 1. 基线模型（Baseline）
```bash
python train.py --model_type mobilenetv2 --epochs 100 --batch_size 64 --experiment_name baseline_mobilenetv2
```

### 2. ECA模型（全局ECA注意力）
```bash
python train.py --model_type eca --epochs 100 --batch_size 64 --experiment_name eca_global
```

### 3. CBAM模型 - 不同插入位置

#### 浅层CBAM（1,2层）
```bash
python train.py --model_type cbam --aux_positions "1,2" --epochs 100 --batch_size 64 --experiment_name cbam_shallow_1-2
```

#### 中层CBAM（7,8层）
```bash
python train.py --model_type cbam --aux_positions "7,8" --epochs 100 --batch_size 64 --experiment_name cbam_middle_7-8
```

#### 深层CBAM（15,16层）
```bash
python train.py --model_type cbam --aux_positions "15,16" --epochs 100 --batch_size 64 --experiment_name cbam_deep_15-16
```

### 4. SE模型 - 不同插入位置

#### 浅层SE（1,2层）
```bash
python train.py --model_type se --aux_positions "1,2" --epochs 100 --batch_size 64 --experiment_name se_shallow_1-2
```

#### 中层SE（7,8层）
```bash
python train.py --model_type se --aux_positions "7,8" --epochs 100 --batch_size 64 --experiment_name se_middle_7-8
```

#### 深层SE（15,16层）
```bash
python train.py --model_type se --aux_positions "15,16" --epochs 100 --batch_size 64 --experiment_name se_deep_15-16
```

### 5. 混合模型（SE + CBAM）
```bash
python train.py --model_type hybrid --se_positions "1,2" --cbam_positions "15,16" --epochs 100 --batch_size 64 --experiment_name hybrid_se1-2_cbam15-16
```

## 批量训练脚本

为了方便执行所有实验，你可以创建一个批量训练脚本 `run_all_experiments.sh`：

```bash
#!/bin/bash

echo "Starting all experiments..."

# 1. Baseline
echo "Training Baseline..."
python train.py --model_type mobilenetv2 --epochs 100 --batch_size 64 --experiment_name baseline_mobilenetv2

# 2. ECA
echo "Training ECA Global..."
python train.py --model_type eca --epochs 100 --batch_size 64 --experiment_name eca_global

# 3. CBAM models
echo "Training CBAM Shallow (1,2)..."
python train.py --model_type cbam --aux_positions "1,2" --epochs 100 --batch_size 64 --experiment_name cbam_shallow_1-2

echo "Training CBAM Middle (7,8)..."
python train.py --model_type cbam --aux_positions "7,8" --epochs 100 --batch_size 64 --experiment_name cbam_middle_7-8

echo "Training CBAM Deep (15,16)..."
python train.py --model_type cbam --aux_positions "15,16" --epochs 100 --batch_size 64 --experiment_name cbam_deep_15-16

# 4. SE models
echo "Training SE Shallow (1,2)..."
python train.py --model_type se --aux_positions "1,2" --epochs 100 --batch_size 64 --experiment_name se_shallow_1-2

echo "Training SE Middle (7,8)..."
python train.py --model_type se --aux_positions "7,8" --epochs 100 --batch_size 64 --experiment_name se_middle_7-8

echo "Training SE Deep (15,16)..."
python train.py --model_type se --aux_positions "15,16" --epochs 100 --batch_size 64 --experiment_name se_deep_15-16

# 5. Hybrid model
echo "Training Hybrid (SE 1,2 + CBAM 15,16)..."
python train.py --model_type hybrid --se_positions "1,2" --cbam_positions "15,16" --epochs 100 --batch_size 64 --experiment_name hybrid_se1-2_cbam15-16

echo "All experiments completed!"
```

给脚本执行权限并运行：
```bash
chmod +x run_all_experiments.bat
./run_all_experiments.bat
```

## 实验设计说明

### 实验分组逻辑：

1. **基线对比**：
   - `baseline_mobilenetv2`：无注意力机制
   - `eca_global`：全局ECA注意力

2. **位置敏感性分析**：
   - CBAM在浅/中/深层的效果对比
   - SE在浅/中/深层的效果对比

3. **注意力类型对比**：
   - 相同位置下CBAM vs SE的效果

4. **主辅混合验证**：
   - `hybrid_se1-2_cbam15-16`：验证浅层SE+深层CBAM的组合效果

### 预期分析维度：

1. **准确率对比**：哪个模型/配置在CIFAR-10上表现最好
2. **效率分析**：参数量、FLOPs、推理速度的权衡
3. **位置影响**：注意力模块在不同深度的影响
4. **类型影响**：CBAM vs SE在不同场景下的优劣
5. **混合效果**：主辅式设计是否优于单一注意力

### 结果整理建议：

训练完成后，你可以使用以下命令快速查看所有实验结果：

```bash
# 查看所有实验的最佳准确率
find ./results/metrics -name "*metrics.json" -exec echo {} \; -exec grep -H '"best_acc"' {} \;

# 查看参数量对比
find ./results/metrics -name "*metrics.json" -exec echo {} \; -exec grep -H '"parameters_total"' {} \;
```

这个实验设计非常系统，能够全面验证你论文中提出的"主-辅"式注意力机制的有效性。每个实验都有明确的对比目标，结果将非常有说服力。

