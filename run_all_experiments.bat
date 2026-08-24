@echo off

echo "Starting all experiments..."

# 1. Baseline
echo "Training Baseline..."
python train.py --model_type mobilenetv2 --epochs 100 --batch_size 64 --experiment_name baseline_mobilenetv2

# 2. ECA
echo "Training ECA Global..."
python train.py --model_type eca --epochs 200 --batch_size 64 --experiment_name eca_global

# 3. CBAM models
echo "Training CBAM Shallow (1,2)..."
python train.py --model_type cbam --aux_positions "1,2" --epochs 200 --batch_size 64 --experiment_name cbam_shallow_1-2

echo "Training CBAM Middle (7,8)..."
python train.py --model_type cbam --aux_positions "7,8" --epochs 200 --batch_size 64 --experiment_name cbam_middle_7-8

echo "Training CBAM Deep (15,16)..."
python train.py --model_type cbam --aux_positions "15,16" --epochs 200 --batch_size 64 --experiment_name cbam_deep_15-16

# 4. SE models
echo "Training SE Shallow (1,2)..."
python train.py --model_type se --aux_positions "1,2" --epochs 200 --batch_size 64 --experiment_name se_shallow_1-2

echo "Training SE Middle (7,8)..."
python train.py --model_type se --aux_positions "7,8" --epochs 200 --batch_size 64 --experiment_name se_middle_7-8

echo "Training SE Deep (15,16)..."
python train.py --model_type se --aux_positions "15,16" --epochs 200 --batch_size 64 --experiment_name se_deep_15-16

# 5. Hybrid model
echo "Training Hybrid (SE 1,2 + CBAM 15,16)..."
python train.py --model_type hybrid --se_positions "1,2" --cbam_positions "15,16" --epochs 200 --batch_size 64 --experiment_name hybrid_se1-2_cbam15-16

echo "Training Hybrid (SE 15,16 + CBAM 1,2)..."
python train.py --model_type hybrid --se_positions "15,16" --cbam_positions "1,2" --epochs 200 --batch_size 64 --experiment_name hybrid_se15-16_cbam1-2

echo "All experiments completed!"