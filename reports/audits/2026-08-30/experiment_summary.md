# 正式实验审计汇总

本表从服务器原始 summary、逐 epoch 日志及 manifest 复算；百分比仅展示四舍五入值。

- 本次采集快照：`/Users/salt/Jlu/paper/classification-code/artifacts/audits/2026-08-30/snapshot.json`（本地、Git 忽略）。
- 原文副本：同目录 `source_files/`；权重未下载，只有服务器文件 SHA-256。
- `source_index.json` 为源文件校验索引；`audit_results.json` 包含未舍入统计和逐 run 核查。

## 分组结果

均值 ± 样本标准差（ddof=1）；n=1 不计算标准差。所有比较只用 validation。旧 baseline 的 test 标志字段缺失，但已保存 test 指标；单独标记其推断来源，不将字段缺失当作未评估。

| 数据集 | 变体 | Seeds | n | Best validation (%) | 已有 test (%) |
|---|---|---|---:|---:|---:|
| cifar10 | Baseline | 42,43,44 | 3 | 87.69 ± 0.16 | 87.48 ± 0.23 |
| cifar10 | CSGHA v1 | 42 | 1 | 87.50（单次） | 未评估 |
| cifar10 | CSGHA v2 | 42 | 1 | 88.20（单次） | 未评估 |
| cifar10 | CSGHA v3 | 42,43,44 | 3 | 88.01 ± 0.58 | 未评估 |
| cifar10 | Independent deep | 42 | 1 | 88.36（单次） | 未评估 |
| cifar10 | Independent middle | 42,43,44 | 3 | 87.94 ± 0.53 | 未评估 |
| cifar10 | Independent shallow | 42,43,44 | 3 | 88.27 ± 0.46 | 未评估 |
| cifar100 | Baseline | 42,43,44 | 3 | 56.94 ± 0.62 | 56.68 ± 0.24 |

## 同 seed 配对差值

单位：百分点；胜出次数不能代替显著性检验。

| 比较 | Seed 42 | Seed 43 | Seed 44 | 平均差值 | 胜出 |
|---|---:|---:|---:|---:|---:|
| CSGHA v3 - Baseline | +0.82 | -0.06 | +0.20 | +0.320 | 2/3 |
| CSGHA v3 - Independent shallow | -0.10 | -0.54 | -0.14 | -0.260 | 0/3 |
| CSGHA v3 - Independent middle | +0.22 | -0.34 | +0.34 | +0.073 | 2/3 |
| Independent shallow - Baseline | +0.92 | +0.48 | +0.34 | +0.580 | 3/3 |

## 逐次实验台账

| ID | 数据集 / 变体 | Seed | Epochs | Best val (%) | Best epoch | Test (%) | Params | 核查 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [E01](#e01) | cifar10 / Baseline | 42 | 200 | 87.86 | 190 | 87.22 | 2236682 | 通过；有溯源提示 |
| [E02](#e02) | cifar100 / Baseline | 42 | 200 | 56.84 | 177 | 56.76 | 2351972 | 通过；有溯源提示 |
| [E03](#e03) | cifar10 / Baseline | 43 | 200 | 87.68 | 192 | 87.62 | 2236682 | 通过；有溯源提示 |
| [E04](#e04) | cifar100 / Baseline | 43 | 200 | 57.60 | 188 | 56.88 | 2351972 | 通过；有溯源提示 |
| [E05](#e05) | cifar10 / Baseline | 44 | 200 | 87.54 | 193 | 87.61 | 2236682 | 通过；有溯源提示 |
| [E06](#e06) | cifar100 / Baseline | 44 | 200 | 56.38 | 167 | 56.41 | 2351972 | 通过；有溯源提示 |
| [E07](#e07) | cifar10 / CSGHA v1 | 42 | 200 | 87.50 | 178 | — | 2239080 | 通过；有溯源提示 |
| [E08](#e08) | cifar10 / CSGHA v2 | 42 | 200 | 88.20 | 193 | — | 2239178 | 通过；有溯源提示 |
| [E09](#e09) | cifar10 / CSGHA v3 | 42 | 200 | 88.68 | 180 | — | 2239178 | 通过；有溯源提示 |
| [E10](#e10) | cifar10 / CSGHA v3 | 43 | 200 | 87.62 | 200 | — | 2239178 | 通过 |
| [E11](#e11) | cifar10 / CSGHA v3 | 44 | 200 | 87.74 | 197 | — | 2239178 | 通过 |
| [E12](#e12) | cifar10 / Independent shallow | 42 | 200 | 88.78 | 170 | — | 2237080 | 通过；有溯源提示 |
| [E13](#e13) | cifar10 / Independent shallow | 43 | 200 | 88.16 | 170 | — | 2237080 | 通过 |
| [E14](#e14) | cifar10 / Independent shallow | 44 | 200 | 87.88 | 170 | — | 2237080 | 通过 |
| [E15](#e15) | cifar10 / Independent deep | 42 | 200 | 88.36 | 198 | — | 2243400 | 通过；有溯源提示 |
| [E16](#e16) | cifar10 / Independent middle | 42 | 200 | 88.46 | 170 | — | 2238024 | 通过；有溯源提示 |
| [E17](#e17) | cifar10 / Independent middle | 43 | 200 | 87.96 | 196 | — | 2238024 | 通过 |
| [E18](#e18) | cifar10 / Independent middle | 44 | 200 | 87.40 | 179 | — | 2238024 | 通过 |

## 可追溯来源

当前采集时 HEAD 不等于每次运行时 HEAD。无正式批次 commit 记录的运行标为未记录，不能靠时间戳补写。

### E01

- Run ID：`baseline_mobilenetv2_seed42_mobilenetv2_cifar10`
- Summary：`artifacts/runs/baseline_mobilenetv2_seed42_mobilenetv2_cifar10/summary.json`
- Best checkpoint SHA-256：`f5897673a4e136ded0edc2633ee618cd1581fd9560d6071fd3c5966bc9827409`
- 运行时 commit（manifest 记录）：`a99a701418abf025ffaaf37d9392d16b58aa104f`；来源 `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`；配置完全一致：True。
- 提示：`legacy_test_flag_absent_inferred_from_saved_test_metrics`。

### E02

- Run ID：`baseline_mobilenetv2_seed42_mobilenetv2_cifar100`
- Summary：`artifacts/runs/baseline_mobilenetv2_seed42_mobilenetv2_cifar100/summary.json`
- Best checkpoint SHA-256：`8f58c175b8492e1e7d1e7888cb5eab3a2f7738f7a41e33450cf575eb3c4bc2d3`
- 运行时 commit（manifest 记录）：`a99a701418abf025ffaaf37d9392d16b58aa104f`；来源 `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`；配置完全一致：True。
- 提示：`legacy_test_flag_absent_inferred_from_saved_test_metrics`。

### E03

- Run ID：`baseline_mobilenetv2_seed43_mobilenetv2_cifar10`
- Summary：`artifacts/runs/baseline_mobilenetv2_seed43_mobilenetv2_cifar10/summary.json`
- Best checkpoint SHA-256：`8ee3ff08982d0c532f7d9c327501e5612868c33d3bd3575e174f5633b6a75b08`
- 运行时 commit（manifest 记录）：`a99a701418abf025ffaaf37d9392d16b58aa104f`；来源 `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`；配置完全一致：True。
- 提示：`legacy_test_flag_absent_inferred_from_saved_test_metrics`。

### E04

- Run ID：`baseline_mobilenetv2_seed43_mobilenetv2_cifar100`
- Summary：`artifacts/runs/baseline_mobilenetv2_seed43_mobilenetv2_cifar100/summary.json`
- Best checkpoint SHA-256：`9359e38f97e6642676b9c8fff6fc38cc153abdbb48a790ca03076c5388e26761`
- 运行时 commit（manifest 记录）：`a99a701418abf025ffaaf37d9392d16b58aa104f`；来源 `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`；配置完全一致：True。
- 提示：`legacy_test_flag_absent_inferred_from_saved_test_metrics`。

### E05

- Run ID：`baseline_mobilenetv2_seed44_mobilenetv2_cifar10`
- Summary：`artifacts/runs/baseline_mobilenetv2_seed44_mobilenetv2_cifar10/summary.json`
- Best checkpoint SHA-256：`6c0b0ab4204f806679fd7b62e8af6a398a28481933c1308a1896c4e490410256`
- 运行时 commit（manifest 记录）：`a99a701418abf025ffaaf37d9392d16b58aa104f`；来源 `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`；配置完全一致：True。
- 提示：`legacy_test_flag_absent_inferred_from_saved_test_metrics`。

### E06

- Run ID：`baseline_mobilenetv2_seed44_mobilenetv2_cifar100`
- Summary：`artifacts/runs/baseline_mobilenetv2_seed44_mobilenetv2_cifar100/summary.json`
- Best checkpoint SHA-256：`77fb974f5f974f3242f95b862512e660ac457ef4d6b52669455085c56b126a7a`
- 运行时 commit（manifest 记录）：`a99a701418abf025ffaaf37d9392d16b58aa104f`；来源 `artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`；配置完全一致：True。
- 提示：`legacy_test_flag_absent_inferred_from_saved_test_metrics`。

### E07

- Run ID：`csgha_se1-2_cbam7-8_csgha_se1-2_guide2_cbam7-8_cifar10`
- Summary：`artifacts/runs/csgha_se1-2_cbam7-8_csgha_se1-2_guide2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`a2c5c7894249d0b4cdec387760b6bb2d47b4013db658fa08105eea7aaf6ac5c0`
- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。
- 提示：`run_time_git_commit_not_recorded`。

### E08

- Run ID：`csgha_v2_se1-2_cbam7-8_csgha_se1-2_guide2_cbam7-8_cifar10`
- Summary：`artifacts/runs/csgha_v2_se1-2_cbam7-8_csgha_se1-2_guide2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`8cc19728b5c2de64ef8cf338d28759711c19012ec152d99289a8b2486bcb5662`
- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。
- 提示：`run_time_git_commit_not_recorded`。

### E09

- Run ID：`csgha_v3_se1-2_cbam7-8_csgha_se1-2_guide2_cbam7-8_cifar10`
- Summary：`artifacts/runs/csgha_v3_se1-2_cbam7-8_csgha_se1-2_guide2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`60bca2fb3f63fe74c46be5d0a60ed49fdf70a849995e9027373103280e6d49b7`
- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。
- 提示：`run_time_git_commit_not_recorded`。

### E10

- Run ID：`csgha_v3_se1-2_cbam7-8_seed43_csgha_se1-2_guide2_cbam7-8_cifar10`
- Summary：`artifacts/runs/csgha_v3_se1-2_cbam7-8_seed43_csgha_se1-2_guide2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`6610571e33dadbe63d3eb6076fede84479ae38c192bebadaf53ca68d99f5cba4`
- 运行时 commit（manifest 记录）：`64b79160b6a582086443eb45533311bfc1aaf1f0`；来源 `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`；配置完全一致：True。

### E11

- Run ID：`csgha_v3_se1-2_cbam7-8_seed44_csgha_se1-2_guide2_cbam7-8_cifar10`
- Summary：`artifacts/runs/csgha_v3_se1-2_cbam7-8_seed44_csgha_se1-2_guide2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`a3fa3970fe865a20b08422390ab5968775b7a5ef9e4e581e09ab20ffda31eea7`
- 运行时 commit（manifest 记录）：`64b79160b6a582086443eb45533311bfc1aaf1f0`；来源 `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`；配置完全一致：True。

### E12

- Run ID：`position_se1-2_cbam1-2_hybrid_se1-2_cbam1-2_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam1-2_hybrid_se1-2_cbam1-2_cifar10/summary.json`
- Best checkpoint SHA-256：`c561c76ab7f78a81c988ed379ad337682e0d76aaeebb17524e4550286e2f4cb5`
- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。
- 提示：`run_time_git_commit_not_recorded`。

### E13

- Run ID：`position_se1-2_cbam1-2_seed43_hybrid_se1-2_cbam1-2_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam1-2_seed43_hybrid_se1-2_cbam1-2_cifar10/summary.json`
- Best checkpoint SHA-256：`c6b2350a123c5adec7e4ab42745915bab2a82d1285133e4234759859854708d6`
- 运行时 commit（manifest 记录）：`64b79160b6a582086443eb45533311bfc1aaf1f0`；来源 `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`；配置完全一致：True。

### E14

- Run ID：`position_se1-2_cbam1-2_seed44_hybrid_se1-2_cbam1-2_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam1-2_seed44_hybrid_se1-2_cbam1-2_cifar10/summary.json`
- Best checkpoint SHA-256：`05592b6f976f8e5ec3d0c4e3989c07743e096e581c7b67f8acfa5b925757459c`
- 运行时 commit（manifest 记录）：`64b79160b6a582086443eb45533311bfc1aaf1f0`；来源 `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`；配置完全一致：True。

### E15

- Run ID：`position_se1-2_cbam15-16_hybrid_se1-2_cbam15-16_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam15-16_hybrid_se1-2_cbam15-16_cifar10/summary.json`
- Best checkpoint SHA-256：`fb60a8ba63ad0e32f42d4dfe5a78998e9542f5acd009f6f312fdebd38fdb2447`
- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。
- 提示：`run_time_git_commit_not_recorded`。

### E16

- Run ID：`position_se1-2_cbam7-8_hybrid_se1-2_cbam7-8_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam7-8_hybrid_se1-2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`a37703cdc1fcd7ffc3816e1aa1cc0c71607955e2fee34141e58853d483370c22`
- 运行时 commit：未记录；checkpoint/配置/日志文件可追溯，但精确代码版本不可由产物单独确证。
- 提示：`run_time_git_commit_not_recorded`。

### E17

- Run ID：`position_se1-2_cbam7-8_seed43_hybrid_se1-2_cbam7-8_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam7-8_seed43_hybrid_se1-2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`fb8c89dd0d5685401925d385952e8d74d0a6fba12f86764e24fdd6a789477c77`
- 运行时 commit（manifest 记录）：`64b79160b6a582086443eb45533311bfc1aaf1f0`；来源 `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`；配置完全一致：True。

### E18

- Run ID：`position_se1-2_cbam7-8_seed44_hybrid_se1-2_cbam7-8_cifar10`
- Summary：`artifacts/runs/position_se1-2_cbam7-8_seed44_hybrid_se1-2_cbam7-8_cifar10/summary.json`
- Best checkpoint SHA-256：`e204c90b6fd271680fb82af18d8f2d465165df79081fd990ae379e171788097d`
- 运行时 commit（manifest 记录）：`64b79160b6a582086443eb45533311bfc1aaf1f0`；来源 `artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`；配置完全一致：True。
