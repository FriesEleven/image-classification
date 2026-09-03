# 正式论文证据包（约 1.8 GiB）内容与迁移说明

## 1. 文档目的

本文说明如何保存和迁移当前论文所需、但因体积或性质不适合提交 GitHub 的正式实验资产。
证据包用于在另一台电脑上继续完成统计复算、主表与消融表生成、精度—计算量曲线绘制，以及必要的
checkpoint 复核；不包含论文正文。

本说明基于以下冻结状态：

- 服务器项目根目录：`/root/autodl-tmp/image-classification`
- GitHub 仓库：`git@github.com:FriesEleven/image-classification.git`
- 基础提交：`2244cae03555d38fb8d4e42cdf46d066d3119886`
- 统计日期：2026-09-03
- 最终实验状态：P4 CIFAR-100 方法锁定测试完成，停止新增训练，进入论文整理

GitHub 已保存代码、实验 YAML、分析器、版本化审计、精简结果 JSON、策略锁、负结果摘要和时延报告。
本证据包主要补充 `.gitignore` 排除的 checkpoint、逐 run 日志、TensorBoard、划分索引、原始
logits/routes、不可变审计快照和测试访问标记。

## 2. 体积与数量基线

服务器实测：

| 内容 | 数量或体积 |
|---|---:|
| P0--P4 正式 early-exit run | 36 个 |
| 36 个正式 run 总占用 | 约 1.6 GiB |
| `model_best.pth` | 36 个，合计约 325 MiB |
| `model_latest.pth` | 36 个，合计约 325 MiB |
| `final.pth` | 36 个，合计约 960 MiB |
| 官方/外部分布 logits 与 routes | 12 个 NPZ，约 48 MiB |
| sweep、审计快照、策略锁、访问标记和部署档案 | 数十 MiB |
| 推荐完整证据包预留 | 约 1.8 GiB |

压缩包的最终大小取决于 `tar/gzip` 版本。模型权重通常不易进一步压缩，因此下载和本地解压时均应
至少预留 2.5 GiB；如果同时保留压缩包和解压目录，建议预留 4 GiB。

## 3. 必须包含的 36 个正式 run

所有路径均相对于项目根目录。每个 seed 都包含一组 matched baseline 和一组 multi-exit 模型。

| 阶段 | 数据集 | 训练 seed | run 数 | 路径匹配模式 | 论文用途 |
|---|---|---|---:|---|---|
| P0a | CIFAR-10 | 51--53 | 6 | `artifacts/runs/early_exit_p0_*_p0a_*` | 探索失败与共享阈值重设计来源 |
| P1b | CIFAR-10 | 54--56 | 6 | `artifacts/runs/early_exit_p1_*_p1b_*` | 独立校准、锁定策略和正式 test 主结果 |
| P2a | CIFAR-10 | 57--59 | 6 | `artifacts/runs/early_exit_p2_*_p2a_*` | 冻结阈值向未见训练 seed 的迁移 |
| P3a | CIFAR-100 | 60--65 | 12 | `artifacts/runs/early_exit_p3_*_p3a_*` | 严格零最差类别风险失败及边界诊断 |
| P4a | CIFAR-100 | 66--68 | 6 | `artifacts/runs/early_exit_p4_*_p4a_*` | 新划分独立确认和方法锁定 test 主结果 |

每个正式 run 应保留以下结构：

```text
artifacts/runs/<experiment_id>/
├── benchmark.json
├── config.yaml
├── metrics.json
├── provenance.json
├── split_indices.json
├── summary.json
├── checkpoints/
│   ├── model_best.pth
│   ├── model_latest.pth
│   └── final.pth
└── logs/
    ├── training.csv
    └── tensorboard/events.out.tfevents.*
```

三类 checkpoint 都应保留。`model_best.pth` 用于论文推理和曲线复算，`model_latest.pth` 与
`final.pth` 用于训练终态核验和审计。所有正式 run 的 `epoch_*.pth` 周期快照已在完成审计后删除，
不属于本证据包，也不应被误报为缺失。

## 4. 必须包含的辅助证据

### 4.1 Sweep manifest 与源码快照

```text
artifacts/sweeps/cifar10_early_exit_p0_serial_p0a_20260902_110000/
artifacts/sweeps/cifar10_early_exit_p1_serial_p1b_20260902_144353/
artifacts/sweeps/cifar10_early_exit_p2_transfer_serial_p2a_20260902_182501/
artifacts/sweeps/cifar100_early_exit_p3_serial_p3a_20260903_110049/
artifacts/sweeps/cifar100_early_exit_p4_confirmation_serial_p4a_20260903_150809/
```

这些目录中的 `manifest.json` 固定正式 run 集合、完成状态、PID/时间线和返回码；`source_snapshot/`
固定训练时的代码与配置，不能用当前源码替换。

### 4.2 不可变审计快照

```text
artifacts/audits/2026-09-02-early-exit-p0a/
artifacts/audits/2026-09-02-early-exit-p1b/
artifacts/audits/2026-09-03-early-exit-p2a/
artifacts/audits/2026-09-03-early-exit-p3a/
artifacts/audits/2026-09-03-early-exit-p4a/
```

对应的精简审计报告已经提交到 `reports/audits/`；这里保存的是被 Git 忽略的原始 `snapshot.json`。

### 4.3 P0 重设计分析

```text
artifacts/analyses/early_exit_p0_p0a_20260902_v1/
artifacts/analyses/early_exit_p0_p0a_class_guard_20260902_v1/
```

P0 是探索/失败定位证据，不能当作独立论文正结果，但需要保留以解释为何 P1 改为跨 seed 共享的
保守 exit8 阈值。

### 4.4 P1 策略选择与 CIFAR-10 正式测试

```text
artifacts/policy_selections/early_exit_p1b_20260902_165900/
artifacts/locked_tests/early_exit_p1b_20260902/
artifacts/test_access_registry/early_exit_p1b_8dce5d938e06ae9d.started.json
artifacts/test_access_registry/early_exit_p1b_8dce5d938e06ae9d.completed.json
```

`locked_tests` 中应有 seeds 54--56 共 3 份 `test_logits_and_routes_seed*.npz`。永久 started/completed
标记必须随包保存，用于证明并阻止重复执行正式 CIFAR-10 test。

### 4.5 P2 CIFAR-10.1 v6 外部分布测试

```text
artifacts/external_tests/early_exit_p2_cifar10_1_v6_20260903/
artifacts/external_test_access_registry/early_exit_p2_cifar10_1_v6_ef86f749fba03255.started.json
artifacts/external_test_access_registry/early_exit_p2_cifar10_1_v6_ef86f749fba03255.completed.json
```

`external_tests` 中应有 seeds 54--59 共 6 份 `cifar10_1_v6_logits_and_routes_seed*.npz`。数据集本体不必
进入证据包，Git 中的 `reports/data/2026-09-03-cifar10-1-v6/source_receipt.json` 已固定其来源与哈希。

### 4.6 P4 CIFAR-100 方法锁定测试

```text
artifacts/official_tests/early_exit_p4_cifar100_20260903/
artifacts/official_test_access_registry/early_exit_p4_cifar100_20f885a53f86e1be.started.json
artifacts/official_test_access_registry/early_exit_p4_cifar100_20f885a53f86e1be.completed.json
```

`official_tests` 中应有 seeds 66--68 共 3 份 `cifar100_test_logits_and_routes_seed*.npz`。永久访问标记不能
删除；P3 official test 从未打开，不应存在 P3 test 输出。

### 4.7 部署时延与运行日志

```text
artifacts/deployment_profiles/early_exit_p1b_rtx4090d_batch1_20260902/
artifacts/launcher_logs/
artifacts/cleanup/
```

RTX4090D 档案是 batch-1、同步 wall-clock 的分路径测量。它支持服务器 GPU 实现上的真实续算收益，
但不能写成手机时延或能耗证据。`launcher_logs/` 和 `cleanup/` 体积很小，完整保留可避免漏掉正式批次
入口日志和周期 checkpoint 清理回执。

## 5. 已由 GitHub 保存、无需重复打包的内容

在另一台电脑上先克隆仓库并检出基础提交，即可取得：

- `src/`、`scripts/`、`configs/` 和 `tests/`；
- `docs/early_exit_*.md` 与 `docs/handoff.md`；
- `reports/audits/2026-09-0*-early-exit-*`；
- `reports/experiments/2026-09-0*-early-exit-*`；
- `reports/diagnostics/2026-09-0*-early-exit-*`；
- `reports/profiles/2026-09-02-early-exit-p1b-rtx4090d/`；
- CSGHA 与静态阶段稀疏注意力的版本化负结果摘要。

恢复时必须以提交
`2244cae03555d38fb8d4e42cdf46d066d3119886` 为代码基线。之后如需修改制表或绘图脚本，应创建新的
提交，不能改写冻结实验的源码快照、策略锁或原始结果。

## 6. 明确排除的内容

推荐的约 1.8 GiB 包不包含：

- 论文目录、`.tex`、`.bib`、论文 PDF 或旧论文图片；
- `data/cifar-10-batches-py/`、`data/cifar-100-python/` 和 `data/cifar-10.1-v6/`，合计约 362 MiB，
  均可按已有来源重新取得；
- CSGHA v1--v6、旧 position screening、静态 budget-stage probe 的原始 checkpoint；它们的论文级
  负结果与哈希摘要已经提交 GitHub；
- P1a 用户中断目录、性能 smoke、scheduler/split smoke 和临时诊断 run；
- `.pytest_cache/`、`.ruff_cache/`、`__pycache__/` 和其他可再生缓存；
- SSH 私钥、密码、token、个人 SSH 配置或任何凭据；
- 已删除的 `epoch_*.pth` 周期 checkpoint。

如果目标是保存服务器上的一切而不是保存论文正式证据，则当前 `artifacts/` 约 7.2 GiB，连同数据集
约 7.6 GiB；这不属于本说明定义的 1.8 GiB 正式包。

## 7. 推荐的服务器打包方法

服务器端口可能随实例迁移改变，操作时以最新 SSH 地址为准。下面的命令只列入第 3--4 节定义的正式
资产，不包含数据集或凭据。执行前应先确认服务器没有训练或评估进程。

```bash
cd /root/autodl-tmp/image-classification

tar -czf /root/autodl-tmp/image-classification-paper-evidence-20260903.tar.gz \
  docs/paper_evidence_bundle.md \
  artifacts/runs/early_exit_p0_*_p0a_* \
  artifacts/runs/early_exit_p1_*_p1b_* \
  artifacts/runs/early_exit_p2_*_p2a_* \
  artifacts/runs/early_exit_p3_*_p3a_* \
  artifacts/runs/early_exit_p4_*_p4a_* \
  artifacts/sweeps/cifar10_early_exit_p0_serial_p0a_20260902_110000 \
  artifacts/sweeps/cifar10_early_exit_p1_serial_p1b_20260902_144353 \
  artifacts/sweeps/cifar10_early_exit_p2_transfer_serial_p2a_20260902_182501 \
  artifacts/sweeps/cifar100_early_exit_p3_serial_p3a_20260903_110049 \
  artifacts/sweeps/cifar100_early_exit_p4_confirmation_serial_p4a_20260903_150809 \
  artifacts/audits/2026-09-02-early-exit-p0a \
  artifacts/audits/2026-09-02-early-exit-p1b \
  artifacts/audits/2026-09-03-early-exit-p2a \
  artifacts/audits/2026-09-03-early-exit-p3a \
  artifacts/audits/2026-09-03-early-exit-p4a \
  artifacts/analyses/early_exit_p0_p0a_20260902_v1 \
  artifacts/analyses/early_exit_p0_p0a_class_guard_20260902_v1 \
  artifacts/policy_selections/early_exit_p1b_20260902_165900 \
  artifacts/locked_tests/early_exit_p1b_20260902 \
  artifacts/external_tests/early_exit_p2_cifar10_1_v6_20260903 \
  artifacts/official_tests/early_exit_p4_cifar100_20260903 \
  artifacts/deployment_profiles/early_exit_p1b_rtx4090d_batch1_20260902 \
  artifacts/test_access_registry/early_exit_p1b_8dce5d938e06ae9d.started.json \
  artifacts/test_access_registry/early_exit_p1b_8dce5d938e06ae9d.completed.json \
  artifacts/external_test_access_registry/early_exit_p2_cifar10_1_v6_ef86f749fba03255.started.json \
  artifacts/external_test_access_registry/early_exit_p2_cifar10_1_v6_ef86f749fba03255.completed.json \
  artifacts/official_test_access_registry/early_exit_p4_cifar100_20f885a53f86e1be.started.json \
  artifacts/official_test_access_registry/early_exit_p4_cifar100_20f885a53f86e1be.completed.json \
  artifacts/launcher_logs \
  artifacts/cleanup

cd /root/autodl-tmp
sha256sum image-classification-paper-evidence-20260903.tar.gz \
  > image-classification-paper-evidence-20260903.tar.gz.sha256
du -h image-classification-paper-evidence-20260903.tar.gz
```

创建压缩包会暂时额外占用约 1.5--1.8 GiB 服务器空间。若服务器空间不足，应直接使用 `rsync` 分目录
下载，不要删除现有正式证据来腾空间。

## 8. 在另一台电脑上下载与恢复

另一台电脑必须具有已加入服务器 `authorized_keys` 的 SSH 私钥。不要把私钥写入仓库或证据包。

```bash
rsync -avP -e "ssh -p <CURRENT_PORT> -i <PRIVATE_KEY_PATH>" \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/image-classification-paper-evidence-20260903.tar.gz \
  <LOCAL_DOWNLOAD_DIR>/

rsync -avP -e "ssh -p <CURRENT_PORT> -i <PRIVATE_KEY_PATH>" \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/image-classification-paper-evidence-20260903.tar.gz.sha256 \
  <LOCAL_DOWNLOAD_DIR>/
```

在 Linux 上校验：

```bash
cd <LOCAL_DOWNLOAD_DIR>
sha256sum -c image-classification-paper-evidence-20260903.tar.gz.sha256
```

在 macOS 上校验：

```bash
cd <LOCAL_DOWNLOAD_DIR>
shasum -a 256 -c image-classification-paper-evidence-20260903.tar.gz.sha256
```

校验成功后恢复：

```bash
git clone git@github.com:FriesEleven/image-classification.git
cd image-classification
git checkout 2244cae03555d38fb8d4e42cdf46d066d3119886
tar -xzf <LOCAL_DOWNLOAD_DIR>/image-classification-paper-evidence-20260903.tar.gz
```

解压后 `git status --ignored --short` 显示 `artifacts/` 被忽略是正常现象；不要使用 `git add -f` 将这些
大文件推入 GitHub。

## 9. 下载后验收标准

至少核对以下项目：

1. Git checkout 为 `2244cae03555d38fb8d4e42cdf46d066d3119886`；
2. P0--P4 正式 run 目录合计 36 个；
3. `model_best.pth`、`model_latest.pth`、`final.pth` 各 36 个，共 108 个 checkpoint；
4. CIFAR-10 test NPZ 3 个、CIFAR-10.1 v6 NPZ 6 个、CIFAR-100 test NPZ 3 个，共 12 个；
5. P1、P2 外部分布和 P4 的 started/completed 访问标记均存在；
6. P3 不存在 official-test 输出；
7. `reports/**/source_index.json` 引用的文件大小与 SHA-256 均能在恢复目录中匹配；
8. 不存在 SSH 私钥、token 或密码文件。

## 10. 数据使用边界

- 不得重新执行 P1 CIFAR-10、P2 CIFAR-10.1 或 P4 CIFAR-100 正式评估器；
- 不得根据保存的官方 test logits 修改阈值、seed、checkpoint 或类别保护规则；
- 官方 test logits 只用于复核既有结果、逐类统计和描述性论文绘图；
- 精度—计算量阈值曲线应优先从校准/确认划分复算，官方 test 仅标出已经锁定的工作点；
- P0 与 P3 后验诊断必须标注为探索/边界证据，不能改写成预注册正结果；
- P1 阈值固定为 `0.984`，P4 阈值固定为 `0.903`；
- MAC 仅为 Conv/Linear 操作代理；RTX4090D 时延不能表述为手机端时延或能耗。

