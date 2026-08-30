# 验证记录

环境：RTX 3080 Ti，Python 3.12.3，PyTorch 2.8.0+cu128。所有运行都在服务器项目 `/root/autodl-tmp/image-classification` 内；没有正式 test 评估。

## 已完成的检查

- 历史五个 checkpoint 的原始完整 validation 准确率全部精确复现；各条件覆盖 5,000 张样本。
- 全项目测试：`OMP_NUM_THREADS=4 python -m pytest -q`，60 passed，另 3 subtests passed。
- `python scripts/diagnostics/check_model.py`：九个模型/数据集配置前向通过，包括 `hybrid_leaky` 和 `csgha_v4`。
- `python scripts/launch_csgha_v4.py --dry-run`：两种模型 × seeds 42/43/44，共六个互不冲突的 validation-only ID；正式训练未启动。
- `python3 -m compileall -q src scripts tests train.py` 和 `git diff --check` 通过。
- 新增诊断、启动、溯源与测试文件的 Ruff 静态检查通过；本地/服务器 68 份 Python 源码、YAML 配置和项目依赖定义的 SHA-256 完全一致。
- 从 NPZ 逐项重算全部 35 个条件的准确率和预测变化计数通过；验证置换覆盖原集合、无自身配对，训练/验证索引互不相交。
- 68 份源文件的快照复制与逐文件哈希核查通过；离线复算生成的两个诊断报告文件哈希保持不变。
- 诊断表从本地归档 JSON 复算，校验了五份配对预测 NPZ、逐 run JSON/manifest 一致性及原诊断源码副本。当前诊断源码后续仅做静态检查清理，原执行版本保存在 `source/` 中，以其记录的哈希为准。

## 两次一 epoch 冒烟训练

这是启动流程和数值有效性检查，不作为新方法性能证据。与正式 200 epochs 相比，OneCycleLR 总步数不同，不能比较其准确率来决定方法优劣。

| 组别 | Epoch 耗时 | Validation accuracy | Best checkpoint SHA-256 |
|---|---:|---:|---|
| Matched leaky control | 20.73 s | 27.08% | `10c955b84cfbb898b9f10136a0b17840e9c5e6e8c7e5423a08f27ea49ba0d99a` |
| CSGHA v4 | 21.66 s | 25.02% | `8267183046aa3c78e9fc72e5062ec4c117e75707013a6ad06931282be5fb61ee` |

Run IDs：

- `smoke_csgha_v4_diagnostic_20260830_control_hybrid_leaky_se1-2_cbam7-8_cifar10`
- `smoke_csgha_v4_diagnostic_20260830_v4_csgha_v4_se1-2_guide2_cbam7-8_cifar10`

两组均保存完整 summary、配置、provenance 和实际划分索引；核实 train=45,000、validation=5,000 且不相交，`test_evaluated=false`、无 test accuracy 字段。两份划分文件 SHA-256 相同：`25d918f697c7902c53b19a033fb1f9be2ab4b277abcc0dad7d6fd11dc73f3be4`。

旧 v3 的初始化、state_dict 键/值及前向输出已通过与 `f11d0af` 历史实现逐项相等的回归测试；新增 LeakyReLU 的负半轴梯度测试与跨架构严格加载拒绝测试通过。

尚待用户启动并取得结果：正式 v4 / matched control 六组实验；v4 的完整验证准确率、稳定性、分支活性及干预收益均未知。
