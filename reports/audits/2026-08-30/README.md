# 2026-08-30 实验审计与论文路线复评

本次审计的结论是：**现有结果不支持将 CSGHA v3 冻结为论文的核心有效方法；shallow 独立组合是当前较强的经验参照，而不是已经验证的新机制。**

## 阅读入口

- [实验汇总](experiment_summary.md)：18 次正式实验、逐 seed 结果、配对差值、运行目录、checkpoint 校验值和已记录的代码版本。
- [方法路线评估](method_route_assessment.md)：哪些结论成立、哪些证据不足，以及接下来应如何决策。
- [机器可读结果](audit_results.json)：未舍入统计、每次运行的配置、核查问题和来源。
- [源文件索引](source_index.json)：原文件 SHA-256、大小、采集时间、排除目录及历史代码引用。

## 审计范围与边界

采集时间为北京时间 **2026-08-30 20:06:01**，来自服务器项目 `/root/autodl-tmp/image-classification`。

18 次正式运行均有连续的 200-epoch 日志、完成摘要及 best/latest/final checkpoint 文件，共核对 3,600 行 epoch 记录；18 份 best validation 和首次最佳 epoch 均与日志一致。最新六组 stability sweep 的 manifest 记录六组全部 completed、退出码均为 0，批次于 19:43:48 完成。另有 10 个 smoke/非正式目录不计入结果。

“通过”指产物之间的一致性检查通过，不代表性能已独立重现。审计没有加载权重重算准确率，也没有检验 checkpoint 内部张量与摘要是否一致。

- 12 次运行有正式 manifest 中的运行时 commit；另外 6 次 seed 42 单独运行缺失这一字段，报告明确标注，不以采集时 HEAD 补填。
- 已在本地保存配置、summary、metrics、benchmark、完整逐 epoch CSV、相关 manifest、已有诊断 JSON 及若干历史源文件原文。
- 启动日志只保存校验值、末尾片段和关键标记，**不是完整日志备份**。
- 对 18 个 best checkpoint 在服务器计算 SHA-256，总计 167,102,490 字节。**权重本体、数据集及其他 checkpoint 未下载**。服务器释放前仍需另做权重备份。
- 原始快照和小文件副本在 Git 忽略目录 `artifacts/audits/2026-08-30/`；仅推送 `reports/` 不会备份这些原始证据。
- 本次未训练、未运行官方 test 评估、未修改训练代码或服务器文件，也未改写论文 LaTeX。

## 本地离线复算

在项目根目录运行，只需 Python 标准库；无需 GPU、PyTorch 或网络：

```bash
python3 scripts/analysis/audit_experiments.py
```

它读取 `artifacts/audits/2026-08-30/snapshot.json`，重新生成三个机器生成文件：`experiment_summary.md`、`audit_results.json`、`source_index.json`。不会覆盖本页和人工撰写的方法路线评估。检测到运行产物不一致时返回非零退出码。

快照本身的 SHA-256 保存在 `source_index.json`。分享或搬迁时，需要同时携带快照；只有汇总表不能完整复算。哈希用于识别文件内容，不是运行来源的数字签名。

需要再次只读采集时，应使用新的快照和报告目录，保留本次证据：

```bash
python3 scripts/analysis/audit_experiments.py --ssh-host connect.westb.seetacloud.com --snapshot artifacts/audits/next-review/snapshot.json --output reports/audits/next-review
```

采集依赖本机现有 SSH 配置及服务器 Python/PyYAML；同名快照存在时脚本拒绝覆盖。当前脚本针对本轮 baseline / position / CSGHA 审计范围，不是任意新实验的通用汇总器；扩展实验范围时需同步更新筛选规则和比较定义。

## 验证命令

```bash
python3 -m unittest tests/unit/test_experiment_audit.py -v
python3 -m compileall -q scripts/analysis/audit_experiments.py tests/unit/test_experiment_audit.py
git diff --check
```

本次验证：10 项审计单元测试通过，语法检查与 `git diff --check` 通过；离线重建后三个生成文件的 SHA-256 均保持一致，报告链接、快照校验值及 18 次运行 / 3,600 行记录核对通过。未运行全项目训练或模型测试，本次未改动模型与训练循环。

历史 `reports/tables/experiment_results_summary.csv` 和 `.xlsx` 保持不变，但不能直接合并进本次正式比较。新实验和论文写作优先参考本目录，而非旧表中的预设结论。
