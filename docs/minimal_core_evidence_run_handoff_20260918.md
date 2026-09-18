# 2026-09-18 最小核心证据：运行交接

## 实验范围与授权

执行依据为 `docs/minimal_core_evidence_experiment_guide_20260918.md`。本轮仅补充 M1 共享阈值直接对照和 M2 六个 A3 的同模型精度—MAC—GPU singleton 时延工作点，最后 M3 统计与审计。没有新增训练、ImageNet-100、第二 backbone、CPU 或大 batch 重测。

用户在当前任务中明确批准了：CIFAR-10 A3 seeds54/55/56、阈值0.984；CIFAR-100 A3 seeds66/67/68、阈值0.903；对这六个固定checkpoint进行一次锁定官方评估，并以同一测试范围的图像测量RTX3080Ti/batch1时延。授权不包含测试阈值搜索或按结果换模型。

服务器根目录 `/root/autodl-tmp/image-classification`；Python `/root/miniconda3/bin/python`；SSH端口50799。

## 新输出与一次启动

唯一输出：`artifacts/analyses/minimal_core_evidence_20260918_r1/`。

M0已执行，保存冻结协议、输入/环境清单、拟合/审计ID和SHA。M1由准备者先执行并审查；启动入口验证已完成M1产物的哈希，复用这些结果继续M2和M3，不重复选阈值。这是为遵守指南的“M1后先审查再投入官方评估”顺序。

在服务器项目目录启动：

```bash
/root/miniconda3/bin/python scripts/launch_minimal_core_evidence.py --authorize-a3-cifar-official-test
```

入口后台执行、GPU任务串行，输出PID；日志为 `artifacts/launcher_logs/minimal_core_evidence_20260918_r1.log`。进度记录在 `batch_status.json`。不要同时运行其他训练/基准；不要重复启动。

完成后精简结果自动整理到 `reports/experiments/2026-09-18-minimal-core-evidence/`，原始预测和计时数组留在本轮artifacts。不要因精简报告已生成而删除原始证据。

## 图像划分的协议修订

盘点发现CIFAR-100 confirmation的5000图像与source不同，不能简单按数组位置或独立shuffle划分。使用以下预先冻结的规则：

1. 按dataset分别使用PCG64 seed20260918；source池每类按ID排序，再随机分一半F、一半E。
2. 同一图像ID出现在其他池时，继承source的F/E归属；新出现的confirmation图像按ID排序后使用同一RNG随机填满每类50/50配额。
3. 15个模型均按ID对应到划分，每队列F/E各2500张，逐类均衡。验证source-F与target-E交集为空；同ID标签冲突立即停止。

修订及实际ID/哈希已写入协议与`fit_audit_partitions.json`。这些图像在历史研究中已经被看过，因此只能称本轮拟合/审计分离；不能称全新未见图像验证。

## 计时与正确性范围

- M2精度：每模型完整10000张官方test，以实际部署batch1执行final、exit8与dynamic；按自身final参照统计policy精度、类别风险与校正MAC。
- 正确性：全10000张检查dynamic路由、argmax和reference logits；任一路由/argmax错误或logit容差错误阻止该运行继续计时。rtol1e-4、atol1e-5。
- 计时：每dataset用固定PCG64 permutation的前1000个official ID；跨模型、跨轮次保留相同子集/顺序；五个配对轮次，每模式100warmup、1000timed calls，两个模式顺序随机化。
- 计时范围是驻留输入、严格GPU同步的模型内墙钟；不含预处理、输入传输、服务队列、计时外路径回传。计时子集早退率/MAC与全test早退率/MAC分列。
- p95是每seed五轮p95的平均，不是合并所有调用的总体p95；模型层统计n=3。历史CPU和大batch负结果继续保留。

## 一次性访问与停止

`artifacts/minimal_core_official_access_20260918_r1.json`和本轮`official_started.json`在首次实例化官方测试图像前创建。存在访问/启动标记时拒绝再次评估。技术失败保留部分输出，不自动重试；修复需新协议版本和输出目录，保留本次记录。

科学负结果不触发自动删seed、改预算或重选阈值。官方benchmark此前被其他项目方法接触过，本次只称六个固定A3模型的补充锁定评估。P3停止、P6停止、P7失败不因本批完成而改变。

## 实现验证与后续交付

新源码：`scripts/analysis/minimal_core_evidence.py`、`scripts/launch_minimal_core_evidence.py`。
测试：`tests/unit/test_minimal_core_evidence.py`。

已验证ID错位/标签冲突、不同池重叠图像的划分隔离、成本手算、移除类别约束时的失败披露、固定模型图像配对bootstrap。六个真实A3 checkpoint已经通过合成输入动态路径冒烟检查，没有读取官方测试图像。

完成后检查M1逐seed72行、M2六模型、计时60行、全部原始数组哈希和正确性，再采用共享对照表、A3完整工作点表及离散取舍图更新论文。完整投稿前指南仍是上游要求；完成不意味着所有科学门槛通过。

## 已完成 M1 的审查结果

全部四策略、三队列、source-F/target-E共72条记录完整；三个source-F/target-E交集均为0。以下是target-E三seed均值；最差类别列为三seed中的最大值。

| 队列 | 策略 | Policy accuracy (%) | 校正MAC节省 (%) | 最差类别下降最大值 (pp) | 全seed风险通过 | 全seed节省下限通过 |
|---|---|---:|---:|---:|---|---|
| CIFAR-10 | S | 86.6800 | 36.2657 | 0 | 是 | 是 |
| CIFAR-10 | I | 86.7067 | 46.4983 | 2.4 | 否 | 是 |
| CIFAR-10 | A | 86.6800 | 36.2657 | 0 | 是 | 是 |
| CIFAR-10 | C | 86.7067 | 48.6980 | 0.8 | 否 | 是 |
| CIFAR-100 strict | S | 56.6933 | 9.6980 | 4 | 否 | 否 |
| CIFAR-100 strict | I | 56.8933 | 14.8838 | 4 | 否 | 否 |
| CIFAR-100 strict | A | 56.6933 | 9.6980 | 4 | 否 | 否 |
| CIFAR-100 strict | C | 57.5600 | 57.0025 | 16 | 否 | 是 |
| CIFAR-100 relaxed | S | 57.2400 | 24.1056 | 8 | 否 | 是 |
| CIFAR-100 relaxed | I | 57.0667 | 21.7906 | 4 | 是 | 是 |
| CIFAR-100 relaxed | A | 57.2400 | 24.1056 | 8 | 否 | 是 |
| CIFAR-100 relaxed | C | 56.8667 | 57.0025 | 36 | 否 | 是 |

I使用target-F进行重校准，S/A/C只使用source-F，这是不同的信息条件。不能根据这些结果宣称共享全胜，也不能隐瞒放宽队列中I通过而S失败。

S与A完全相同还具有结构原因：固定一个共享标量阈值时，每个seed的早退率和MAC节省都随阈值增加而不增加。在相同可行候选集合中，min与mean节省都偏好较低阈值；配合规定的并列规则，两目标选择同一候选。因此本轮不能给出最差seed目标优于平均目标的实证贡献。无需改参数制造不同结果。

M1使用既有评分实现的FP64 softmax；M2继承部署实现的Torch FP32 softmax。两个部分模型版本与数值范围均分别标明，不将M1阈值用于M2。

准备阶段验证：相关单元测试9项通过；六模型合成输入检查通过；启动器dry-run为ready。正式M2官方数据尚未实例化，待用户执行上述启动命令。
