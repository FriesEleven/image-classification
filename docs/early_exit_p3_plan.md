# CIFAR-100 P3 最终第二数据集复现协议

## 1. 目的与停止边界

CIFAR-10 P1b、未见重训版本 P2a 和一次性 CIFAR-10.1 v6 外部分布评测均已给出正结果。当前论文证据
的主要缺口是只有一个训练数据集和一个 backbone。对 CCF-C 目标，最小的下一步不是继续扫描注意力、
阈值或更多 CIFAR-10 seed，而是在 CIFAR-100 上完整复现一次“source 选择共享策略→未见 target 模型
零重校准迁移”的机制。

P3 是当前方向的最后一批计划训练。通过后不再追加训练 seed；失败则作为第二数据集边界归档，不修改
gate、不查看 P3 模型的官方 test、不用新阈值重跑。

## 2. 一批 12 组的冻结矩阵

- 数据：CIFAR-100；固定 `split_seed=20260903`。
- 每组：40,000 train / 5,000 checkpoint-selection validation / 5,000 policy calibration，三者互斥且
  每个划分按100类均衡；官方10,000张 test训练期间不迭代。
- source training seeds：60/61/62；target training seeds：63/64/65。
- 每个 seed 一对 matched 模型：原生 MobileNetV2 与 `features[8]`、`features[16]` 两个轻量头的
  multi-exit MobileNetV2，共 `6×2=12` 次。
- 训练契约保持 P1/P2 不变：200 epochs、batch128、AdamW/OneCycleLR、AMP、CUDA Graph、8 workers、
  prefetch8、串行 `jobs=1`；最终头只用自身CE，两个出口分别使用0.2/0.3权重、alpha0.5、temperature3
  的停止梯度最终头蒸馏；best checkpoint只由最终头 validation accuracy 选择。
- 运行 ID 默认包含全新 tag `p3a`；任何目标目录已存在、仓库不干净或已有训练进程时启动器拒绝运行。

一次性训练12组而不是先6组再开第二批，是为了在任何 CIFAR-100 策略结果可见前就冻结 target 模型，
同时避免多一次人工启动。预计串行约4.5小时；小模型利用率低于满载是正常现象，不并发改变训练条件。

## 3. 预注册策略选择与迁移 gate

策略仍只允许 exit8→final；exit16仅作为训练辅助头，不参与路由。source seeds60/61/62共同使用一个
exit8最大softmax阈值。候选固定为0.000–1.000、步长0.001，加一个 final-only 哨兵；无类别保护。
目标依次最大化三个source模型中的最小MAC节省和平均MAC节省。

source每个模型都必须满足：总体、balanced及worst-class经验下降≤0；早退率15%–95%；MAC代理节省
≥15%。选中阈值随后原样作用于target seeds63/64/65；target候选阈值数为0、不得逐模型重校准，并须
通过同一组风险/动态路由/预算gate。六个matched pair的multi-exit最终头还必须满足：相对baseline
平均validation差值≥−0.30pp、每个seed≥−0.75pp。任一项失败即`stop_without_test`。

MAC路径成本按100类输出头重新计算：exit8/exit16/final分别为
2,682,624/5,249,088/6,240,128 MAC，归一化为42.99%/84.12%/100%；所以≥15%节省实际要求
早退率至少约26.31%。这些仍只是conv/linear操作代理。论文延迟证据继续使用已有真实分阶段续算的
配对GPU测量，不把MAC直接称为延迟。

## 4. 审计和方法锁定后的官方 test

训练结束后先对唯一completed manifest做文件级审计：12组/串行时间线/200轮有限指标、首次best、
best/latest/final和20个周期checkpoint、40k/5k/5k划分、源码快照/provenance、无test及无预测文件。
只有`issues={}`才执行预注册P3分析器。

若全部开发集gate通过，先把manifest、审计、selection、12个best checkpoint和评估器哈希写入版本化
lock并提交；然后只允许一次“方法锁定后的 CIFAR-100 official-test 评估”。历史 seeds42/43/44 的旧
baseline曾保存过 CIFAR-100 test 指标，因此不得把这次描述为全项目首次盲测；准确表述是P3模型和
P3早退策略从未在test上评估或调参。

test结果不用于再选择策略。预先固定的论文确认边界为：六个模型各自总体/balanced下降≤0.50pp、
worst-class下降≤2.00pp、早退15%–95%、MAC节省≥15%，且六seed平均总体下降≤0.20pp。无论结果
是否通过，都保留完整逐seed/逐类结果和logits/routes，禁止调整后重跑。

对应入口已在训练前固定为`scripts/analysis/freeze_early_exit_p3_test.py`和
`scripts/analysis/evaluate_early_exit_p3_locked_test.py`。评测器要求lock、audit、selection和自身都已
提交且工作区干净，在构造test loader前独占创建started marker；即使中途失败，该marker也禁止第二次
访问。`--verify-only`只核验哈希，不构造或迭代test loader。

## 5. 完成后保留与清理

保留12组best/latest/final、训练CSV/TensorBoard、配置、provenance、split、manifest/source snapshot、
审计、选择结果和一次性test记录。正式审计与后续推理均完成后，才删除12组`epoch_*.pth`周期优化器
快照并写清理回执。若test确认边界通过，停止新增训练并进入论文表格、图和写作；若失败，保留边界
结论后重新评估论文叙事，不追加seed掩盖失败。
