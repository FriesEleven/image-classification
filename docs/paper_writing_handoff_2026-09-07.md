# 论文写作交接：截至 2026-09-07 的完整实验与结论边界

## 1. 任务和当前状态

这份文档供另一个对话接手论文写作。目标期刊为 Springer **The Journal of Supercomputing**，用户希望尽快产出可投稿稿件。当前任务是利用已有证据完成英文论文、表格、PDF 图片和最终编译验收；实验完成不代表保证录用。

- 服务器：`ssh -p 29695 root@connect.westb.seetacloud.com`。使用本机已有 SSH key，不在文档或仓库中存储凭据。
- 实验仓库：`/root/autodl-tmp/image-classification`。以下未特别说明的路径均相对此根目录。
- 本文编写前仓库 HEAD：`d06ec75e9a4c37b2fd79d69c1cd2a11eac84f95a`。
- 新实验硬件为 RTX 3080 Ti；CPU 训练使用 12 workers，P8 CPU 基准为 12 threads。RTX 4090D 仅是历史证据，后续无法再用。
- P7 六个 target 训练运行全部结束；一次性官方验证也已完成。**官方验证执行成功，但策略门槛未全部通过。**
- 当前停止新增训练和官方/外部测试访问，进入证据整理与写作。不得为让结果通过而改阈值、删 seed、替换模型或重跑。
- 用户希望已授权的工作直接推进，正常整理、复算已有数组、制图、写作无需反复询问。遇到确实缺失的数据、作者信息、无法安全合并的用户改动或新增实验需求，先完成不受影响的工作，再说明具体阻塞。

## 2. 阅读顺序与旧文档纠错

开始时检查 `git status --short --branch` 和 `git rev-parse HEAD`，保留所有已有用户修改。先读根目录 `AGENTS.md`，再读本文，随后按需读：

1. `docs/ccf_c_experiment_plan.md`：扩展实验动机与 P5–P8 设计。
2. `docs/paper_writing_guide.md`、`docs/paper_evidence_bundle.md`：历史 P1–P4 的写作规范与证据，但其当前状态须由本文和最新回执纠正。
3. `docs/p7_handoff.md`、`docs/p8_handoff.md`、`docs/p6_handoff.md`：阶段设计背景；其中尚称“待执行”的步骤必须用实际结果更新。
4. 下文列出的机器可读协议、回执、测试结果和对应原始数组索引。
5. 实际论文目录的说明、LaTeX 主文件、参考文献和模板。

重要纠错：

- `docs/handoff.md` 第 28 节的“P4 后停止实验”是历史决策，之后用户已授权扩展。
- P5-C 已做组件消融，不能再照抄“没有 no-KD/辅助头消融”的旧限制。
- 简化方法 A3 去掉 KD；P1/P4 历史完整配方的 official test 不能重新贴上 A3 标签。
- P6 实际是 ResNet-18 source 阶段停止，不能写成已通过第二 backbone 的完整 target/test 确认。
- P7 实际数据集是公共固定 **ImageNet-100**，不是计划早期写的 Tiny ImageNet-200，也不是完整 ImageNet-1K。
- P8 主硬件证据使用 v3 RTX 3080 Ti/CPU；旧 RTX 4090D 结果不能代替当前实际动态路由结果。
- `docs/p7_handoff.md` 中“官方验证待执行”已过时：现在已完成且未通过全部策略门槛，禁止再次启动其中命令。
- 既有 `reports/paper/` 表图和 claim ledger 主要覆盖 P1–P4；应更新版本，不能直接认定为最终论文资产。

若本文的四舍五入数字与原始文件冲突，以通过哈希核验的原始机器可读数据为准。不得为了消除冲突改写原始数据。

## 3. 推荐论文主线

建议以“跨训练 seed 共享的经验风险约束早退策略，以及其计算、类别风险和真实硬件收益的适用边界”为主线。

可用工作标题：

> Cross-Seed Shared Early Exiting: Empirical Risk Constraints and Hardware Trade-offs in Lightweight Image Classification

贡献应分别阐述：共享阈值选择及明确停止规则；同协议策略比较与组件消融；新模型 seed、自然分布转移、第二 backbone 和较大图像规模的成功与失败边界；动态实现的硬件条件性收益。

不要将 early exit、softmax confidence、KD 或辅助头本身宣称为首创。诚实报告负结果本身不等于方法创新，必须通过 P5 比较说明具体技术价值。不得宣称 SOTA、理论/分布无关风险保证、跨数据集普遍泛化或普遍硬件加速。

原稿若仍是 PSHA-Net/CSGHA/静态 SE-CBAM 注意力论文，需要结构性重写，不能仅替换方法名。可保留 Springer 模板和经核实的作者、单位、基金信息；不要虚构作者元数据。

## 4. 证据地图

### 4.1 历史正式测试和迁移

| 阶段 | 主要证据 | 正确用途 |
|---|---|---|
| P1b CIFAR-10 | `reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json`、`test_results.json`、`source_index.json` | 历史方法锁定测试，阈值 0.984 |
| P2 新 seed | `reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json` | 固定策略的新模型迁移 |
| CIFAR-10.1 v6 | `reports/experiments/2026-09-03-early-exit-p2-cifar10-1-v6/external_results.json`、`source_index.json` | 已完成的一次性自然分布转移评估 |
| P3 CIFAR-100 | `reports/experiments/2026-09-03-early-exit-p3-cifar100/selection.json` | 严格门槛不可行的停止边界；不是测试结果 |
| P4 独立确认 | `reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json` | 新 seed、新 split 的确认，阈值 0.903 |
| P4 正式测试 | `reports/experiments/2026-09-03-early-exit-p4-cifar100-test/test_results.json`、`source_index.json` | 历史配方的方法锁定测试 |

P3 后验诊断在 `reports/diagnostics/` 对应目录，只能作为诊断。项目历史上无关 baseline 曾接触 CIFAR-100 test，不能声称整个项目从未见过该 benchmark。

### 4.2 P5 策略比较和组件消融

- P5-A/B 最终候选输出：`artifacts/analyses/early_exit_p5ab_20260905_092909/`。必须先读其 `README.md`、`p5ab_results.json`、`development_logits_manifest.json`，核验完成状态和设计对应关系；同名前面的时间戳目录不得自动混用。
- 设计：`reports/experiments/2026-09-05-early-exit-p5-design/protocol_manifest.json` 和 `protocol_amendment_1.json`。
- 重点表：`tables/strategy_comparison.csv`、`matched_compute.csv`、`paired_bootstrap.csv`、`calibration_metrics.csv`、`decision_complementarity.csv`、`per_class_route_risk.csv`、`threshold_sensitivity.csv`。
- 现有图位于该目录 `figures/`；使用前核验数据范围、基线身份和标注，不以存在文件代替审计。
- P5-C 正式回执：`reports/experiments/2026-09-05-early-exit-p5c-analysis/analysis_receipt.json`。
- P5-C 原始分析：`artifacts/analyses/early_exit_p5c_20260905_143000/`；包含 `p5c_results.json`、`tables/variant_seed_metrics.csv`、`aggregate_metrics.csv`、`factorial_effects.csv`、`failure_gates.csv`。

P5-C 新训练完成 18 个运行，结论为 `simplify_to_a3_no_kd`。CIFAR 上 A3 为 exit8 + 训练专用 exit16，CE-only。KD 未呈现稳定同方向收益；辅助头收益依赖数据集。上述结论是 development-only，不能伪装成新 A3 已完成 CIFAR official test。不要把 historical/full 与 A3 结果拼成同一方法的统一测试行。

### 4.3 P6 第二 backbone

正式回执：`reports/experiments/2026-09-06-early-exit-p6-source-analysis/analysis_receipt.json`。

ResNet-18 在 CIFAR-10/100 上完成 source 阶段，但两个数据集都没有符合全部冻结要求的共享阈值，状态 `stop_without_target`。报告架构拓展尝试与不可行边界，不宣称成功迁移至 ResNet-18，也不要新增 target 实验。

### 4.4 P7 ImageNet-100

设计：

- `reports/experiments/2026-09-06-early-exit-p7-imagenet100-design/protocol_manifest.json`
- 同目录 `architecture_profile.json`
- `reports/experiments/2026-09-07-early-exit-p7-source-analysis-design/protocol_manifest.json`
- `reports/experiments/2026-09-07-early-exit-p7-target-design/protocol_manifest.json`

结果：

- Source 回执：`reports/experiments/2026-09-07-early-exit-p7-source-analysis/analysis_receipt.json`
- Source lock：`artifacts/analyses/early_exit_p7_source_20260907/source_lock.json`
- Target 回执、audit、清理回执及 official lock：`reports/experiments/2026-09-07-early-exit-p7-target-analysis/`
- Target 原始结果：`artifacts/analyses/early_exit_p7_target_20260907/target_results.json`
- **最新官方结果**：`artifacts/analyses/early_exit_p7_official_test/test_results.json`
- 同目录：`started.json`、`completed.json`、`image_inventory.json`、六份 `seed{80,81,82}_{A0,A3}.npz`。

官方结果目前主要位于 ignored artifacts；写作第一阶段应生成版本化的精简正式回执和结果表，并把输入哈希列入论文 evidence manifest，保留原始文件不动。

关键协议：126,689 张训练池图片，106,689/10,000/10,000 划分为训练、模型选择验证、策略校准；固定 100 类；输入 224×224。Source seeds 77–79，target seeds 80–82；A3 为 exit8 + 训练专用 exit15，无 KD；共享阈值 0.850。Source/target 使用相同校准图像，因此 target 是**新模型 seed 确认，不是新图像 holdout**。独立图像 holdout 是官方验证的 5,000 张图片，每类 50 张。

成本：final path 299,622,272 MACs；exit8 path 含头为 131,141,376；exit8 head 为 6,400；fallback 为 299,628,672。部署跳过 exit15，不能将其计入或作为第二路由出口。MAC 仅按现有 conv/linear 计数口径，不等于完整时延或能耗。

### 4.5 P8 真实硬件

正式目录：`reports/experiments/2026-09-05-early-exit-p8-v3-analysis/`，读取 `analysis_receipt.json`、`README.md`、`seed_summary.csv`、`aggregate_summary.csv` 和 `actual_latency_saving_by_batch.pdf`。

原始输出：`artifacts/analyses/early_exit_p8_v3_20260905_192249_293181/`。完整审计覆盖 1,200 个数组、1,200,000 个时延观测，以及路由和数值正确性检查。

- RTX 3080 Ti、batch=1：CIFAR-10 时延节省 17.14% ± 1.45 pp；CIFAR-100 为 -0.04% ± 2.74 pp，近乎持平。
- GPU batch=4/8/16/32 均出现减速，必须保留在图表中。
- CPU CIFAR-10 在测量的 batch 下均有收益；CIFAR-100 随 batch 变化，不能一概而论。
- 参考对象是同一 A3 checkpoint 的 final-only path，不能写成独立 A0 模型。
- 先在每 seed 内平均重复轮次，再报告三个 seed 的均值与样本标准差；不能把百万次时延采样当成百万个独立训练重复。
- CPU RSS 是进程 lifetime high-water；遥测快照不是能耗积分。
- P8 v2 因 TF32 数值一致性问题停止，不能混用其时延。旧 RTX 4090D 结果最多作为历史补充并准确区分估计与实测。

## 5. P7 官方验证：必须如实报告的结论

验证运行成功，结果/marker 哈希和六份 logits 已检查，并从保存数组复算准确率、路由率和最差类别下降一致。没有重新执行模型或重开测试集。

- 官方结果 SHA-256：`ac009a3203a3579997c8a5665b63ba1d766cc685f53d865b24017c3700afb749`
- 官方 lock SHA-256：`9b6d4908ec3138b06e5e565a5dfad7e71bf1b4e3e6c28f5dd5eebc7fae0028f2`
- 状态 `completed`；`all_policy_gates_passed=false`；阈值候选数 0。

| Seed | A0 accuracy (%) | A3 final (%) | Policy (%) | Overall/balanced drop (pp) | Worst-class drop (pp) | MAC saving (%) |
|---|---:|---:|---:|---:|---:|---:|
|80|81.64|82.82|82.52|0.30|4.00|14.3598|
|81|81.58|82.40|81.94|0.46|8.00|14.3036|
|82|81.18|82.14|81.92|0.22|4.00|14.5173|

冻结门槛：总体与 balanced accuracy 下降各 ≤0.50 pp、最差类别下降 ≤4 pp、MAC 节省 ≥15%、早退率 15%–95%。三个 seed 的计算节省均不足 15%，seed81 最差类别超标（类别 `n03259280`）。每类 50 个样本使 1 个样本对应 2 pp；8 pp 是净少判对 4 个样本，不能因样本少而把失败改判成功。

三 seed 均值：A0 81.4667%，A3 final 82.4533%，policy 82.1267%；policy 相对 A0 +0.6600 pp，相对 A3 final -0.3267 pp；MAC 节省 14.3936%。均值不能掩盖“每个 seed 都须满足”的门槛失败。

可写：

> On the fixed ImageNet-100 official validation subset, the locked policy reduced MACs by 14.39% on average while retaining a 0.66 percentage-point accuracy advantage over the independently trained baseline. However, it did not satisfy all preregistered constraints: savings fell below the 15% floor for every seed, and one seed exceeded the 4-point worst-class degradation budget.

不可写：P7 完整通过、约束具有独立测试保证、成功泛化到所有类/所有数据集、测试后选择较低阈值仍属于原预注册结果。

这一结果不自动决定拒稿，也不自动构成创新。论文质量仍取决于创新性、合理基线、系统意义和论证深度。不得保证 CCF-C 录用或编造录用概率。

## 6. 写作执行步骤与交付物

### A. 先建立可追溯证据包

1. 核对上述结果路径、原始状态、checkpoint/array 哈希与 seed 数量。锁和 access marker 永久保留。
2. 为 P7 官方结果新增版本化回执：记录真实失败项、六个模型、阈值、输入哈希、状态及 independent-array-recalculation。不要把原始 NPZ/图片/checkpoint 加入 Git。
3. 更新 `reports/paper/evidence_manifest.json` 与 `claim_to_evidence.md`，扩展到 P5–P8；分开 original recipe 与 A3、development 与 official test。
4. 盘点现有 LaTeX 实际位置，不假设旧电脑路径有效。先备份或纳入版本管理，保留作者修改。实验代码根目录与论文模板目录可能不同。

### B. 从源数据生成最终表图

优先复用 `scripts/analysis/build_early_exit_paper_tables.py`、`scripts/visualization/plot_early_exit_paper.py` 及 P5/P8 已有脚本；先审查是否硬编码旧配方、设备或状态。

至少需要：

1. Protocol/model-version 表：数据、划分、seed、配方、阈值来源、评估阶段。
2. 历史正式测试与迁移表：P1/P2/CIFAR-10.1/P4，准确标记配方。
3. P5 策略比较表：同风险/同计算量比较，公平区分逐模型重校准和跨 seed 共享；oracle 只能是上界。
4. P5-C factorial 消融表：按 dataset/seed 报告 KD 与训练辅助头影响，区分 final/early/policy。
5. P6/P7 边界表：source、target、official 分列，显式显示停止与失败项。
6. P8 实际时延、p95、吞吐与 batch 关系；仅使用现有且口径成立的指标。
7. 方法流程图、经验风险–计算图、消融/迁移图和实际时延随 batch 的图。新图片输出 PDF，字体、裁切和图例经渲染检查。

从 JSON 原始小数计算，再统一展示四舍五入；accuracy % 与差值 pp 分清，均值±SD 使用样本标准差（ddof=1）。统计单位明确：训练 seed、图像和计时轮次不可混淆；只有三个 seed 时不能作强显著性结论。

允许复算已保存 logits/routes、统计和绘图；禁止在 official/external test logits 上扫描新阈值、选择策略或挑 checkpoint。已有测试数组只用于冻结策略复核、统计和预先定义结果展示。若需要新探索，应限于已授权的 development 材料并明确后验性质，不能改变正式结论。

### C. 先写方法与实验，再写摘要

- Method：定义单个部署出口、训练辅助头、shared threshold、经验总体/balanced/worst-class 约束、跨 source seeds 的可行性和 max-min saving 目标，解释 fallback 额外头成本。
- Experimental setup：明确配方演变、数据边界、seed 和硬件。P7 target 新模型不等于新图像。
- Results：依次给策略比较、消融、迁移和失败边界、实际系统表现；完整区分历史方法与简化 A3。
- Discussion：讨论为何开发集门槛不保证独立样本门槛、类别样本离散性、confidence/routing 变化，以及 MAC 节省不必然转为 GPU 时延收益。未经验证的原因必须写成可能解释，不能当作已证实机制。
- Limitations：三 seed、有限每类样本、固定100类子集、P6 source stop、P7 official constraints failed、硬件/batch 依赖、非统计风险保证、方法版本测试覆盖不齐。
- 最后完成 Introduction、Related Work、Abstract、Conclusion，使每一项贡献和数字都能对应正式表格。

Related Work 和期刊格式必须用原始论文/出版社当前页面核验；不要仅照抄旧计划中的文献条目。期刊官方范围入口：`https://link.springer.com/journal/11227/aims-and-scope`。它强调计算系统、算法、性能与深入的新进展；系统性能证据和方法比较是投稿定位的重点，不能只靠分类精度提升。

### D. 编译与最终验收

保留合适的 Springer 模板，更新 BibTeX，核对所有引用真实性。用实际工具链编译，逐页渲染检查 PDF，修复 undefined citation/reference、重复 label、明显 overfull 和图表裁切。

交付：英文论文源文件、BibTeX、PDF 图、生成脚本、源数据表、最终 PDF、claim-to-evidence 清单和剩余限制说明。对摘要—正文—表图—结论逐项核对同一数字和单位。当前目标是完成可审阅论文，不保证录用。

## 7. 保留和清理边界

P7 source/target 各已删除 60 个周期快照（每批 1,724,722,152 bytes）。best/latest/final、配置、训练日志、splits、provenance、manifest、原始 logits、官方图片 inventory 与永久访问标记均保留。不要再运行旧 cleanup 脚本，它们可能假设文件尚存在。

不要把失败 seed、P3/P6/P7 负结果、P8 v2 失败运行或旧配方证据视为“与论文无关”。这些关系到复现与研究过程。写作阶段无须扩大数据删除范围。

## 8. 可复制到另一个对话的启动提示

```text
请连接 ssh -p 29695 root@connect.westb.seetacloud.com，在
/root/autodl-tmp/image-classification 中先检查 Git 状态，读取 AGENTS.md 和
docs/paper_writing_handoff_2026-09-07.md，再按其中证据入口开展论文写作。

目标为 Springer The Journal of Supercomputing，正文使用英文。实验已结束，不新增训练，
不重新访问任何 official/external test，也不扫描测试集阈值。P7 官方验证运行成功但门槛失败：
三 seed MAC 节省均低于15%，其中一个 seed 最差类别下降8pp；必须完整报告。

先核对机器可读证据并把 P7 官方结果整理成版本化回执，更新表格与 claim-to-evidence；
分清历史完整配方和简化A3的测试覆盖、development和official、新模型seed和新图像holdout。
P8以RTX3080Ti/CPU v3实际测量为主，保留大batch减速和负结果。禁止声称普遍风险保证或加速。

定位实际LaTeX稿，保护旧稿与作者修改，围绕共享经验风险约束早退方法结构性重写。
从现有数据生成表格和PDF图，再写方法、实验、讨论、局限，最后写摘要和引言；
核验原始参考文献，编译并逐页验收最终PDF。继续到可审阅稿件与可追溯资产完整，
只在实质阻塞或需要新增实验时说明证据并请求必要信息，不反复询问已授权的实现细节。
```
