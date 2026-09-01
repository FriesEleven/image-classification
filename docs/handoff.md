# 实验项目交接：CSGHA负结果归档 / 预算阶段选择（更新至2026-09-01）

本文用于让新的对话接续当前代码、GPU服务器实验和论文计划。**先读本文件，再检查实时状态；不要把历史建议、短测或旧清单的残留字段当成当前正式结果。**

最新状态见第18、19节；第1、7、9、15、16节保留CSGHA阶段的历史计划，若冲突，以第18、19节为准。当前没有由Codex启动新的正式训练。

> **2026-09-01当前状态：** v6六组已完成正式审计和版本匹配诊断，三个seed全部落后matched control；CSGHA v3–v6已作为负结果/机制消融归档，不进入v7。服务器上不存在v7文件、v7进程或其他训练进程。当前正向路线改为“预算约束下的阶段感知稀疏注意力部署”；代码、硬件成本档案、30组最小校准矩阵、串行后台启动器和选择脚本均已准备并验证，下一步只需用户启动第19节的`probe1`批次。

## 1. 一分钟了解当前进度

- 研究方向：MobileNetV2上的轻量跨阶段引导注意力（CSGHA），数据集CIFAR-10/100。用户明确要求**坚持新机制路线**，不是把独立SE+CBAM换名包装成创新。
- 已完成：CIFAR-10/100数据配置、45k/5k划分、统一训练循环、18次正式历史实验审计、v1/v2/v3五个checkpoint的版本匹配诊断。
- 当前结论：guidance确实利用了输入信息，但v3对同位置独立组合的三seed平均优势仅 **0.073个百分点**，尚不能声称稳定有效。
- 已实现下一候选：v4仅将目标block的deep channel MLP激活从ReLU换成LeakyReLU(0.1)，同时实现完全匹配的无guidance控制。
- 当前正式候选配置：`configs/experiments/csgha_v4_leaky_middle.yaml`、`independent_leaky_middle.yaml`。
- 最新执行后端：**perf2，两路独立实验并行**，训练CUDA Graph，每组batch128、8个data workers、1个PyTorch CPU线程。不是两张卡，也不是DDP或把两个模型合成batch256。
- 下一批：两种模型 × seeds42/43/44，共6次200epoch、validation-only训练，全部从头运行。
- **核实时正式perf2六组尚未启动**，六个目标目录均不存在，没有训练进程，GPU为0%/0MiB。用户可能在本文生成后自行启动，所以新对话必须重新检查。
- 旧v4批次首组在epoch100/200后按用户要求中断，其他五组未开始；日志/权重保留，不能当成六组完成，也不从这个checkpoint续训。
- 最新验证：GPU服务器 **75 tests passed + 3 subtests passed**；两种模型各3轮的串行/并行权重逐参数一致、完整training.csv字节一致。
- 正式v4准确率、三seed稳定性和v4机制诊断结果均**未知**。吞吐优化不等于方法精度提升。

用户此前的固定偏好：需要新实验时，准备可运行的后台Python脚本，给出**一行启动命令**，由用户启动；不要在读完交接后自行启动长训练。用户如果明确授权启动，再执行。

## 2. 工作目录、环境与Git

### 2.1 目录与连接

| 用途 | 位置 |
|---|---|
| GPU服务器代码与产物根目录 | `/root/autodl-tmp/image-classification` |
| 本交接文件服务器位置 | `/root/autodl-tmp/image-classification/docs/handoff.md` |
| 服务器Python | `/root/miniconda3/bin/python` |
| 本地代码目录 | `/Users/salt/Jlu/paper/classification-code` |
| 本地论文目录（另一个项目） | `/Users/salt/Jlu/paper/sn-article-template` |
| 论文原始大修交接 | `/Users/salt/Jlu/paper/sn-article-template/docs/handoff.md` |
| 论文主文件 / 参考文献 | `sn-article.tex` / `sn-bibliography.bib`，位于论文目录 |

本机已配置并验证SSH公钥登录。连接使用现有SSH配置的host：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 connect.westb.seetacloud.com
```

端口与IdentityFile以本机SSH配置为准，实例曾迁移，不沿用旧聊天中的端口。文档和Git中不要保存密码、私钥内容或令牌。VSCode终端已经位于服务器项目目录时，不需要再SSH一次。

已验证环境：Python3.12.3，Torch2.8.0+cu128，RTX3080Ti 12GB。容器CPU配额为12核，尽管可见逻辑CPU更多。`nvidia-smi`显示的CUDA13.0是驱动支持上限，不是当前PyTorch使用的运行时；`torch.version.cuda`为12.8。不要因为这个显示差异重装环境。

```bash
cd /root/autodl-tmp/image-classification
/root/miniconda3/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

### 2.2 Git与发布约定

- GitHub：`https://github.com/FriesEleven/image-classification`；发布目标为`main`。
- 本地开发分支为`feature/support-remote-deploy`，服务器使用`main`。不要仅凭本地分支名误判服务器版本。
- 此轮工作前共同历史起点为`64b79160b6a582086443eb45533311bfc1aaf1f0`。新的v4、诊断、Graph和并行队列并不是这个旧commit自带的代码。
- 本次发布使用普通快进推送，不强推。不把旧审计JSON中记录的commit替换为当前commit。
- 进入新对话后执行以下命令了解实际提交；不要猜测发布哈希或默认工作区干净：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 -- docs/handoff.md
```

服务器先前通过rsync部署，曾出现`HEAD旧、工作区新`。历史实验的准确执行版本应结合`provenance.json`、逐文件hash和`source_snapshot/`确认；只有HEAD不足以复现dirty工作区。新发布后也不能回填历史运行版本。

## 3. 项目结构与职责

```text
image-classification/
├── AGENTS.md                         # 仓库约定，开始改代码前阅读
├── README.md                         # 安装、常用入口；旧实验命令不等于当前待办
├── pyproject.toml                    # 包与依赖定义（未完全锁定所有版本）
├── train.py                          # 历史兼容入口
├── src/image_classification/
│   ├── config.py                     # ExperimentConfig、YAML/CLI、实验ID/架构版本
│   ├── paths.py                      # 项目/数据/产物路径，避免依赖调用者cwd
│   ├── cli/train.py                  # 训练CLI到engine的薄入口
│   ├── data/cifar.py                 # CIFAR10/100、分层划分、增强、DataLoader
│   ├── data/cifar10.py               # 旧接口兼容
│   ├── models/attention.py           # SE/CBAM/Guided-CBAM、激活与guidance公式
│   ├── models/mobilenetv2.py         # backbone包装、位置插入、跨阶段描述传递
│   ├── models/factory.py             # 按model_type构建模型
│   ├── models/eca.py                 # 项目适配的ECA实现
│   ├── training/engine.py            # epoch训练/验证/选best/保存/可选test
│   ├── training/evaluate.py          # epoch指标、FP32验证、预测导出
│   ├── training/cuda_graph.py        # 训练fwd/bwd图捕获；eval/尾batch走eager
│   ├── training/optimizer_step.py    # AMP跳步与scheduler更新保护
│   ├── training/checkpoint.py        # CSV与checkpoint持久化
│   ├── training/benchmark.py         # 参数/估算FLOPs/独占推理计时
│   ├── training/provenance.py        # 源码hash、依赖、Git信息、源码快照
│   ├── training/sweep.py             # 2/3路有界队列、日志、失败/中断清理
│   ├── diagnostics/guidance.py       # 历史版本模型加载与guidance干预
│   └── utils/reproducibility.py      # 随机种子
├── configs/
│   ├── experiments/                  # 可运行、版本化的实验YAML
│   └── sweeps/                       # baseline/位置/stability/v4批次清单
├── scripts/
│   ├── train.py                      # 推荐训练入口
│   ├── launch_csgha_v4.py            # 当前唯一推荐的正式新实验启动器
│   ├── run_baselines.py              # 通用记录型runner，支持--jobs 1/2/3
│   ├── launch_*.py / launch_baselines.sh  # 历史批次入口，先确认是否真的要重跑
│   ├── run_experiments.py            # 历史全矩阵入口，当前不要直接运行全矩阵
│   ├── analysis/                    # 审计、统计、表格生成
│   ├── diagnostics/                 # 模型/数据检查、历史干预与吞吐短测
│   └── visualization/               # 混淆矩阵、Grad-CAM、ROC、t-SNE
├── tests/unit/                       # 配置、模型、调度、溯源、审计等测试
├── tests/integration/                # 输出、validation-only、CUDA Graph一致性
├── tests/fixtures/                   # 少量版本化测试图片
├── third_party/eca_net/              # 上游代码和许可证，不是运行时主实现
├── docs/                             # 本交接和当前/历史计划
├── reports/                          # 版本化的审计结论、精简证据、论文图表
├── assets/                           # 展示样例
├── data/                             # 本地CIFAR数据，除README外Git忽略
└── artifacts/                        # 原始实验产物，全部Git忽略
```

新模型逻辑放package，不在脚本里复制架构。先保留用户已有改动。论文项目是独立目录，本轮没有改写LaTeX，也不把整个论文目录拷到代码仓库。

## 4. 证据阅读顺序

1. 本文：当前运行状态、接续步骤与限制。
2. [当前实验计划](experiment_plan.md)。
3. [18次正式实验审计](../reports/audits/2026-08-30/README.md)及[逐seed汇总](../reports/audits/2026-08-30/experiment_summary.md)。
4. [版本匹配guidance诊断结论](../reports/diagnostics/2026-08-30-guidance/findings.md)及[可复算结果](../reports/diagnostics/2026-08-30-guidance/results.md)。
5. [最新perf2并行验证](../reports/diagnostics/2026-08-30-throughput/perf2_parallel.md)和对应`perf2_evidence.json`。
6. [perf1训练图优化历史](../reports/diagnostics/2026-08-30-throughput/findings.md)。

时间顺序很重要：审计阶段曾建议降低方法主张，但用户随后明确选择继续新机制，之后才有版本匹配诊断和v4。旧报告的“不自动进入v4”不能覆盖用户后续决定；也不能把诊断里的早期“串行六组/5–7小时”当成最新perf2配置。

`docs/paper_plan.md`已标明是历史ECA prime/深层auxiliary构想，含预写效果结论，不是当前执行计划。`reports/tables/`和很多`reports/figures/`也是旧协议材料，不能与新18次审计结果直接拼成主表。

## 5. 当前研究问题与已完成结果

### 5.1 统一协议的18次正式实验

以下均为 **best validation accuracy（%）**。均值±样本标准差（ddof=1）；不是test结果。

| 数据集/变体 | Seeds | 结果 |
|---|---|---:|
| CIFAR-10 baseline | 42/43/44 | 87.69 ± 0.16 |
| Independent shallow（SE1–2、CBAM1–2） | 42/43/44 | 88.27 ± 0.46 |
| Independent middle（SE1–2、CBAM7–8） | 42/43/44 | 87.94 ± 0.53 |
| Independent deep（SE1–2、CBAM15–16） | 42 | 88.36（单次） |
| CSGHA v1 middle | 42 | 87.50（单次） |
| CSGHA v2 middle | 42 | 88.20（单次） |
| CSGHA v3 middle | 42/43/44 | 88.01 ± 0.58 |
| CIFAR-100 baseline | 42/43/44 | 56.94 ± 0.62 |

v3逐seed validation为88.68/87.62/87.74；matched middle为88.46/87.96/87.40。配对差值为+0.22/−0.34/+0.34个百分点，均值+0.073。v3对shallow为−0.10/−0.54/−0.14，三次都落后。

因此v3“用了guidance”与“guidance稳定提升任务性能”是两回事。不能只展示seed42，也不能把相对baseline的所有增益归因于guidance。

历史baseline已有test结果：CIFAR-10为87.48±0.23%，CIFAR-100为56.68±0.24%；旧配置可能没有`evaluate_test`字段，但不能据此称test未用。它们不能与其他模型88.xx%的validation直接计算test提升。

历史manifest：

- baseline：`artifacts/sweeps/mobilenetv2_baselines_20260829_162111/manifest.json`，运行commit `a99a701418abf025ffaaf37d9392d16b58aa104f`。
- 三组补seeds43/44的stability：`artifacts/sweeps/cifar10_attention_stability_20260830_151514/manifest.json`，于2026-08-30 19:43:48完成，运行commit `64b79160b6a582086443eb45533311bfc1aaf1f0`。
- 部分seed42 standalone缺少运行时commit。保留这个缺口，不用审计时HEAD补造。

### 5.2 Guidance版本匹配诊断已完成

五个best checkpoint使用对应历史源码、strict load，完整5,000张validation均精确复现原准确率；没有新test评估。

| checkpoint | 原始准确率 | 全validation置换guidance后的平均下降 |
|---|---:|---:|
| v1 / seed42 | 87.50 | 4.09个百分点 |
| v2 / seed42 | 88.20 | 5.25个百分点 |
| v3 / seed42 | 88.68 | 0.67个百分点 |
| v3 / seed43 | 87.62 | 0.43个百分点 |
| v3 / seed44 | 87.74 | 0.61个百分点 |

置换使用7301/7302/7303，覆盖整个validation、无自身配对，保留描述边际分布；它们不是额外训练seeds。另做置零、训练集平均描述、训练集平均guidance加项。均值只来自对应45k训练子集的evaluation transform，不用validation或test估计。

v3目标block7/8的deep logits零值比例：seed42为100%/100%，43为95.14%/99.98%，44为83.16%/99.92%。负的deep MLP第一层输出被ReLU截断。这个观察提供v4假设，但尚不证明它是精度不足的原因。

结论边界：置换同时改变浅/深联合分布，不能把精度下降称为互信息量或从头训练消融；内部tanh饱和也不等于输出完全没有输入信息。

证据：`artifacts/diagnostics/csgha_information_20260830_v1/`包含manifest、run JSON、配对预测NPZ、索引和诊断源码副本。报告在`reports/diagnostics/2026-08-30-guidance/`。

### 5.3 v4方法定义

浅层SE位于torchvision MobileNetV2 `features`索引1、2；描述取SE之后的block2：`g_s=GAP(F_2^SE)`，维度24。目标Guided-CBAM位于索引7、8，通道64，各有独立投影`24 → 6 → 64`。

```text
deep = MLP(AvgPool(F_t)) + MLP(MaxPool(F_t))
guidance = tanh(alpha_t) * tanh(P_t(LayerNorm(g_s)))
channel_gate = sigmoid(deep + guidance)
```

之后仍有标准CBAM空间注意力。只传24维通道描述，不传完整高分辨率特征。每个alpha标量初始化为0。

- `csgha`代表旧v3：deep激活ReLU，guidance用有界加项。
- `csgha_v4`：仅deep MLP的激活换成LeakyReLU(0.1)，guidance投影内部仍是原ReLU/LN/tanh，不额外改损失、归一化或位置。
- `hybrid_leaky`：相同SE1–2/CBAM7–8、相同deep LeakyReLU，但无guidance，是本批核心matched control。
- 参数量：control 2,238,024；v4 2,239,178；新增guidance相关1,154参数。LeakyReLU本身不是创新点。
- `deep_activation_version`是持久化buffer，防止无参数激活变化被旧权重静默兼容；新旧架构应strict load并核对`architecture_version`。

旧v2/v3可能权重键完全相同，strict load成功仍不代表公式相同；这正是需要版本匹配源码的原因。

## 6. 统一训练协议及不可悄悄改变的事项

| 项目 | 当前正式perf2 |
|---|---|
| 数据 | CIFAR-10，本轮只跑该数据集；已支持CIFAR-100 |
| 划分 | 官方50k训练图像按类别分层成45k/5k；另有官方10k test |
| seed | 42/43/44；同一个seed同时控制划分和训练 |
| 输入/backbone | 32×32，torchvision MobileNetV2，weights=None随机初始化 |
| epochs | 200，无early stopping |
| batch/accumulation | 128 / 1；每轮352 batches，最后72张保留 |
| optimizer | 默认非fused AdamW，weight_decay=1e-4，betas=(0.9,0.999) |
| scheduler | OneCycleLR，max_lr=0.01，pct_start=0.3，cos，div_factor=25，final_div_factor=100 |
| 更新规则 | 每次实际optimizer更新后step；AMP溢出跳步时scheduler不推进 |
| loss | CrossEntropy，无label smoothing；日志loss仍为batch means的平均 |
| 增强 | crop32/padding4，水平翻转，ColorJitter各0.2，rotation10°，normalize，RandomErasing p0.1 |
| validation/test | 仅ToTensor和各数据集normalize；eager FP32 |
| 选checkpoint | 最高validation accuracy；相同时保留首次最佳epoch |
| test | `evaluate_test: false`，不参与位置、版本或参数选择 |
| 执行 | cuda_graph=true；AMP cache禁用；torch_num_threads=1 |
| 数据加载 | 每组8个persistent workers，prefetch_factor=4，CUDA时pin_memory |
| 并发 | 2个完全独立进程，各有模型、optimizer、RNG、输出目录 |
| 推理测量 | measure_inference=false，benchmark.json明确skipped/null |

`lr=0.01`是OneCycle的max_lr，不是从头到尾固定lr。训练与划分seed绑定，三seed标准差包含划分变化，不是固定validation上的纯初始化方差；不要自行拆开seeds而沿用旧表。

CUDA Graph要求AMP关闭权重缓存，和旧eager/cache的浮点轨迹不完全相同；新v4与新control必须同后端比较。从旧v3到新v4的跨批次精度差不能单独归因于LeakyReLU。若要做激活的因果对照，应在同一后端再跑ReLU版本并单独计划。

图捕获会预热BN/Dropout，当前实现捕获后恢复模型state与CPU/CUDA RNG；最后不足整批走eager，不pad/drop。optimizer/scaler/scheduler仍在图外，不能为追求利用率破坏AMP跳步保护。

## 7. 下一批的启动、检查与结束条件

### 7.1 先查，后决定是否启动

```bash
cd /root/autodl-tmp/image-classification
git status --short
nvidia-smi
ps -eo pid,ppid,args | grep -E '[s]cripts/(train|run_baselines)\.py'
ls -lt artifacts/launcher_logs/ | head
```

状态核实时没有正式perf2目录。预计实验ID如下（每项3个seed）：

```text
independent_leaky_se1-2_cbam7-8_perf2_seed{42,43,44}_hybrid_leaky_se1-2_cbam7-8_cifar10
csgha_v4_leaky_se1-2_cbam7-8_perf2_seed{42,43,44}_csgha_v4_se1-2_guide2_cbam7-8_cifar10
```

花括号仅表示命名模式，不是已存在的文件。先做不训练的校验：

```bash
/root/miniconda3/bin/python scripts/launch_csgha_v4.py --dry-run
```

给用户的一行正式后台启动命令：

```bash
/root/miniconda3/bin/python scripts/launch_csgha_v4.py
```

启动器调用`run_baselines.py --sweep configs/sweeps/csgha_v4_matched.yaml --jobs 2`。它检查配置/锁/现有进程/所有目标目录；任一目标目录已存在即拒绝覆盖，不自动续训或清理。不要重复启动，也不要用旧全矩阵入口替代。

### 7.2 运行期间

启动器返回实际PID、主日志路径及`tail -f`命令。日志名为`artifacts/launcher_logs/csgha_v4_matched_perf2_<timestamp>.log`；manifest为`artifacts/sweeps/cifar10_csgha_v4_matched_perf2_<timestamp>/manifest.json`。

主日志转发epoch摘要；各子任务完整stdout分别在sweep的`run_logs/`。manifest记录`concurrent_jobs=2`、各run的pid/process_group/log_path/status/return_code/summary。数据worker进程不是额外实验；两个train.py进程是预期行为，**不要再按“只能有一个训练进程”误杀第二组**。

只读查看所有perf2清单（目录不存在时不报错、不写文件）：

```bash
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('artifacts/sweeps').glob('cifar10_csgha_v4_matched_perf2_*/manifest.json')):
    data = json.loads(path.read_text())
    print(path, data['status'])
    for row in data['runs']:
        print(row['experiment_id'], row['status'], row.get('pid'), row.get('return_code'))
PY
```

训练期间不要编辑src/scripts/configs或pull新实现。runner会检查源码hash，检测变化则中止以避免同批混合版本。状态查询仅做相关只读检查；不要擅自启动诊断占用同GPU。

### 7.3 停止与失败处理

只有用户要求停止时才停止。先从当前启动输出、manifest和`ps`核实runner PID，再向**该runner**发送SIGINT或SIGTERM；parallel queue会清理自己创建的子进程组，包括data workers。不要`killall python`或按用户名杀进程，不影响VSCode/Jupyter/TensorBoard。

```text
kill -INT <已核实的本批runner_PID>
```

上面是人工填入已验证PID的操作模板，不是可以原样执行的命令。停止后核对所有所属PID退出、GPU显存释放；不能只看主进程消失。当前两个子任务各有独立进程组，不能假设杀一个旧PGID便能停止全部。

默认任一任务失败会停止仍运行的所属任务，未启动任务保留pending。保留失败日志和产物，检查原因后再决定新编号重跑。`--continue-on-error`不是当前启动器默认策略。

checkpoint当前没有保存完整AMP scaler与CPU/CUDA RNG状态，也没有经验证的精确resume CLI。因此不要把“有epoch_100.pth”当成可无损续训能力；新批次应从头启动。

### 7.4 六组完成的判定

不能只依据GPU空闲或一条“训练结束”日志。至少核对：

1. 最新目标manifest顶层completed，恰好六组、全部completed、return_code=0。
2. 每组config/summary与manifest相符，model_type、seed、perf2、evaluate_test=false均正确。
3. 每组training.csv连续200个epoch，summary的best_validation_accuracy和首次best_epoch与CSV一致。
4. best/latest/final checkpoint存在；best哈希与summary一致；实际train/validation索引45k/5k不相交，同seed两模型划分一致。
5. provenance记录正确架构、Graph/no-cache、线程数、源文件hash；source_snapshot完整，不能将smoke目录混入。
6. 没有test accuracy/predictions被意外生成；benchmark.json是预期skipped，不是错误地把null当0ms。

## 8. 已保留的中断与短测产物

### 8.1 旧v4中断批次

- 清单：`artifacts/sweeps/cifar10_csgha_v4_matched_20260830_204041/manifest.json`。
- 原runner PID/PGID35024，训练PID35070，均已退出。这些是历史标识，**不要对它们直接发信号**。
- 2026-08-30 21:03:13中断；首组ID为`independent_leaky_se1-2_cbam7-8_seed42_hybrid_leaky_se1-2_cbam7-8_cifar10`，最后完整epoch100，best val85.36%。
- 清单顶层interrupted，但旧代码留下首行running；这是历史状态字段未更新，不表示进程活着。新runner已修复这种行状态更新。
- source_snapshot是旧循环性能A/B的冻结参考；不要删除或改写来使新源码“匹配”。

### 8.2 最新perf2验证

- `artifacts/sweeps/smoke_perf2_queue_20260830_214703/manifest.json`：两个模型各3轮，队列成功完成。
- `artifacts/diagnostics/perf2_serial_parallel_equality.json`：两模型相同配置串并行最终state最大差异0、training.csv字节一致。
- `artifacts/diagnostics/perf2_scheduling_receipt.json`、`perf2_scheduling_extra_receipt.json`：并发/worker吞吐与GPU遥测。
- 其他`smoke_*`、`training_perf1_*`、`perf2_*`产物是功能/吞吐验证，不是正式方法结果。

性能预期：两个模型实际并行稳定epoch约9.5–9.9秒，各自同配置串行约5.3–5.5秒；两组同时推进的稳定吞吐约提升11%。两个3轮子进程串行合计约65秒、并行约46秒还包含冷启动重叠，不线性外推为200轮节时。六组粗估1.5–1.8小时，启动后按真实epoch耗时更新。

显存约1.4GB；GPU利用率会因CPU增强、验证和进程初始化波动，不能承诺持续100%。三路并发未显示稳定额外收益，fused AdamW约3%收益且改变数值轨迹，均未作为默认配置。不要仅为占满12GB显存扩大batch。

## 9. 接下来需要做的工作（按依赖推进）

### P0：确认或取得正式perf2六组结果

- 若用户尚未启动：确认无残留任务，给第7节的一行命令；不要自动训练。
- 若正在运行：只读报告真实进度与预计剩余时间；保留日志，不修改正在运行的源码。
- 若已完成：按第7.4节审计后再进入P1。
- 若失败/中断：先诊断，不删除证据、不自动混用已完成/未完成版本。

### P1：扩展可追溯审计，比较v4与matched control

现有`scripts/analysis/audit_experiments.py`主要针对18个旧run，含固定前缀、变体分类和比较列表；还不能直接作为新v4/perf2的完整审计工具。`summarize_results.py`等旧通用脚本也可能仍按旧model_type分类或默认有推理数据。

需要新增或扩展manifest驱动的汇总：

- 精确选出本次六个ID，不收集smoke、epoch100中断或其他执行批次。
- 核对完成性、配置、源码版本、checkpoint hash、split_indices hash。
- 记录每seed best/final validation、best epoch、参数量、执行后端和出处。
- 计算n=3的mean±sample std；按seed配对`v4 − hybrid_leaky`，报告全部差值、均值和胜出次数，不只报最大值。
- 保存新的原始小文件快照到新的`artifacts/audits/<新审计名>/`，精简可追溯结果到新的`reports/`子目录；不覆盖2026-08-30旧审计。
- 把旧v3/middle/shallow作为历史参照单独列出，标注执行后端差异；不把跨后端改善解释为单一激活的因果作用。

### P2：为新v4与matched control做版本匹配checkpoint诊断

现有`audit_csgha_information.py`默认只选旧审计E07–E11；`diagnostics/guidance.py`还把`version == 'v3'`作为有界tanh分支判断。**不能简单把version改为'v4'传入，否则可能错误走无bound路径。**

需要针对新manifest/源码快照实现正确入口，至少：

1. 根据该run架构构建模型、strict load best checkpoint，记录源码和权重hash。
2. 先用同一份完整5k validation、FP32 eager复现原summary的best accuracy；不匹配时先找原因，不继续解释干预。
3. 验证诊断里拆出的deep+guidance计算与实际forward一致，正确保留v4 deep LeakyReLU和v3有界guidance公式。
4. 比较原始、置零guidance、全validation跨样本无自身配对置换、训练集平均描述、训练集平均加项；固定置换种子和样本索引。
5. 对control和v4都统计deep激活、deep logits零值/幅度、guide raw/bounded/gated幅度、门跨样本变化、饱和率；必要时做匹配的deep置零干预。
6. 保存配对预测/logits和索引到artifacts，报告条件差值及适用范围。只从训练子集估计均值，不使用test。

不要把单batch图、guidance/deep的零分母巨大比值或“门不饱和”当成成功证明。来自最终checkpoint的干预不同于从头训练消融，报告中分开。

### P3：根据结果决定是否继续机制优化

- v4对新control三个配对seed一致更好、平均收益有实质扩大，且能正面面对shallow参照：可继续验证，但n=3不自动意味着显著或足以投稿。
- v4改善而control改善相同/更多：没有证明guidance额外贡献；跨后端历史差值也不能证明LeakyReLU因果作用。
- deep活性改善但准确率不改善：不把激活图更漂亮写成任务收益；修正当前假设。
- 仅seed42胜出或三seed仍很小/不稳定：明确结论不足，再提出有限、单因素、有匹配对照的候选，与用户确认后准备脚本。不自动无期限v5/v6。

用户已选择机制路线，不要无说明改回“只研究独立模块位置”。但也不能预写正结果；是否缩小主张应结合证据与用户决定。

### P4：候选可信后补齐论文的最小实验矩阵

原论文大修要求尚未全部完成。后续仍需：

1. CIFAR-100的新方法与匹配独立控制；当前正式CIFAR-100只有baseline，尚无新方法泛化证据。
2. 核心五行消融：baseline、SE-only、CBAM-only、SE+CBAM独立组合、加cross-stage guidance。统一位置/协议，关键三组至少3seeds。
3. 统一recipe的ECA与Coordinate Attention；旧ECA表并不能直接当新协议结果。
4. 原计划推荐两个近期轻量backbone，如MobileOne-S0和RepViT最小可运行版本，必要时MobileViTv2替代。接入前核对官方实现、版本、许可和32×32适配；不要把原论文ImageNet成绩混入CIFAR排名，也不要声称这些是当前所有最新SOTA。
5. 在候选/位置/协议锁定后做一次最终test评估。当前没有已完成的通用“所有best checkpoint最终test批量评估器”；需要验证版本匹配入口，不能为评估test再调用train.py重新训练一遍。
6. 统一独占GPU测量推理latency/throughput、精确参数量和可信FLOPs。现有FLOPs是解析估计（约91M基值+注意力修正），不是完整profiler结果；perf2的null latency不是0ms。

新基线需要同一科学训练配方；不要盲目把只针对固定形状/当前MobileNetV2验证过的CUDA Graph用于任意新架构。执行后端不同要记录和做必要数值验证，影响可比性时重跑直接对照。

### P5：最后再改论文

对应论文目录中的原始handoff：创新性与近期对比是核心拒稿问题。最终必须有可追溯CIFAR-10/100主表、核心消融、位置选择依据和真实机制说明。

待修改要点：公式/命名与最终代码一致；清除摘要编辑残留、重复段落、无关引用/`nocite{*}`；统一validation/test、优化器与lr描述；正面报告不利结果；GPU延迟不冒充移动端实时部署证据；t-SNE/Grad-CAM仅作定性观察。改完LaTeX后重新编译并逐页检查PDF。

范围仍限最小必要大修：不自行扩张到ImageNet、NAS、检测/分割、真实手机部署或五seed大规模搜索。论文目录的编辑需保留原有变更，并遵循它自己的仓库约定。

## 10. 验证与开发工作流

服务器项目根目录：

```bash
/root/miniconda3/bin/python -m pytest -q
/root/miniconda3/bin/python scripts/diagnostics/check_model.py
/root/miniconda3/bin/python scripts/diagnostics/check_data.py
/root/miniconda3/bin/python scripts/launch_csgha_v4.py --dry-run
/root/miniconda3/bin/python -m compileall -q src scripts tests train.py
git diff --check
```

截至交接，75项测试与3个subtests通过，相关新增/修改文件Ruff通过，9种模型/数据集前向检查此前通过；CUDA Graph首次cuBLAS context警告不是训练失败。不要把“此前通过”写成新对话又运行过。

模型改动：跑unit和check_model；训练循环改动：额外用新smoke ID做至少1epoch完整数据检查，evaluate_test=false。优化性能需要同条件对照、热身后计时、明确是否包含数据/验证/保存，不能只看nvidia-smi截图。

本机系统Python可能没有Torch/Ruff；此前完整GPU验证使用服务器环境，不要为跑本地测试擅自重装GPU服务器依赖。

## 11. 产物、备份和GitHub发布边界

每个新run通常包含：

```text
artifacts/runs/<experiment_id>/
├── config.yaml                # 实际解析配置+设备/时间
├── provenance.json            # 架构、代码/依赖/线程、后端、图捕获信息
├── split_indices.json         # 实际train/validation索引
├── metrics.json               # 参数量和估算FLOPs
├── benchmark.json             # perf2显式skipped/null
├── summary.json               # 仅训练成功后写入；最佳val和checkpoint/split hash
├── checkpoints/
│   ├── model_best.pth         # state_dict
│   ├── model_latest.pth       # state_dict
│   ├── epoch_10.pth ...       # 每10epoch，含model/optimizer/scheduler等
│   └── final.pth              # 完成标记之一，不是完整可无损resume契约
├── logs/training.csv
├── logs/tensorboard/
└── predictions/               # validation-only时不应生成官方test导出
```

提交GitHub：源代码、可复用脚本、configs、tests、docs、精简且可追溯的reports、许可证/必要小型样例。历史JSON中的路径/hash是出处记录，不是原数据已上传的证明。

不提交：私钥/密码/token/SSH私有配置、数据集、checkpoints、原始训练日志、预测数组、TensorBoard、虚拟环境、缓存、build、egg-info及整个artifacts。不要用`git add -f`绕过忽略规则。

**GitHub不是实验备份。** 18个历史best checkpoint此前只在服务器计算过hash，权重本体没有作为审计的一部分下载。服务器释放/更换前，需要另行备份重要run目录、source_snapshot和diagnostic原始证据，并校验hash。当前用户只授权GitHub发布必要文件，不等于已安排云盘或权重备份。

本地旧审计原文在`artifacts/audits/2026-08-30/snapshot.json`及`source_files/`；历史guidance诊断原文和配对NPZ也在artifacts。一个干净Git clone只有reports，不能凭空复算这些原始文件；需要从已有本机/服务器备份取回。

离线历史审计复算命令（须已有对应artifacts；默认会重新生成旧报告，先确认不要覆盖）：

```bash
python3 scripts/analysis/audit_experiments.py
python3 scripts/analysis/summarize_guidance_information.py
```

对新实验应使用新的snapshot/output目录并扩展分类/版本逻辑，而不是直接重用旧默认范围。

## 12. 给下一对话的建议开场

> 请先阅读`/root/autodl-tmp/image-classification/AGENTS.md`和`docs/handoff.md`，检查服务器实时进程、最新perf2 manifest、Git状态及六组目标目录。保持新机制路线、validation-only和可追溯协议。若正式六组未开始，给我现有一行启动命令；若已运行，先检查进度；若已完成，先审计六组结果，再扩展版本匹配的v4/control诊断并据结果规划下一步。不要混入smoke或旧epoch100中断结果，不要自动启动额外长实验。

本文件是交接时的状态记录；后续对话应把新的完成状态、明确决策、运行ID和证据入口持续补充到本文或新的日期化报告中，避免再次只靠聊天记忆。

## 13. 2026-08-31 perf2 首批失败与 retry1

首批正式manifest：`artifacts/sweeps/cifar10_csgha_v4_matched_perf2_20260831_101828/manifest.json`，主日志：`artifacts/launcher_logs/csgha_v4_matched_perf2_20260831_101824_756003.log`。保留全部目录，不删除、不覆盖，也不把部分结果写成六组完成。

- `independent` seed42、43完成，best validation分别为87.88%、87.82%。
- `csgha_v4` seed42在epoch71正常摘要后以return code -6（SIGABRT）终止；子日志没有Python traceback或CUDA错误。
- runner随后按fail-fast策略以SIGINT取消正在运行的`independent` seed44；`csgha_v4` seed43、44保持pending。
- 诊断时GPU/磁盘正常，cgroup `memory.failcnt=0`，没有OOM、DataLoader或源码变化证据。容器没有journal、coredumpctl或`/var/crash`，无法从本批进一步区分底层运行时主动abort与外部SIGABRT。

后续训练子进程强制启用：

```text
PYTHONFAULTHANDLER=1
PYTHONUNBUFFERED=1
TORCH_SHOW_CPP_STACKTRACES=1
```

runner同时将负退出码解析为manifest的`termination_signal`，以便后续原生崩溃留下全部Python线程栈、PyTorch C++异常上下文和明确的信号名称。

retry1使用全新六组ID，命名包含`_perf2_retry1_seed{42,43,44}`，仍从头训练、两路并行、validation-only，不复用不完整checkpoint。启动命令：

```bash
/root/miniconda3/bin/python scripts/launch_csgha_v4.py --experiment-tag retry1
```

## 14. 2026-08-31 retry1完成、P1审计与P2入口

retry1 manifest `artifacts/sweeps/cifar10_csgha_v4_matched_perf2_retry1_20260831_112717/manifest.json` 已于12:59完成。六组均completed/return_code=0并通过文件级审计。Control seeds42/43/44为87.88/87.82/88.12%，v4为88.44/87.90/87.90%；配对差值+0.56/+0.08/-0.22个百分点，均值+0.140±0.393，胜出2/3。结果很小且不完全稳定，不能声称guidance稳定提升。

P1正式证据：`artifacts/audits/2026-08-31-csgha-v4-retry1/`；版本化报告：`reports/audits/2026-08-31-csgha-v4-retry1/`。该审计只选择retry1 manifest六组，未混入smoke、首批失败目录或test。

P2已实现独立的v4/control版本匹配入口。dry-run已验证六个best checkpoint strict load，当前模型/数据源码与训练时source snapshot一致；control与v4均做deep统计和deep-zero，v4另做guidance置零、训练均值及三次全validation无自身配对置换。不打开官方test，不训练模型。

P2一行后台启动命令（由用户启动，不要重复运行）：

```bash
/root/miniconda3/bin/python scripts/launch_csgha_v4_diagnostics.py
```

原始P2输出固定为`artifacts/diagnostics/csgha_v4_retry1_information_20260831_v1/`；该任务现已完成，后续结论与v5入口见第15节。

## 15. 2026-08-31 P2完成、v5机制与下一批入口

### 15.1 P2版本匹配诊断结论

P2 manifest为`artifacts/diagnostics/csgha_v4_retry1_information_20260831_v1/manifest.json`，六组均completed，严格加载best checkpoint并在相同5,000张validation上精确复现；未训练、未读取官方test。manifest SHA-256为`abf1a62f4cf5b067c2f1248ebac73fefbbd34789d1414f48d183f774cc7d1e9d`。

版本化结果位于`reports/diagnostics/2026-08-31-csgha-v4-retry1/`：`diagnostic_summary.json`、`results.md`、`findings.md`。主要证据：

- v4三次全validation无自身配对guidance置换平均精度变化分别为−0.9333/−0.4867/−0.2600个百分点，跨seed平均−0.5600；说明最终checkpoint确实使用了与输入配对的浅层信息。
- v4的guidance置零变化为−0.66/+0.02/−0.32个百分点，仍不是三个seed一致受益。
- v4 deep-zero干预为−3.14/−2.74/−3.16个百分点，control为−16.40/−8.84/−6.16；deep logits及LeakyReLU后硬零比例均为0，说明v4已解决v3观察到的hard-dead分支。
- v4 guidance的`tanh(raw)`饱和率约96.24%–97.15%，引导加项绝对均值约0.91–0.99。引导分支几乎退化成符号型信号，是下一步最直接的机制瓶颈。
- v4相对matched control的三seed训练优势仍只有+0.56/+0.08/−0.22个百分点，均值+0.14；不能写成稳定任务收益。

因此下一候选不再调整deep分支，也不增加损失或位置搜索，只处理已被P2直接观察到的guidance输出尺度饱和。

### 15.2 CSGHA v5严格单变量定义

令`z=P_t(LayerNorm(g_s))`，v5在原`tanh`前加入逐样本、跨目标通道的无参数RMS归一化：

```text
z_hat = z / sqrt(mean_c(z^2) + 1e-6)
guidance = tanh(alpha_t) * tanh(z_hat)
channel_gate = sigmoid(deep_leaky + guidance)
```

其余均保持v4不变：SE1–2、Guided-CBAM7–8、24→6→64投影及其内部ReLU、LeakyReLU(0.1) deep分支、alpha零初始化、空间注意力、参数量、loss、优化器、seed划分及perf2执行协议。RMS归一化对投影权重的整体正尺度近似不变，直接阻止仅靠放大投影权重重新进入`tanh`饱和区。RMS本身不是创新点；它只是根据P2证据修正跨阶段引导的信号整形。

代码标识：`model_type=csgha_v5`，`architecture_version=csgha_v5_rms_normalized_guidance_deep_leaky_relu_0.1`。v5新增持久化`guidance_output_normalization_version` buffer；v4/v5严格交叉加载会失败，避免相同参数键被静默解释成不同公式。v4路径与历史初始化未改。

实现与入口：

- 模型：`src/image_classification/models/attention.py`、`mobilenetv2.py`、`factory.py`。
- 配置：`configs/experiments/csgha_v5_rms_middle.yaml`。
- matched sweep：`configs/sweeps/csgha_v5_matched.yaml`。
- 唯一推荐启动器：`scripts/launch_csgha_v5.py`，固定`experiment_tag=rms1`、`jobs=2`。
- 测试：`tests/unit/test_csgha_v5.py`；v5与v4均为2,239,178参数，control仍为2,238,024参数。

### 15.3 验证记录与下一组实验

2026-08-31在服务器Python/Torch环境完成：本次改动文件Ruff通过，`compileall`通过，`check_model.py`的10种配置均前向通过，完整测试为92 passed + 3 subtests passed；唯一警告是既有CUDA集成测试首次建立cuBLAS context。全仓库Ruff仍有22个早先存在、与v5无关的问题，没有借本次工作扩大修改范围。

下一批固定为unchanged `hybrid_leaky` control与`csgha_v5`，各seeds42/43/44，共6次200epoch、validation-only、全部从头训练。六组均带`_perf2_rms1_seed{42,43,44}`新ID，不覆盖retry1或任何失败证据。dry-run已确认配置矩阵和命令，不会复用旧输出。

正式训练前只需再次确认没有训练进程、GPU可用；启动器自身也会检查锁、进程和目标目录。不要在训练运行中修改`src/`、`scripts/`或`configs/`，否则runner的源码指纹保护会中止批次。Codex不自行启动长训练；需要实验时交给用户的一行命令为：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_csgha_v5.py
```

批次完成后先做manifest驱动审计，按seed报告`v5 − hybrid_leaky`全部差值、mean±sample std及胜出数；然后用版本匹配诊断确认RMS确实降低饱和、输入置换效应仍存在，并检查任务收益是否比v4更稳定。未经这两步，不进入CIFAR-100或论文正结果表。

### 15.4 并发原生崩溃与串行入口

`rms1`与`rms3`两批均在两个`hybrid_leaky` control并发、CUDA Graph+AMP运行时发生PyTorch/CUDA原生`SIGABRT`；失败分别出现在不同seed和epoch，v5均尚未启动。另一进程可继续运行，且无OOM、磁盘、PID或非有限checkpoint证据。`rms2`是误用前台runner后由用户中断的短运行。三批证据全部保留，不覆盖、不续训。

为隔离两进程并发这一共同条件，下一批改为`jobs=1`串行，其余科学配置不变；使用新配置`configs/experiments/*_serial.yaml`、新sweep `configs/sweeps/csgha_v5_matched_serial.yaml`和`_serial_s1_seed{42,43,44}`新ID。后台启动命令：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_csgha_v5.py
```

若串行仍出现同类`SIGABRT`，不要继续盲目换ID；下一步应禁用CUDA Graph或启用同步CUDA诊断后再运行。

## 16. 2026-08-31 v5串行完成、机制复核与v6入口

### 16.1 v5串行正式审计

串行s1 manifest为`artifacts/sweeps/cifar10_csgha_v5_rms_matched_serial_s1_20260831_150114/manifest.json`，六组均completed/return_code=0；串行执行未再出现此前双进程CUDA Graph+AMP条件下的随机`SIGABRT`。正式审计仅从该manifest选择六组，不混入并发失败批次、smoke、旧control或test。

证据位于`artifacts/audits/2026-08-31-csgha-v5-serial-s1/`，版本化报告位于`reports/audits/2026-08-31-csgha-v5-serial-s1/`。manifest SHA-256为`6ea44ff9ce47fd8682667419ac279c993cfbe3801bbc977e0ddd396d3100fea2`，审计未发现文件级问题。matched control seeds42/43/44为87.88/87.82/88.12%，v5为87.62/87.18/88.14%；配对差值为−0.26/−0.64/+0.02个百分点，均值−0.293±0.331，胜出1/3。串行control的best checkpoint与v4 retry1对应seed逐文件哈希相同，证明协议和control结果可复现。v5未改善稳定任务收益，不能作为正结果。

### 16.2 v5版本匹配诊断

原始诊断输出为`artifacts/diagnostics/csgha_v5_serial_information_20260831_v1/`，manifest SHA-256为`7ac860f1a56d2ce96c8b9594fb515ff88bf8b3e02edb881598c85c6731f4aee3`；版本化结果位于`reports/diagnostics/2026-08-31-csgha-v5-serial-s1/`。六个best checkpoint均strict load并在相同5,000张validation上精确复现；没有训练，也没有打开官方test。主要结论：

- RMS达到了直接目标：v5各模块/seed的`tanh`饱和率约0.29%–1.76%，显著低于v4约96%–97%。
- 三次无自身配对guidance置换在三个seed均降低精度，跨seed平均为−1.72个百分点（v4为−0.56），说明输入相关浅层信息耦合更强。
- guidance置零变化为−0.84/−0.70/−0.82个百分点，三个seed都表明v5 checkpoint内部使用了guidance。
- v5 deep-zero变化为−1.28/−1.42/−1.60个百分点，平均−1.433；明显弱于v4约−3.013和matched control约−10.467，表明deep分支进一步被弱化。
- guidance contribution绝对均值几乎等于其允许幅度，说明学习到的`abs(tanh(alpha))`接近1。RMS虽修复投影输出饱和，却让guidance长期使用最大允许加性logit幅度；更强的输入相关性没有转化成更高任务精度，且挤占了deep分支。

因此v6不再改归一化、投影、位置、deep激活或训练目标，只限制guidance在channel-gate logit中的最大残差幅度。

### 16.3 CSGHA v6严格单变量定义

v6保持v5的RMS归一化，仅加入固定、无参数的0.25幅度上限：

```text
z_hat = RMSNorm(P_t(LayerNorm(g_s)))
guidance = 0.25 * tanh(alpha_t) * tanh(z_hat)
channel_gate = sigmoid(deep_leaky + guidance)
```

相对v5的唯一机制变量是`guidance_scale_cap=0.25`；其作用是把跨阶段guidance限制为小幅残差修正，避免取代deep分支。SE1–2、Guided-CBAM7–8、投影、RMS、LeakyReLU(0.1)、alpha零初始化、空间注意力、参数量、loss、优化器、数据划分、seed和串行协议均不变。0.25本身不作为创新声明，只是由v5干预证据驱动的信任区间。

代码标识为`model_type=csgha_v6`，`architecture_version=csgha_v6_rms_guidance_cap_0.25_deep_leaky_relu_0.1`。v6新增持久化`guidance_scale_cap_version` buffer，使v5/v6严格交叉加载失败，避免相同参数键被静默解释成不同公式。v4、v5、v6均为2,239,178参数，control为2,238,024参数。

入口文件：实验配置`configs/experiments/csgha_v6_capped_middle_serial.yaml`，matched sweep `configs/sweeps/csgha_v6_matched_serial.yaml`，启动器`scripts/launch_csgha_v6.py`，测试`tests/unit/test_csgha_v6.py`。启动器固定`experiment_tag=v6s1`、`jobs=1`，生成六个全新`_serial_v6s1_seed{42,43,44}` ID，不复用v5或历史control输出。

### 16.4 验证记录与下一组实验

服务器环境已完成本次改动文件Ruff、`compileall`、`git diff --check`；`scripts/diagnostics/check_model.py`的11种配置全部前向通过；完整测试为104 passed + 3 subtests passed。唯一警告仍是CUDA graph集成测试首次建立cuBLAS context。启动器dry-run已确认六组串行命令与全新目录，检查时没有创建v6训练目录，也没有启动训练。

下一批只运行unchanged `hybrid_leaky` control与`csgha_v6`，各seeds42/43/44，共6次200epoch、validation-only、全部从头训练。训练期间不要修改`src/`、`scripts/`或`configs/`。Codex不自行启动长训练；需要实验时交给用户的一行后台启动命令为：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_csgha_v6.py
```

批次完成后先做manifest驱动审计，按seed报告`v6 − hybrid_leaky`全部差值、mean±sample std和胜出数；再运行版本匹配诊断，重点检查guidance contribution是否确实被限制在±0.25、deep-zero效应是否从v5恢复，以及输入置换效应和任务收益能否同时保留。未经审计与诊断，不进入CIFAR-100或论文正结果表。

## 17. 2026-08-31 论文资产清理记录

为保留论文复现、绘图和机制诊断所需资产，同时释放服务器空间，已从32个完成且非v6的run中删除640个`checkpoints/epoch_*.pth`周期性优化器快照，合计约16.3GiB；另删除17个`__pycache__`/pytest/Ruff可再生缓存目录和`data/cifar-100-python/file.txt~`编辑器备份。清理后项目占用约5.0GB，`/root/autodl-tmp`可用约45GB。

所有49个已完成、非当前v6 run的`model_best.pth`、`model_latest.pth`和`final.pth`共147个文件均已核验存在。训练曲线CSV、TensorBoard events、metrics、summary、config、split indices、provenance、sweep manifest、source snapshot、正式审计、版本匹配诊断、报告、图表及此前要求保留的失败/中断证据均未删除。因此论文绘图、best-checkpoint诊断、结果审计和从latest恢复的能力不受影响；正式审计代码本来也不依赖`epoch_*.pth`。

正在运行的v6批次被明确排除，没有删除或修改其任何文件。v6完成并完成manifest审计后，可按同一规则只删除其`epoch_*.pth`，继续保留best/latest/final及全部论文证据；不要在训练进行中清理其目录或修改保存策略。

## 18. 2026-09-01 v6审计、诊断与CSGHA路线停止

### 18.1 v6正式审计

精确manifest为`artifacts/sweeps/cifar10_csgha_v6_capped_matched_serial_v6s1_20260831_173039/manifest.json`，SHA-256为`44b5903955ea8b2aa7a68b1887d726dc0a06c632d85235092f8df4f3ab0866bf`。六组均为completed/return_code=0、连续200 epochs、validation-only，配置、划分、checkpoint和源码快照核验无问题。

版本化审计位于`reports/audits/2026-09-01-csgha-v6-serial-v6s1/`，审计结果SHA-256为`ae1af2caafe54881927facf66e62b833c0c2857a58f94a8fdfef601f3dd1cf2c`。matched control seeds42/43/44为87.88/87.82/88.12%，v6为87.76/87.68/87.92%；配对`v6-control`为−0.12/−0.14/−0.20个百分点，均值−0.153±0.042，胜出0/3。对应control best checkpoint与v5串行control逐seed哈希完全相同。

### 18.2 v6版本匹配诊断

原始输出为`artifacts/diagnostics/csgha_v6_serial_information_20260901_v1/`，manifest SHA-256为`83453bc7b9088f403b25a114d166c4803c48449b983f72908e56f870cfbe0313`；版本化报告位于`reports/diagnostics/2026-09-01-csgha-v6-serial-v6s1/`。六个best checkpoint均strict load，在各自完整5,000张validation上精确复现；没有训练、没有官方test。

- v6 guidance置零变化为−0.22/−0.24/−0.08pp，平均−0.18pp。
- 九次无自身配对置换平均约−0.244pp，明显低于v5的−1.72pp。
- v6 deep-zero为−0.78/−1.16/−2.72pp，平均−1.553pp；与v5的−1.433pp接近，仍远弱于matched control的−10.467pp。
- 六个目标模块的引导贡献绝对均值约为bounded guidance绝对均值的0.25，证明固定cap确实处于最大幅度；投影`tanh`饱和率仍低（约0.34%–1.12%）。

cap实现了设计目标，却同时削弱输入耦合，并未恢复deep分支或任务收益。因此停止“浅层描述直接加到深层channel logits”的CSGHA路线，不做v7，也不再扫描cap、归一化、激活或位置。归档结论位于`reports/negative_results/2026-09-01-csgha-v3-v6/`；允许作为负结果和机制消融，不允许包装为稳定正向方法。

实时核查时`find ... -iname '*v7*'`无输出，`pgrep`也没有v7、train.py或run_baselines.py进程，因此“停止v7”的实际操作是确认其从未启动，无需发送kill信号，也没有删除任何证据。

## 19. 预算感知阶段稀疏选择器与下一批

### 19.1 冻结设计

完整方案见`docs/budget_stage_selector_plan.md`。统一`model_type=stage_sparse`在三个stage packet上独立选择None/ECA/SE/CBAM：shallow=`features[1,2]`、middle=`[7,8]`、deep=`[15,16]`，共`4^3=64`个候选，每阶段最多一种注意力，无跨阶段信息传递。

首批只探测all-none及9个“模块×阶段”单元，使用新开发seeds45/46/47，共10×3=30次。选择器计算每单元相对matched all-none的三seed配对均值与样本标准差，以`sum(mean)-0.5*sqrt(sum(std^2))`作为风险调整的加法搜索代理；然后在参数增量、阶段相关attention运算代理、实测延迟和激活阶段数四重约束下穷举64个候选。all-none始终可选，全部稳健效用为负时不得强制选注意力。多阶段预测只是选择代理，被选架构后续必须用新seeds48/49/50从头确认。

Coordinate Attention暂不进入首轮，避免同时扩大候选模块和验证选择机制；如果确认阶段成功，再按统一协议作为外部候选补入。当前RTX 3080 Ti延迟只是服务器筛选代理，不作为手机部署证据。

### 19.2 已完成准备与硬件档案

实现入口：

- 模型/配置：`src/image_classification/models/mobilenetv2.py`、`config.py`、`factory.py`。
- 枚举、运算代理和选择：`src/image_classification/selection/budget.py`。
- 64候选剖析：`scripts/diagnostics/profile_stage_sparse_candidates.py`。
- 10个实验YAML与三seed sweep：`configs/experiments/budget_probe_*.yaml`、`configs/sweeps/budget_stage_probe.yaml`。
- 默认预算：`configs/budgets/stage_sparse_mobilenetv2.yaml`。
- 唯一首批启动器：`scripts/launch_budget_stage_probe.py`，固定`jobs=1`，默认tag=`probe1`。
- 批次完成后的选择器：`scripts/analysis/select_budget_stage_attention.py`。

正式硬件档案为`artifacts/budget_selector/profiles/stage_sparse_rtx3080ti_paired_cuda_events_20260901_v5/profile.json`，SHA-256为`6f6363b5c933fc109dbd6dd67c1d2d9eaa4485756177b5244685e8dcfdd9069d`。它在空闲RTX3080Ti上对64个未训练候选做3轮随机顺序剖析；每个候选都与同轮、相邻测量的all-none成对归一化，并随机交换测量先后，每次含20次warmup和100次CUDA Event计时。all-none为2,236,682参数，189个配对基线anchor的中位数为3.9576ms；三档预算分别有5/22/58个可行候选。精简报告位于`reports/profiles/2026-09-01-stage-sparse-rtx3080ti/`。

旧v1–v4档案全部保留但禁止用于选择：v1逐次CPU wall-clock同步；v2/v3虽使用CUDA Event，却用不同时间测量的候选/基线独立中位数相除，仍受GPU时钟漂移影响；v4首次采用配对协议，但其源码快照早于最终“效用完全平局优先all-none”契约。v5在看到任何校准准确率前冻结并与最终源码完全匹配。轻量候选三次配对值仍保留系统抖动，若最终候选靠近预算边界，确认前必须增加配对轮数；当前三档可行候选数在v2–v5均保持5/22/58。

服务器验证记录：新增/修改文件Ruff通过，`compileall`通过，`check_model.py`共12种配置前向通过；完整测试为123 passed + 3 subtests passed。唯一warning仍是CUDA Graph集成测试首次创建cuBLAS context。另用30条合成validation-only记录端到端走通“清单验证→九单元统计→三档预算选择→确认计划”dry-run。启动器dry-run精确打印30个全新ID，全部200epoch、validation-only、CUDA Graph、`measure_inference=false`、串行且从头训练；dry-run和剖析均未创建训练目录。

### 19.3 用户下一步只需启动

启动前不要再修改`src/`、`scripts/`或`configs/`；runner会冻结并持续检查源码指纹。启动器还会拒绝已有训练进程、已有目标目录或重复锁。给用户的一行后台启动命令：

```bash
cd /root/autodl-tmp/image-classification && /root/miniconda3/bin/python scripts/launch_budget_stage_probe.py
```

该命令本身创建独立后台session并立即返回PID、日志路径和监控命令。不要额外加`nohup`，不要并发启动第二批，也不要在训练中运行GPU诊断。

完成后先审计精确`cifar10_budget_stage_probe_serial_probe1_<timestamp>/manifest.json`的30组完成性、配置、划分、checkpoint和无test事实，再使用上面的v5 profile运行选择器。选择结果冻结后才准备seeds48/49/50确认清单；未经确认不进入CIFAR-100、第二backbone或最终test。
