# P2 后一次性 CIFAR-10.1 v6 外部分布验证

## 1. 授权依据与唯一目的

P2a 的正式审计 `issues={}`，冻结迁移分析状态为 `ready_for_external_shift_test`。P1b 在
source seeds54/55/56 上选出的 exit8 最大 softmax 阈值 `0.984`，已经原样通过 target
seeds57/58/59 的全部预注册 gate；P2 没有考虑任何新阈值。

本次外部验证只回答：这个冻结策略在独立采集的 CIFAR-10.1 v6 分布上，是否仍能跨 source 和
target 六个模型版本保持经验风险不退化与动态计算收益。不得据外部结果改阈值、加类别保护、选
checkpoint 或剔除 seed。

## 2. 冻结输入

- P1b source selection SHA-256：
  `8dce5d938e06ae9d7432bd6baafd0c0ed9f978678fb53767ab02eeafecc723ab`
- P1b manifest SHA-256：
  `1545e85651103400c425bf3eaae0e70e4d1a4d0e413203158d2f6b6ff16c8c88`
- P2a manifest SHA-256：
  `c262be6b7ab6c2d84e0a14247504323833b966a0d314fe8bc0876267dbd50531`
- P2a audit SHA-256：
  `376d08415ff892bd041efa1c3e12e6e7678ef2208f33045ea2bdf1bcfd7e8bab`
- P2a transfer result SHA-256：
  `ef86f749fba03255b5d31e9ae8ff7d7e4e1b78d3fc12bdf6ea1bec7a5eb79706`
- 模型：matched baseline 与 multi-exit，source seeds54/55/56、target seeds57/58/59，共六对
  best checkpoints；所有 checkpoint 哈希分别来自冻结 selection 或 P2 transfer result。
- 策略：只允许 exit8→final，阈值精确为 `0.984`，无类别保护，外部候选阈值数为 0。

## 3. 外部数据与预处理

数据固定为官方 [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1) 仓库 commit
`d9982abb0bfc4846b8d13a11e66b887d946205d0` 的 v6：

- `cifar10.1_v6_data.npy`：6,144,128 bytes，SHA-256
  `2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6`；
- `cifar10.1_v6_labels.npy`：8,128 bytes，SHA-256
  `ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448`。

文件预检只读取了哈希、数组形状/类型、像素范围和标签计数，没有运行模型。数据为2,000张
32×32 RGB uint8图片，10类各200张。预处理固定为除以255后使用项目现有 CIFAR-10 mean/std
归一化，不做增强；顺序固定、batch128、`shuffle=false`、`num_workers=0`。加载器不会构造或
迭代原始 CIFAR-10 test。

## 4. 一次性执行与冻结 gate

`scripts/analysis/evaluate_early_exit_cifar10_1_v6.py` 在任何模型推理前校验上述文件、manifest、
audit、transfer、selection和12个checkpoint哈希，并要求源码提交及工作区干净。随后先以独占
方式创建 started access marker；marker、原始输出或版本化报告任一存在时，均拒绝再次执行。

六个 seed 全部必须满足：

1. 策略总体 accuracy drop 相对各自最终头 ≤ 0；
2. balanced accuracy drop ≤ 0；
3. worst-class accuracy drop ≤ 0；
4. 早退率位于 15%–95%；
5. MAC 代理节省 ≥ 15%。

全部通过才输出 `external_shift_confirmed`；任一失败输出 `stop_external_shift_failure`，归档并停止
当前“无重校准迁移”主张。零下降仍只是每类200张样本上的经验事实，不是统计零风险保证。

评估保存六个 seed 的 baseline/final/exit8/exit16 logits、冻结策略预测和路径到 ignored artifacts，
版本化 JSON、逐 seed/逐类结果、输入哈希及 access marker 索引。原始 CIFAR-10 test 永不重开。

## 5. 结果后的决策

- 若失败：不调外部阈值，不追加 seed 掩盖结果，停止本方向并保留负证据。
- 若通过：跨模型版本与自然分布转移信号成立，但架构证据仍只有 MobileNetV2；冻结结果后，再决定
  一组最小的第二数据集或第二 backbone 确认实验。未经这一步，不宣布整篇论文证据完成。
