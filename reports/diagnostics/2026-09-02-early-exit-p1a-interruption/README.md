# P1a 用户中断与吞吐诊断

P1a 于 2026-09-02 14:00:21 启动。用户观察到平台 CPU/GPU 利用率约 40% 后要求暂停；Codex 精确定位 runner PID/PGID/SID `62684`，在 baseline seed54 的 epoch125 完成后向 runner 发送 SIGINT。manifest 已写为 `interrupted`，其余五组保持 `pending`；这不是模型失败，也没有任何 completed P1 正式 run。

保留证据：

- Manifest：`artifacts/sweeps/cifar10_early_exit_p1_serial_p1a_20260902_140021/manifest.json`，SHA-256 `c6cd523d0ec7e83be7d2f31cb906f0ab1d1f5ee30ccc7fb237afdd6ad8050cab`。
- Launcher log：`artifacts/launcher_logs/early_exit_p1_serial_p1a_20260902_140017_138584.log`，SHA-256 `f0e2f8afc56ea26a0b1f485ff112812d62e48b2c8bcc6c98b27ccba9ed083e4a`。
- 半程训练 CSV：125 个连续 epoch，SHA-256 `cf66fff07f6291fe9661372f4d4f3aa9ab28e1ba4528c3b673a8de94bea91c9a`。
- seed54 半程目录、best/latest 和 epoch10–120 周期 checkpoint 全部保留；无 summary、无 test 评估，禁止混入 P1b。

旧串行 runner 收到 SIGINT 后成功回写 interrupted manifest，但 `subprocess.run` 没有主动回收 DataLoader workers；主训练进程退出后，Codex按已核实的原 PGID `62684` 对残留 workers 发送 SIGTERM，随后训练进程与 GPU compute process 均为零。P1b 改用独立 child session 并对完整子进程组执行 SIGINT→SIGTERM→SIGKILL 有界回收，manifest 还会记录 child PID/PGID。

吞吐结论见 `throughput_benchmark.json`。当前后台模式已经关闭 tqdm，只在每个 epoch 结束打印一行，日志不是 CPU↔GPU 切换瓶颈。RTX 4090D 上单个 CIFAR-10 32×32 MobileNetV2、batch128 本身就是小负载；P1a 125 轮平均 5.984 秒/epoch，训练没有卡死。

增加 workers 到 12/16 没有提速，且改变了数据增强 RNG 分配，所以仍固定 8 workers。`prefetch_factor=8` 在两个反向顺序配对试验中与原值4得到完全相同的 checkpoint SHA；排除首轮 worker 启动后，六个稳定 epoch 的均值由 6.105 秒降为 5.907 秒（约3.25%），中位数由 6.235 秒降为 5.930 秒（约4.89%）。因此 P1b 只采用这一等价的小幅优化，继续 `jobs=1`，不改变 batch、学习率、优化器、模型、损失或数据划分，也不重引入此前不稳定的并发训练。

所有吞吐 smoke 均为 1/3/5 epoch、validation/calibration-only，不评估官方 test；在记录上表哈希后应删除其可再生 run 目录。P1b 必须使用全新 ID 从头训练，P1a 不续训、不覆盖。
