# P2结论与下一机制假设

## 结论

1. v4三个seed在置换正确配对的guidance后均下降，支持模型使用输入相关浅层信息。
2. LeakyReLU使v4与control的deep分支在完整validation上不再硬失活；v4 deep-zero均下降约3个百分点。
3. 任务收益仍不稳定：P1中v4对control仅+0.140±0.393个百分点，seed44为负。机制在工作不等于稳定优于control。
4. v4两个目标block的guidance tanh饱和率仍约96%–97%，投影幅度几乎被压成符号门，是当前最明确的剩余单因素问题。

## 下一候选：CSGHA v5 RMS-normalized guidance

仅在guide projection输出进入tanh前做逐样本、跨通道无参数RMS归一化：
`z=P(LN(g_s)); z_hat=z/sqrt(mean_c(z^2)+eps); guidance=tanh(alpha)*tanh(z_hat)`。

该修改保持Leaky deep、位置、投影结构、损失和训练协议不变。它直接限制投影尺度，避免网络只靠放大权重重新进入tanh饱和；不把RMS归一化本身包装为创新。
下一批仍需v5与完全相同的hybrid_leaky control按seeds42/43/44配对从头训练。P2结果不能替代该训练对照。
