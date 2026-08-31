# v5诊断结论与v6机制假设

## 结论

1. RMS达到预期信号效果：六个目标block的guidance tanh饱和率降至约0.29%–1.76%，不再是v4的96%–97%符号门。
2. v5三个seed的guidance置零均下降0.70–0.84个百分点；三次置换在每个训练seed上均下降，跨seed平均−1.72个百分点，输入配对依赖强于v4。
3. 任务结果反而变差：v5对matched control为−0.26/−0.64/+0.02个百分点，配对均值−0.293±0.331。信号更可变、更依赖输入并不自动产生更高准确率。
4. v5 deep-zero仅下降1.28/1.42/1.60个百分点，跨seed平均−1.43；control平均−10.47，v4此前约−3.01。v5 deep logits幅度也明显小于control。
5. v5 contribution与bounded的绝对均值几乎相等，说明可学习scale的`tanh`绝对值接近1；RMS guidance达到最大允许幅度并更容易替代deep分支。

## 下一候选：CSGHA v6 capped residual guidance

只把v5的最大guidance logit修正限制为±0.25：
`z_hat=RMSNorm(P(LN(g_s))); guidance=0.25*tanh(alpha)*tanh(z_hat)`。

该单变量保留v5的RMS、Leaky deep、位置、投影、参数量、损失与训练协议。±0.25形成明确的logit trust region，使跨阶段信息只能作为小幅残差修正，目标是防止再次压低deep分支；0.25本身不是创新主张。
下一批仍需v6与完全相同的hybrid_leaky control按seeds42/43/44串行配对从头训练。checkpoint干预不能替代训练对照。
