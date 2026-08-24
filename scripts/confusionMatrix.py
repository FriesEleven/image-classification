import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. 读取.npz文件中的真实标签和预测标签
data = np.load("../results/predictions/hybrid_se1-2_cbam15-16_hybrid_se1-2_cbam15-16_val_confusion_matrix.npz")
true_labels = data["true_labels"]  # 单独提取，代码更清晰
predictions = data["predictions"]

# 2. 计算混淆矩阵（确保标签是整数类型，避免报错）
cm = confusion_matrix(true_labels.astype(int), predictions.astype(int))

# 3. 绘制混淆矩阵（优化矢量图显示，避免像素化元素）
plt.figure(figsize=(10, 8), dpi=100)  # dpi仅影响预览，不影响SVG精度
# seaborn热图：annot=True显示数值，fmt="d"指定整数格式，cmap选合适配色
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,  # 增加格子线，SVG中会保留矢量线条
    edgecolor="black",  # 格子线颜色，增强清晰度
    annot_kws={"fontsize": 10}  # 调整数值字体大小，避免重叠
)

# 4. 设置图表标签和标题（优化字体，适配SVG显示）
plt.xlabel("Predicted", fontsize=12, fontweight="bold")
plt.ylabel("True", fontsize=12, fontweight="bold")
plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)

# 5. 保存为SVG矢量格式（核心修改）
# 文件名建议去掉.npz后缀，避免误解（SVG文件后缀为.svg）
output_path = "../result/hybrid_se1-2_cbam15-16_confusion_matrix.svg"
plt.savefig(
    output_path,
    format="svg",  # 指定输出格式为SVG
    bbox_inches="tight",  # 去除图表周围多余空白
    dpi=300,  # 仅影响文本等元素的采样精度，SVG本身无像素
    transparent=False  # 背景不透明（若需透明设为True）
)

# 可选：关闭图形释放内存（批量绘图时建议添加）
plt.close()

print(f"SVG矢量图已保存至：{output_path}")