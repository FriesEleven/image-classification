import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 加载保存的用于绘制 ROC 曲线的数据
data = np.load(
    REPO_ROOT / "results/predictions/eca_global_eca_val_roc_data.npz",
    allow_pickle=True,
)
true_labels = data["true_labels"]
probabilities = data["probabilities"]

# 类别数量，假设 CIFAR - 10 为 10 类
num_classes = 10

# 对真实标签进行二值化处理
true_labels_bin = label_binarize(true_labels, classes=list(range(num_classes)))

# 初始化存储每个类别的 FPR、TPR 和 AUC 的字典
fpr = dict()
tpr = dict()
roc_auc = dict()

# 计算每个类别的 ROC 曲线和 AUC
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制每个类别的 ROC 曲线
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# 绘制随机猜测的对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc="lower right")
plt.grid(True)

# 保存绘制好的 ROC 曲线图
plt.savefig(REPO_ROOT / 'results/visualizations/roc_curve.png')
plt.show()
