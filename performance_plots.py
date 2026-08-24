import matplotlib.pyplot as plt
import seaborn as sns

from sumary_to_csv import summary_df

def create_performance_plots(summary_df):
    """创建性能对比图表"""
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12

    # 1. 准确率 vs 参数量散点图
    plt.figure(figsize=(10, 6))
    for model_type in summary_df['Model_Type'].unique():
        data = summary_df[summary_df['Model_Type'] == model_type]
        plt.scatter(data['Parameters_Total (M)'], data['Best_Val_Acc (%)'],
                    label=model_type, s=100, alpha=0.7)

    plt.xlabel('Total Parameters (M)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Accuracy vs Model Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/visualizations/accuracy_vs_parameters.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 准确率 vs FLOPs散点图
    plt.figure(figsize=(10, 6))
    for model_type in summary_df['Model_Type'].unique():
        data = summary_df[summary_df['Model_Type'] == model_type]
        plt.scatter(data['FLOPs_Total (M)'], data['Best_Val_Acc (%)'],
                    label=model_type, s=100, alpha=0.7)

    plt.xlabel('Total FLOPs (M)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Accuracy vs Computational Cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/visualizations/accuracy_vs_flops.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 推理速度对比
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x='Experiment', y='Inference_Latency (ms)')
    plt.xticks(rotation=45)
    plt.title('Inference Latency Comparison')
    plt.tight_layout()
    plt.savefig('./results/visualizations/inference_latency_comparison.svg', dpi=300, bbox_inches='tight')
    plt.close()


create_performance_plots(summary_df)

