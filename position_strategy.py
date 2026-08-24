import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sumary_to_csv import summary_df


def analyze_position_strategy(summary_df):
    """分析不同位置策略的效果"""

    # 按位置深度分组
    position_groups = {
        'Shallow': ['cbam_shallow', 'se_shallow'],
        'Middle': ['cbam_middle', 'se_middle'],
        'Deep': ['cbam_deep', 'se_deep'],
        'Mixed': ['cbam_mixed', 'se_mixed']
    }

    position_results = []
    for group, exps in position_groups.items():
        for exp in exps:
            if exp in summary_df['Experiment'].values:
                data = summary_df[summary_df['Experiment'] == exp].iloc[0]
                position_results.append({
                    'Strategy': group,
                    'Model_Type': data['Model_Type'],
                    'Accuracy': data['Best_Val_Acc (%)'],
                    'Parameters': data['Parameters_Total (M)'],
                    'FLOPs': data['FLOPs_Total (M)']
                })

    position_df = pd.DataFrame(position_results)

    # 绘制位置策略对比
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.barplot(data=position_df, x='Strategy', y='Accuracy', hue='Model_Type')
    plt.title('Accuracy by Position Strategy')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.barplot(data=position_df, x='Strategy', y='Parameters', hue='Model_Type')
    plt.title('Parameters by Position Strategy')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    sns.barplot(data=position_df, x='Strategy', y='FLOPs', hue='Model_Type')
    plt.title('FLOPs by Position Strategy')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('./results/visualizations/position_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return position_df


position_analysis = analyze_position_strategy(summary_df)