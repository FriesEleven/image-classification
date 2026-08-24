from scipy import stats
from sumary_to_csv import summary_df

def statistical_analysis(summary_df):
    """统计分析"""

    baseline_acc = summary_df[summary_df['Experiment'] == 'baseline']['Best_Val_Acc (%)'].values[0]

    # 计算相对提升
    summary_df['Accuracy_Improvement (%)'] = summary_df['Best_Val_Acc (%)'] - baseline_acc
    summary_df['Relative_Improvement (%)'] = (summary_df['Best_Val_Acc (%)'] - baseline_acc) / baseline_acc * 100

    # 按模型类型分组统计
    cbam_results = summary_df[summary_df['Model_Type'] == 'eca_cbam']
    se_results = summary_df[summary_df['Model_Type'] == 'eca_se']

    print("=== Statistical Analysis ===")
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"\nCBAM Models - Average Improvement: {cbam_results['Accuracy_Improvement (%)'].mean():.2f}%")
    print(f"SE Models - Average Improvement: {se_results['Accuracy_Improvement (%)'].mean():.2f}%")

    # 假设检验：CBAM vs SE
    if len(cbam_results) > 0 and len(se_results) > 0:
        t_stat, p_value = stats.ttest_ind(cbam_results['Best_Val_Acc (%)'],
                                          se_results['Best_Val_Acc (%)'])
        print(f"\nT-test (CBAM vs SE): t={t_stat:.3f}, p={p_value:.3f}")

        if p_value < 0.05:
            print("Significant difference between CBAM and SE models (p < 0.05)")
        else:
            print("No significant difference between CBAM and SE models")

    return summary_df


summary_df = statistical_analysis(summary_df)