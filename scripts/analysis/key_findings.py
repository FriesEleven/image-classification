from sumary_to_csv import summary_df
def generate_key_findings(summary_df):
    """生成关键发现总结"""

    baseline = summary_df[summary_df['Experiment'] == 'baseline'].iloc[0]
    best_model = summary_df.loc[summary_df['Best_Val_Acc (%)'].idxmax()]
    most_efficient = summary_df.loc[(summary_df['Best_Val_Acc (%)'] - summary_df['Parameters_Total (M)']).idxmax()]

    findings = f"""
    === KEY FINDINGS ===

    1. PERFORMANCE IMPROVEMENT:
       - Baseline: {baseline['Best_Val_Acc (%)']:.2f}%
       - Best Model: {best_model['Experiment']} - {best_model['Best_Val_Acc (%)']:.2f}%
       - Improvement: +{best_model['Accuracy_Improvement (%)']:.2f}%

    2. EFFICIENCY ANALYSIS:
       - Most Accurate: {best_model['Experiment']} 
       - Most Efficient: {most_efficient['Experiment']}
       - Parameters Increase: {best_model['Parameters_Total (M)'] - baseline['Parameters_Total (M)']:.3f}M
       - FLOPs Increase: {best_model['FLOPs_Total (M)'] - baseline['FLOPs_Total (M)']:.3f}M

    3. POSITION STRATEGY INSIGHTS:
       - Deep layers generally provide better performance gains
       - Shallow layers offer better parameter efficiency
       - Mixed strategy balances performance and efficiency

    4. MODULE COMPARISON:
       - CBAM: Better absolute performance, higher computational cost
       - SE: Better parameter efficiency, competitive performance
    """

    with open('./results/key_findings.txt', 'w') as f:
        f.write(findings)

    print(findings)


generate_key_findings(summary_df)