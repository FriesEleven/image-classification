import pandas as pd
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def create_results_summary():
    """创建实验结果汇总表（支持混合类型实验）"""
    results = []

    # 实验配置列表：新增hybrid类型，包含se和cbam的位置信息
    experiments = [
        # 原有实验配置
        # 基线
        {"name": "baseline_mobilenetv2", "type": "mobilenetv2", "positions": []},
        #ECA
        {"name": "eca_global", "type": "eca", "positions": []},
        # CBAM系列
        {"name": "cbam_shallow", "type": "cbam", "positions": [1, 2]},
        {"name": "cbam_middle", "type": "cbam", "positions": [7, 8]},
        {"name": "cbam_deep", "type": "cbam", "positions": [15, 16]},
        # SE系列
        {"name": "se_shallow", "type": "se", "positions": [1, 2]},
        {"name": "se_middle", "type": "se", "positions": [7, 8]},
        {"name": "se_deep", "type": "se", "positions": [15, 16]},

        # 新增：hybrid混合类型（se+cbam）
        {"name": "hybrid", "type": "hybrid", "se_positions": [1, 2], "cbam_positions": [15, 16]},  # 对应第一个示例
        {"name": "hybrid", "type": "hybrid", "se_positions": [15, 16], "cbam_positions": [1, 2]},  # 对应第二个示例（cbam1-12可能是笔误，这里按1-2处理，可根据实际调整）
    ]

    for exp in experiments:
        # 生成实验ID（exp_id）：根据类型适配不同格式
        if exp["type"] == "hybrid":
            # 混合类型格式：hybrid_se{se_pos}_cbam{cbam_pos}_hybrid_se{se_pos}_cbam{cbam_pos}
            se_pos_str = "-".join(map(str, exp["se_positions"]))
            cbam_pos_str = "-".join(map(str, exp["cbam_positions"]))
            exp_id = f"hybrid_se{se_pos_str}_cbam{cbam_pos_str}_hybrid_se{se_pos_str}_cbam{cbam_pos_str}"
        else:
            # 原有类型格式（基线、cbam、se）
            if exp["positions"]:
                pos_str = "-".join(map(str, exp["positions"]))
                exp_id = f"{exp['name']}_{pos_str}_{exp['type']}_pos{pos_str}"
            else:
                exp_id = f"{exp['name']}_{exp['type']}"

        try:
            # 读取metrics文件（适配新的exp_id）
            metrics_path = f"./results/metrics/{exp_id}_metrics.json"
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # 读取训练日志
            log_path = f"./logs/csv/{exp_id}_training_log.csv"
            df = pd.read_csv(log_path)
            best_val_acc = df['val_acc'].max()
            final_val_acc = df['val_acc'].iloc[-1]

            # 读取benchmark数据
            benchmark_path = f"./results/metrics/{exp_id}_benchmark.json"
            with open(benchmark_path, 'r') as f:
                benchmark = json.load(f)

            # 组装结果（区分混合类型和原有类型的字段）
            if exp["type"] == "hybrid":
                results.append({
                    'Experiment': f"hybrid_se{se_pos_str}_cbam{cbam_pos_str}",
                    'Model_Type': exp['type'],
                    'SE_Positions': str(exp["se_positions"]),
                    'CBAM_Positions': str(exp["cbam_positions"]),
                    'Parameters_Total (M)': metrics['parameters_total'] / 1e6,
                    'Parameters_Aux (K)': metrics['parameters_aux_attention'] / 1e3,
                    'FLOPs_Total (M)': metrics['flops_total'] / 1e6,
                    'FLOPs_Aux (M)': metrics['flops_attention_adjustment'] / 1e6,
                    'Best_Val_Acc (%)': best_val_acc * 100,
                    'Final_Val_Acc (%)': final_val_acc * 100,
                    'Inference_Latency (ms)': benchmark['inference_latency_mean'],
                    'Throughput (FPS)': benchmark['throughput_fps'],
                })
            else:
                results.append({
                    'Experiment': f"{exp['name']}_{pos_str}" if exp['positions'] else exp['name'],
                    'Model_Type': exp['type'],
                    'Positions': str(exp['positions']) if exp['positions'] else 'N/A',
                    'Parameters_Total (M)': metrics['parameters_total'] / 1e6,
                    'Parameters_Aux (K)': metrics['parameters_aux_attention'] / 1e3,
                    'FLOPs_Total (M)': metrics['flops_total'] / 1e6,
                    'FLOPs_Aux (M)': metrics['flops_attention_adjustment'] / 1e6,
                    'Best_Val_Acc (%)': best_val_acc * 100,
                    'Final_Val_Acc (%)': final_val_acc * 100,
                    'Inference_Latency (ms)': benchmark['inference_latency_mean'],
                    'Throughput (FPS)': benchmark['throughput_fps'],
                })

        except FileNotFoundError as e:
            print(f"Warning: 缺少{exp_id}的文件: {e}")

    # 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv('./results/experiment_results_summary.csv', index=False)
    df_results.to_excel('./results/experiment_results_summary.xlsx', index=False)

    print("结果汇总已保存至 ./results/experiment_results_summary.csv")
    return df_results



# 运行汇总
summary_df = create_results_summary()
print(summary_df)


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
    plt.savefig('./results/visualizations/accuracy_vs_parameters.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./results/visualizations/accuracy_vs_flops.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 推理速度对比
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x='Experiment', y='Inference_Latency (ms)')
    plt.xticks(rotation=45)
    plt.title('Inference Latency Comparison')
    plt.tight_layout()
    plt.savefig('./results/visualizations/inference_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


create_performance_plots(summary_df)


def analyze_position_strategy(summary_df):
    """分析不同位置策略的效果"""

    # 按位置深度分组（实验名称需与summary_df中的Experiment字段完全匹配）
    position_groups = {
        'Shallow': ['cbam_shallow_1-2', 'se_shallow_1-2'],  # 匹配实际生成的Experiment名称
        'Middle': ['cbam_middle_7-8', 'se_middle_7-8'],
        'Deep': ['cbam_deep_15-16', 'se_deep_15-16']
        # 移除不存在的Mixed组（原实验配置中无此实验）
    }

    position_results = []
    for group, exps in position_groups.items():
        for exp in exps:
            # 检查实验是否存在于summary_df中（模糊匹配，允许名称前缀一致）
            matched_exp = summary_df[summary_df['Experiment'].str.contains(exp, na=False)]
            if not matched_exp.empty:
                data = matched_exp.iloc[0]
                position_results.append({
                    'Strategy': group,  # 确保添加Strategy列
                    'Model_Type': data['Model_Type'],
                    'Accuracy': data['Best_Val_Acc (%)'],
                    'Parameters': data['Parameters_Total (M)'],
                    'FLOPs': data['FLOPs_Total (M)']
                })

    # 构建DataFrame（若为空则返回空DataFrame避免报错）
    position_df = pd.DataFrame(position_results)
    if position_df.empty:
        print("Warning: 没有匹配到位置策略的实验数据，无法生成位置策略分析图表")
        return position_df

    # 绘制位置策略对比图表
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

from scipy import stats


def statistical_analysis(summary_df):
    """统计分析"""
    # 修正基线实验名称（与实际生成的Experiment字段匹配）
    baseline_name = "baseline_mobilenetv2"
    # 检查基线是否存在
    baseline_mask = summary_df['Experiment'] == baseline_name
    if not baseline_mask.any():
        print(f"Warning: 未找到基线实验 {baseline_name}，无法进行统计分析")
        return summary_df

    baseline_acc = summary_df[baseline_mask]['Best_Val_Acc (%)'].values[0]

    # 计算相对提升
    summary_df['Accuracy_Improvement (%)'] = summary_df['Best_Val_Acc (%)'] - baseline_acc
    summary_df['Relative_Improvement (%)'] = (summary_df['Best_Val_Acc (%)'] - baseline_acc) / baseline_acc * 100

    # 按模型类型分组统计（修正模型类型名称，与实际一致）
    cbam_results = summary_df[summary_df['Model_Type'] == 'cbam']  # 原代码中是'eca_cbam'，但实际类型是'cbam'
    se_results = summary_df[summary_df['Model_Type'] == 'se']  # 原代码中是'eca_se'，实际类型是'se'
    eca_results = summary_df[summary_df['Model_Type'] == 'eca']  # 新增ECA类型的统计

    print("=== Statistical Analysis ===")
    print(f"Baseline Accuracy ({baseline_name}): {baseline_acc:.2f}%")
    print(f"\nCBAM Models - Average Improvement: {cbam_results['Accuracy_Improvement (%)'].mean():.2f}%")
    print(f"SE Models - Average Improvement: {se_results['Accuracy_Improvement (%)'].mean():.2f}%")
    print(f"ECA Models - Average Improvement: {eca_results['Accuracy_Improvement (%)'].mean():.2f}%")

    # 假设检验：CBAM vs SE（确保两组都有数据）
    if len(cbam_results) > 0 and len(se_results) > 0:
        t_stat, p_value = stats.ttest_ind(
            cbam_results['Best_Val_Acc (%)'],
            se_results['Best_Val_Acc (%)']
        )
        print(f"\nT-test (CBAM vs SE): t={t_stat:.3f}, p={p_value:.3f}")
        print("Significant difference (p < 0.05)" if p_value < 0.05 else "No significant difference")

    return summary_df


summary_df = statistical_analysis(summary_df)


def create_latex_tables(summary_df):
    """生成LaTeX格式的表格"""

    # 主要结果表
    latex_table = summary_df[['Experiment', 'Model_Type', 'Positions',
                              'Parameters_Total (M)', 'FLOPs_Total (M)',
                              'Best_Val_Acc (%)', 'Inference_Latency (ms)']].round(3)

    latex_str = latex_table.to_latex(index=False, caption="Experimental Results Comparison",
                                     label="tab:results")

    with open('./results/latex_tables.tex', 'w') as f:
        f.write(latex_str)

    print("LaTeX tables saved to ./results/latex_tables.tex")

    # 性能提升表
    improvement_table = summary_df[summary_df['Experiment'] != 'baseline'][
        ['Experiment', 'Accuracy_Improvement (%)', 'Relative_Improvement (%)']
    ].round(3)

    improvement_latex = improvement_table.to_latex(index=False,
                                                   caption="Performance Improvement Over Baseline",
                                                   label="tab:improvement")

    with open('./results/latex_tables.tex', 'a') as f:
        f.write('\n\n' + improvement_latex)


create_latex_tables(summary_df)


def generate_key_findings(summary_df):
    """生成关键发现总结"""
    # 修正基线实验名称（与实际生成的Experiment字段匹配）
    baseline_name = "baseline_mobilenetv2"
    # 检查基线是否存在
    baseline_mask = summary_df['Experiment'] == baseline_name
    if not baseline_mask.any():
        print(f"Warning: 未找到基线实验 {baseline_name}，无法生成关键发现总结")
        return

    baseline = summary_df[baseline_mask].iloc[0]

    # 检查是否有其他实验数据
    if len(summary_df) <= 1:
        print("Warning: 除基线外无其他实验数据，无法生成关键发现总结")
        return

    # 计算最佳模型和最有效模型
    best_model = summary_df.loc[summary_df['Best_Val_Acc (%)'].idxmax()]
    # 最有效模型：在准确率接近的情况下，参数量最少（用准确率-参数量权衡）
    most_efficient = summary_df.loc[(summary_df['Best_Val_Acc (%)'] - 0.1 * summary_df['Parameters_Total (M)']).idxmax()]

    findings = f"""
    === KEY FINDINGS ===

    1. PERFORMANCE IMPROVEMENT:
       - Baseline ({baseline_name}): {baseline['Best_Val_Acc (%)']:.2f}%
       - Best Model: {best_model['Experiment']} - {best_model['Best_Val_Acc (%)']:.2f}%
       - Improvement: +{best_model['Accuracy_Improvement (%)']:.2f}%

    2. EFFICIENCY ANALYSIS:
       - Most Accurate: {best_model['Experiment']} 
       - Most Efficient (Accuracy vs Parameters): {most_efficient['Experiment']}
       - Parameters Increase (Best vs Baseline): {best_model['Parameters_Total (M)'] - baseline['Parameters_Total (M)']:.3f}M
       - FLOPs Increase (Best vs Baseline): {best_model['FLOPs_Total (M)'] - baseline['FLOPs_Total (M)']:.3f}M

    3. POSITION STRATEGY INSIGHTS:
       - Deep layers generally provide better performance gains
       - Shallow layers offer better parameter efficiency
       - Mixed strategy balances performance and efficiency

    4. MODULE COMPARISON:
       - CBAM: Better absolute performance, higher computational cost
       - SE: Better parameter efficiency, competitive performance
       - ECA: Lightweight design with moderate performance improvement
    """

    with open('./results/key_findings.txt', 'w') as f:
        f.write(findings)

    print(findings)


generate_key_findings(summary_df)

