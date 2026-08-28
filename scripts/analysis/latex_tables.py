from sumary_to_csv import summary_df
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