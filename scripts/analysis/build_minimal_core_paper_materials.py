"""Audit saved minimal-core arrays and create manuscript tables without model execution."""
import csv
import sys
import shutil
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from scripts.analysis.paper_readiness import read_json,save,sha,write_csv
from scripts.analysis.minimal_core_evidence import OUT,CASES,costs,load_caches,subset,scores,metrics,risk,bootstrap_pair

DEST=ROOT/'reports/paper/final-20260918'


def rows(path):
    with path.open() as handle:return list(csv.DictReader(handle))


def main():
    if DEST.exists():raise FileExistsError('Preserve existing manuscript material output')
    expected=read_json(OUT/'output_hashes.json')
    for name,checksum in expected.items():
        if sha(OUT/name)!=checksum:raise ValueError(f'Output checksum mismatch: {name}')
    # Recompute M1 from its frozen threshold and ID partitions only; never search again.
    cohorts,_=load_caches();partitions=read_json(OUT/'fit_audit_partitions.json')
    m1=rows(OUT/'shared_vs_individual_seed_metrics.csv')
    for row in m1:
        _,source,target,worst=next(case for case in CASES if case[0]==row['case'])
        cohort=source if row['split']=='source_F' else target
        pool='fit_ids' if row['split']=='source_F' else 'audit_ids'
        record=next(r for r in cohorts[cohort] if r['seed']==int(row['seed']))
        record=subset(record,partitions[cohort][pool])
        if row['selection_status']!='selected':continue
        value=metrics(record,scores(record)>=float(row['threshold']))
        for key in ('accuracy','accuracy_drop','balanced_accuracy_drop','worst_class_accuracy_drop','cost_saving_fraction','early_fraction'):
            if abs(value[key]-float(row[key]))>1e-12:raise ValueError('M1 saved-point recalculation mismatch')
        if str(risk(value,worst))!=row['risk_feasible']:raise ValueError('M1 risk flag mismatch')
    joint=rows(OUT/'a3_joint_operating_points.csv');rounds=rows(OUT/'a3_latency_round_metrics.csv')
    m2_saved={};tail_seed=[]
    for row in joint:
        dataset,seed=row['dataset'],int(row['seed'])
        with np.load(OUT/'m2_raw'/f'{dataset}_s{seed}_predictions.npz') as data:
            early,final,head=costs(dataset)
            value=metrics(dict(labels=data['labels'],exit8_logits=data['exit8_logits'],final_logits=data['final_logits'],exit_cost=early/final,head_cost=head/final),data['early'])
            for key in ('accuracy','reference_accuracy','accuracy_drop','balanced_accuracy_drop','worst_class_accuracy_drop','cost_saving_fraction','early_fraction'):
                if abs(value[key]-float(row[key]))>1e-12:raise ValueError('M2 saved-point recalculation mismatch')
            pred=np.where(data['early'],data['exit8_logits'].argmax(1),data['final_logits'].argmax(1))
            if not np.array_equal(pred,data['policy_predictions']):raise ValueError('M2 policy prediction mismatch')
            m2_saved[(dataset,seed)]=(data['labels'].copy(),(pred==data['labels']).astype(float)-(data['final_logits'].argmax(1)==data['labels']).astype(float))
        group=[r for r in rounds if r['dataset']==dataset and int(r['seed'])==seed]
        p95s=[]
        for rnd in range(5):
            pair={r['mode']:r for r in group if int(r['round'])==rnd}
            for mode,r in pair.items():
                with np.load(OUT/r['raw_path']) as d:
                    if abs(float(np.quantile(d['latency_ms'],.95))-float(r['latency_p95_ms']))>1e-9:raise ValueError('Raw p95 mismatch')
                    if abs(1000/float(d['latency_ms'].mean())-float(r['throughput_samples_per_second']))>1e-9:raise ValueError('Raw throughput mismatch')
            p95s.append(1-float(pair['actual_dynamic']['latency_p95_ms'])/float(pair['final_only']['latency_p95_ms']))
        tail_seed.append(dict(dataset=dataset,seed=seed,paired_p95_saving_percent=100*float(np.mean(p95s))))
    compact_m1=[]
    for row in rows(OUT/'shared_vs_individual_summary.csv'):
        value={k:row[k] for k in ('case','split','strategy','n','all_risk_feasible','all_saving_floor_pass','threshold_parameters','target_recalibrations')}
        for key in ('accuracy','cost_saving_fraction','early_fraction','accuracy_drop'):
            for suffix in ('mean','sample_sd'):
                value[key+'_'+suffix+'_percent_or_pp']=100*float(row[key+'_'+suffix])
        value['worst_class_drop_max_pp']=100*float(row['worst_class_accuracy_drop_max'])
        compact_m1.append(value)
    compact_seed=[]
    for row,tail in zip(joint,tail_seed):
        value={k:row[k] for k in ('dataset','seed','threshold','checkpoint_sha256','risk_feasible','saving_floor_pass')}
        for old,new in [('reference_accuracy','final_accuracy_percent'),('accuracy','policy_accuracy_percent'),('accuracy_drop','policy_drop_pp'),
            ('early_fraction','full_test_early_percent'),('cost_saving_fraction','full_test_mac_saving_percent'),
            ('worst_class_accuracy_drop','worst_class_drop_pp'),('seed_latency_saving_fraction','paired_mean_latency_saving_percent'),
            ('timed_subset_early_fraction','timed_subset_early_percent'),('timed_subset_mac_saving_fraction','timed_subset_mac_saving_percent')]:
            value[new]=100*float(row[old])
        for key in ('final_only_latency_mean_ms','actual_dynamic_latency_mean_ms','final_only_latency_p95_ms','actual_dynamic_latency_p95_ms',
            'final_only_throughput_samples_per_second','actual_dynamic_throughput_samples_per_second'):value[key]=float(row[key])
        value['paired_p95_saving_percent']=tail['paired_p95_saving_percent']
        compact_seed.append(value)
    compact_summary=[]
    for dataset in ('cifar10','cifar100'):
        group=[r for r in compact_seed if r['dataset']==dataset]
        value=dict(dataset=dataset,n=3,all_risk_feasible=all(r['risk_feasible']=='True' for r in group),all_saving_floor_pass=all(r['saving_floor_pass']=='True' for r in group))
        for key in group[0]:
            if key not in ('seed','threshold') and isinstance(group[0][key],float):
                vals=[r[key] for r in group];value[key+'_mean']=float(np.mean(vals));value[key+'_sample_sd']=float(np.std(vals,ddof=1))
        labels,diff=m2_saved[(dataset,int(group[0]['seed']))]
        diffs=[]
        for r in group:
            other_labels,other_diff=m2_saved[(dataset,int(r['seed']))]
            if not np.array_equal(other_labels,labels):raise ValueError('M2 bootstrap image pairing mismatch')
            diffs.append(other_diff)
        boot=bootstrap_pair(labels,diffs)
        value.update({k+'_pp':100*v for k,v in boot.items()})
        value['bootstrap_scope']='2000 stratified paired image draws, conditional on three fixed models; policy minus own final'
        compact_summary.append(value)
    DEST.mkdir(parents=True,exist_ok=False)
    write_csv(DEST/'shared_policy_manuscript.csv',compact_m1)
    write_csv(DEST/'a3_joint_seed_manuscript.csv',compact_seed)
    write_csv(DEST/'a3_joint_summary_manuscript.csv',compact_summary)
    # Two panels from M1; each reported metric retains its original fit/audit scope.
    tex=['% Post-hoc A4 fit/audit comparison. SD uses three training seeds.',
        '\\begin{tabular}{lllrrrr}', '\\hline','Cohort & Scope & Strategy & Accuracy (\\%) & MAC saved (\\%) & Worst drop (pp) & Gates \\\\', '\\hline']
    for r in compact_m1:
        gates='pass' if r['all_risk_feasible']=='True' and r['all_saving_floor_pass']=='True' else 'fail'
        name={'cifar10':'C10','cifar100_strict':'C100 strict','cifar100_relaxed':'C100 relaxed'}[r['case']]
        scope=r['split'].replace('_','-')
        tex.append(f"{name} & {scope} & {r['strategy']} & {r['accuracy_mean_percent_or_pp']:.2f} $\\pm$ {r['accuracy_sample_sd_percent_or_pp']:.2f} & {r['cost_saving_fraction_mean_percent_or_pp']:.2f} $\\pm$ {r['cost_saving_fraction_sample_sd_percent_or_pp']:.2f} & {r['worst_class_drop_max_pp']:.2f} & {gates} \\\\")
    tex+=['\\hline','\\end{tabular}']
    (DEST/'shared_policy_manuscript.tex').write_text('\n'.join(tex)+'\n')
    tex=['% Accuracy/MAC: all 10000 official images; timing: 1000 resident singleton images from same benchmark.',
         '\\begin{tabular}{lrrrrrr}','\\hline','Dataset & Final (\\%) & Policy (\\%) & MAC saved (\\%) & Mean saved (\\%) & p95 saved (\\%) & Savings gates \\\\', '\\hline']
    for r in compact_summary:
        tex.append(f"{r['dataset']} & {r['final_accuracy_percent_mean']:.2f} & {r['policy_accuracy_percent_mean']:.2f} & {r['full_test_mac_saving_percent_mean']:.2f} & {r['paired_mean_latency_saving_percent_mean']:.2f} $\\pm$ {r['paired_mean_latency_saving_percent_sample_sd']:.2f} & {r['paired_p95_saving_percent_mean']:.2f} & {'pass' if r['all_saving_floor_pass'] else 'fail'} \\\\")
    tex+=['\\hline','\\end{tabular}']
    (DEST/'a3_joint_manuscript.tex').write_text('\n'.join(tex)+'\n')
    shutil.copy2(OUT/'shared_policy_tradeoffs.pdf',DEST/'shared_policy_tradeoffs.pdf')
    input_names=['output_hashes.json','protocol_manifest.json','input_manifest.json','fit_audit_partitions.json','shared_vs_individual_seed_metrics.csv',
        'shared_vs_individual_summary.csv','a3_joint_operating_points.csv','a3_latency_round_metrics.csv','analysis_receipt.json']
    inputs={str((OUT/name).relative_to(ROOT)):sha(OUT/name) for name in input_names}
    inputs[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__).resolve())
    save(DEST/'recalculation_audit.json',dict(status='passed',source_outputs_verified=len(expected),m1_recomputed_rows=len(m1),
        m2_recomputed_models=len(joint),raw_timing_arrays_verified=len(rounds),new_model_execution=False,new_official_dataset_access=False,
        bootstrap_repetitions=2000,bootstrap_seed=20260918,
        tail_warning='Paired p95 saving is negative for both datasets; mean latency speedup does not establish tail latency speedup.'))
    save(DEST/'evidence_manifest.json',dict(source_repository_commit=subprocess_commit(),inputs=inputs,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(DEST.iterdir()) if p.is_file()},
        raw_predictions='artifacts/analyses/minimal_core_evidence_20260918_r1/m2_raw/'))
    for row in compact_summary:print(row)
    print(f'Audited {len(expected)} original outputs; manuscript materials: {DEST}')


def subprocess_commit():
    import subprocess
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


if __name__=='__main__':main()
