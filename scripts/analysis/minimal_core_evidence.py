"""M0-M3 minimal core evidence: cached A4 policy study and locked A3 GPU work points."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.analysis.paper_readiness import sha, save, read_json, write_csv, corrected_saving

OUT = ROOT / 'artifacts/analyses/minimal_core_evidence_20260918_r1'
AB_MANIFEST = ROOT / 'artifacts/analyses/early_exit_p5ab_20260905_092909/development_logits_manifest.json'
GUIDE = ROOT / 'docs/minimal_core_evidence_experiment_guide_20260918.md'
CASES = [('cifar10', 'cifar10_source', 'cifar10_target', 0.),
         ('cifar100_strict', 'cifar100_source', 'cifar100_target', 0.),
         ('cifar100_relaxed', 'cifar100_source', 'cifar100_confirmation', .04)]
A3 = {'cifar10': {'seeds': [54, 55, 56], 'threshold': .984},
      'cifar100': {'seeds': [66, 67, 68], 'threshold': .903}}
TOL = 1e-12


def now():
    return datetime.now(timezone.utc).isoformat()


def costs(dataset):
    return (2682624, 6240128, 6400) if dataset == 'cifar100' else (2676864, 6124928, 640)


def risk(metrics, worst, category=True):
    return (metrics['accuracy_drop'] <= TOL and metrics['balanced_accuracy_drop'] <= TOL and
            (not category or metrics['worst_class_accuracy_drop'] <= worst + TOL))


def metrics(record, early):
    from scripts.analysis.run_early_exit_p5ab import route_metrics
    result = route_metrics(record, early)
    result.pop('route_fractions')
    q = float(np.mean(early))
    result.update(early_fraction=q, cost_saving_fraction=corrected_saving(q, record['exit_cost'], record['head_cost']))
    result['expected_cost_fraction'] = 1 - result['cost_saving_fraction']
    return result


def array_hash(array):
    import hashlib
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def split_by_id(cohorts):
    """Primary pool first; overlap inherits assignment; novel target IDs fill class quotas."""
    result, assignments, labels_seen = {}, {}, {}
    for dataset, names in [('cifar10', ['cifar10_source', 'cifar10_target']),
                           ('cifar100', ['cifar100_source', 'cifar100_target', 'cifar100_confirmation'])]:
        generator = np.random.Generator(np.random.PCG64(20260918))
        assignments[dataset], labels_seen[dataset] = {}, {}
        for name in names:
            base = cohorts[name][0]
            ids, labels = base['sample_ids'], base['labels']
            if len(ids) != 5000 or len(set(map(int, ids))) != 5000:
                raise ValueError(f'Expected 5000 unique IDs: {name}')
            for row in cohorts[name]:
                order = np.argsort(row['sample_ids'])
                reference = np.argsort(ids)
                if not np.array_equal(row['sample_ids'][order], ids[reference]) or not np.array_equal(row['labels'][order], labels[reference]):
                    raise ValueError(f'Model ID/label alignment failure: {name}')
            f_ids, e_ids = [], []
            for label in sorted(set(map(int, labels))):
                class_ids = sorted(map(int, ids[labels == label]))
                if len(class_ids) != 5000 // (100 if dataset == 'cifar100' else 10):
                    raise ValueError(f'Unexpected class support: {name}/{label}')
                for item in class_ids:
                    if item in labels_seen[dataset] and labels_seen[dataset][item] != label:
                        raise ValueError('Same image ID has conflicting labels')
                    labels_seen[dataset][item] = label
                old_f = [i for i in class_ids if assignments[dataset].get(i) == 'F']
                old_e = [i for i in class_ids if assignments[dataset].get(i) == 'E']
                fresh = [i for i in class_ids if i not in assignments[dataset]]
                fresh = generator.permutation(fresh).tolist()
                need = len(class_ids) // 2 - len(old_f)
                if not 0 <= need <= len(fresh) or len(old_e) > len(class_ids) // 2:
                    raise ValueError('Cannot form aligned balanced fit/audit pools')
                for i in fresh[:need]: assignments[dataset][i] = 'F'
                for i in fresh[need:]: assignments[dataset][i] = 'E'
                f_ids += old_f + fresh[:need]
                e_ids += old_e + fresh[need:]
            result[name] = dict(fit_ids=sorted(f_ids), audit_ids=sorted(e_ids))
            if len(f_ids) != 2500 or len(e_ids) != 2500 or set(f_ids) & set(e_ids):
                raise ValueError('Invalid fit/audit boundary')
    for _, source, target, _ in CASES:
        if set(result[source]['fit_ids']) & set(result[target]['audit_ids']):
            raise ValueError('Target audit images leaked into source fitting')
    return result


def load_caches():
    from scripts.analysis.run_early_exit_p5ab import COHORTS, REPORTS, load_runs, resolve_recorded_path
    cohorts, inventory = {}, []
    cache_manifest = read_json(AB_MANIFEST)
    for cohort, (dataset, seeds, report) in COHORTS.items():
        matched = load_runs(REPORTS[report], seeds)
        cohorts[cohort] = []
        for seed in seeds:
            item = cache_manifest['files'][f'{cohort}/seed{seed}']
            path = resolve_recorded_path(item['path'])
            if sha(path) != item['sha256']:
                raise ValueError('Development cache checksum mismatch')
            run = matched[('multi_exit', seed)]
            directory = ROOT / 'artifacts/runs' / run['experiment_id']
            checkpoint = directory / 'checkpoints/model_best.pth'
            split = directory / 'split_indices.json'
            if sha(checkpoint) != item['multi_exit_checkpoint_sha256'] or sha(split) != item['split_indices_sha256']:
                raise ValueError('A4 checkpoint/split mismatch')
            with np.load(path, allow_pickle=False) as data:
                row = {k: data[k] for k in ('sample_ids', 'labels', 'final_logits', 'exit8_logits')}
            if not np.array_equal(row['sample_ids'], np.array(read_json(split)['calibration_indices'])):
                raise ValueError('Cache ID provenance mismatch')
            if not all(np.isfinite(row[k]).all() for k in ('final_logits', 'exit8_logits')):
                raise ValueError('Nonfinite logits')
            early, final, head = costs(dataset)
            row.update(seed=seed, dataset=dataset, exit_cost=early/final, head_cost=head/final)
            cohorts[cohort].append(row)
            inventory.append(dict(cohort=cohort, seed=seed, dataset=dataset, variant='A4_historical_full',
                experiment_id=run['experiment_id'], cache_path=str(path.relative_to(ROOT)), cache_sha256=sha(path),
                checkpoint_path=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                split_path=str(split.relative_to(ROOT)), split_sha256=sha(split), resolved_config=run['resolved_config'],
                sample_ids_sha256=array_hash(row['sample_ids']), labels_sha256=array_hash(row['labels']),
                history='Train/validation/calibration as original split; checkpoint selected on validation; development logits already used in historical analyses.'))
    return cohorts, inventory


def subset(row, ids):
    positions = {int(i): j for j, i in enumerate(row['sample_ids'])}
    index = np.array([positions[int(i)] for i in ids])
    return {k: v[index] if isinstance(v, np.ndarray) else v for k, v in row.items()}


def scores(row):
    from scripts.analysis.run_early_exit_p5ab import score_values
    return score_values('msp', row['exit8_logits'])


def candidates(rows):
    pooled = np.concatenate([scores(r) for r in rows])
    return np.unique(np.r_[np.quantile(pooled, np.linspace(0, 1, 1001)), np.nextafter(pooled.max(), np.inf)])


def select(rows, grid, strategy, worst):
    values = [scores(r) for r in rows]
    best, feasible_count = None, 0
    for threshold in grid:
        per_model = [metrics(r, s >= threshold) for r, s in zip(rows, values)]
        if not all(risk(v, worst, strategy != 'C') for v in per_model):
            continue
        feasible_count += 1
        savings = [v['cost_saving_fraction'] for v in per_model]
        mean, minimum = float(np.mean(savings)), min(savings)
        objective = (mean, minimum, float(threshold)) if strategy == 'A' else (minimum, mean, float(threshold))
        if best is None or objective > best[0]:
            best = (objective, float(threshold))
    return dict(threshold=None if best is None else best[1], candidates=len(grid), risk_feasible_candidates=feasible_count,
                status='no_risk_feasible_candidate' if best is None else 'selected')


def verify_inputs(output):
    manifest = read_json(output / 'input_manifest.json')
    for name, checksum in manifest['files'].items():
        if sha(ROOT / name) != checksum:
            raise ValueError(f'Frozen input/source changed: {name}')
    for item in manifest['caches'] + manifest['a3_checkpoints']:
        for key in ('cache', 'checkpoint', 'split'):
            if f'{key}_path' in item and sha(ROOT / item[f'{key}_path']) != item[f'{key}_sha256']:
                raise ValueError(f'Frozen {key} changed')


def prepare(output):
    import torch
    from scripts.analysis.analyze_early_exit_p5c import discover_new_runs
    from scripts.analysis.benchmark_early_exit_p8_v3 import validate_protocol
    from image_classification.training.provenance import runtime_provenance, source_fingerprint
    if output.exists():
        raise FileExistsError('Output already exists; inspect, never silently replace')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3080 Ti':
        raise RuntimeError('RTX3080Ti required')
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError('Need >=2 GiB free')
    validate_protocol()
    cohorts, inventory = load_caches()
    partitions = split_by_id(cohorts)
    runs, sweep_receipts = discover_new_runs('p5c_r1')
    checkpoints = []
    for dataset, matrix in A3.items():
        for seed in matrix['seeds']:
            run = runs[(dataset, 'A3', seed)]
            directory = ROOT / 'artifacts/runs' / run['experiment_id']
            summary = read_json(directory / 'summary.json')
            checkpoint = directory / 'checkpoints/model_best.pth'
            if sha(checkpoint) != summary['best_checkpoint_sha256'] or summary['test_evaluated'] is not False:
                raise ValueError('A3 checkpoint receipt mismatch')
            checkpoints.append(dict(dataset=dataset, seed=seed, variant='A3', experiment_id=run['experiment_id'],
                checkpoint_path=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                split_path=str((directory/'split_indices.json').relative_to(ROOT)), split_sha256=sha(directory/'split_indices.json'),
                summary_path=str((directory/'summary.json').relative_to(ROOT)), summary_sha256=sha(directory/'summary.json'),
                resolved_config=run['resolved_config'], threshold=matrix['threshold'],
                history='No test evaluation in training receipt. Official benchmark previously exposed to other project models; historical calibration/P8 analyses known.'))
    # Check presence/byte hashes of official archive files without instantiating test datasets.
    official_archives = ['data/cifar-10-batches-py/test_batch', 'data/cifar-100-python/test']
    fingerprints = source_fingerprint()
    for path in [GUIDE, AB_MANIFEST] + [ROOT/p for p in official_archives] + [ROOT/i['summary_path'] for i in checkpoints]:
        fingerprints[str(path.relative_to(ROOT))] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_provenance()
    runtime.update(cpu=platform.processor(), os=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                   cpu_info=subprocess.check_output(['lscpu'], text=True), free_bytes=shutil.disk_usage(ROOT).free)
    save(output/'input_manifest.json', dict(files=fingerprints, caches=inventory, a3_checkpoints=checkpoints, sweeps=sweep_receipts, runtime=runtime))
    for name, part in partitions.items():
        part.update(fit_ids_sha256=array_hash(part['fit_ids']), audit_ids_sha256=array_hash(part['audit_ids']))
    save(output/'fit_audit_partitions.json', partitions)
    protocol = dict(schema_version=1, status='frozen_before_result_generation', frozen_at=now(),
        guide=str(GUIDE.relative_to(ROOT)), guide_sha256=sha(GUIDE), variant_m1='historical_A4', score='msp',
        cases=CASES, rng='NumPy PCG64', rng_seed=20260918, quantiles=1001, tolerance=TOL, savings_floor=.15,
        fit_audit_split='Per class sorted IDs then seeded permutation; source pool first, overlapping IDs inherit F/E, novel confirmation IDs fill exactly half-class quota. Reset PCG64 per dataset.',
        data_amendment='CIFAR100 confirmation has a different 5000-image pool. ID-aligned overlap assignment is used instead of independently random splits to ensure target E never overlaps source F. Every cohort still has balanced 2500 F and 2500 E.',
        split_file_sha256=sha(output/'fit_audit_partitions.json'), input_manifest_sha256=sha(output/'input_manifest.json'),
        strategies=dict(S='shared max-min, all class constraints', I='per-model fit recalibration', A='shared mean, all class constraints', C='shared max-min, no class constraint'),
        m2=dict(models=A3, scope='official benchmark supplementary locked assessment; prior project exposure', threshold_searches=0,
                device='cuda', gpu='NVIDIA GeForce RTX 3080 Ti', batch_size=1, cpu_threads=1, rounds=5, warmup=100, timed_calls=1000,
                precision='deterministic FP32 TF32 disabled', rtol=1e-4, atol=1e-5, required_route_errors=0, required_prediction_errors=0,
                full_accuracy='10000 official test images evaluated as singleton resident inputs',
                timing_workload='One seeded permutation per dataset; first 1000 official IDs, identical across six seed-round schedules for that dataset; 100 warmup calls cycle within this subset',
                latency_scope='synchronized model-only wall clock; excludes preprocessing/input transfer, queueing, and path-copy after timing'),
        bootstrap=dict(repetitions=2000, seed=20260918, scope='paired stratified image bootstrap conditional on three fixed models; not population or simultaneous risk guarantee'),
        authorization=dict(m1='current user request', m2='User explicitly approved six fixed A3 checkpoints, CIFAR10/CIFAR100 official locked assessment and singleton timing via question reply in this task; launch requires explicit flag'),
        stop_rules='Input/source/ID/correctness errors fail-stop; keep partial output; no automatic scientific retry')
    save(output/'protocol_manifest.json', protocol)
    print('M0 prepared: 15 A4 caches, 6 A3 checkpoints; partitions frozen; official dataset not instantiated.', flush=True)


def bootstrap_pair(labels, diffs):
    """Multinomial resampling of image-level paired effects, preserving model dependence."""
    generator = np.random.Generator(np.random.PCG64(20260918))
    sims = np.zeros(2000)
    effect = np.mean(np.stack(diffs), axis=0)
    for label in sorted(set(map(int, labels))):
        values = effect[labels == label]
        draws = generator.multinomial(len(values), np.full(len(values), 1/len(values)), size=2000)
        sims += draws @ values
    sims /= len(labels)
    return dict(paired_accuracy_difference=float(effect.mean()), paired_ci95_low=float(np.quantile(sims, .025)),
                paired_ci95_high=float(np.quantile(sims, .975)))


def m1(output):
    verify_inputs(output)
    if (output/'m1_completed.json').exists():
        raise FileExistsError('M1 already completed')
    cohorts, _ = load_caches()
    partitions = read_json(output/'fit_audit_partitions.json')
    if sha(output/'fit_audit_partitions.json') != read_json(output/'protocol_manifest.json')['split_file_sha256']:
        raise ValueError('Partition changed')
    selections, rows, per_class, paired = [], [], [], {}
    for case, source, target, worst in CASES:
        print(f'M1 {case}', flush=True)
        fit = [subset(r, partitions[source]['fit_ids']) for r in cohorts[source]]
        grid = candidates(fit)
        for strategy in ('S', 'I', 'A', 'C'):
            shared = select(fit, grid, strategy, worst) if strategy != 'I' else None
            for split, cohort, pool in [('source_F', source, 'fit_ids'), ('target_E', target, 'audit_ids')]:
                for full in cohorts[cohort]:
                    selection = shared
                    if strategy == 'I':
                        individual_fit = subset(full, partitions[cohort]['fit_ids'])
                        selection = select([individual_fit], candidates([individual_fit]), 'I', worst)
                    selected = dict(case=case, strategy=strategy, split=split, seed=full['seed'], **selection)
                    selections.append(selected)
                    record = subset(full, partitions[cohort][pool])
                    threshold = selection['threshold']
                    if threshold is None:
                        rows.append(dict(case=case, strategy=strategy, split=split, seed=full['seed'], selection_status=selection['status'], risk_feasible=False, saving_floor_pass=False))
                        continue
                    early = scores(record) >= threshold
                    value = metrics(record, early)
                    final_pred, exit_pred = record['final_logits'].argmax(1), record['exit8_logits'].argmax(1)
                    prediction = np.where(early, exit_pred, final_pred)
                    paired[(case, split, strategy, full['seed'])] = (record['labels'], prediction == record['labels'], record['sample_ids'])
                    info = dict(case=case, strategy=strategy, split=split, seed=full['seed'], threshold=threshold,
                        selection_status=selection['status'], candidates=selection['candidates'], model_recipe='A4_historical_full',
                        threshold_parameters=3 if strategy=='I' else 1,
                        target_recalibrations=3 if strategy=='I' and split=='target_E' else 0,
                        source_constraints_include_category=strategy!='C',
                        risk_feasible=risk(value,worst), selection_risk_feasible=risk(value,worst,strategy!='C'),
                        saving_floor_pass=value['cost_saving_fraction'] >= .15-TOL,
                        evaluated_ids_sha256=array_hash(record['sample_ids']), fit_ids_sha256=array_hash(partitions[cohort if strategy=='I' else source]['fit_ids']), **value)
                    rows.append(info)
                    for cls in sorted(set(map(int, record['labels']))):
                        mask=record['labels']==cls
                        per_class.append(dict(case=case,strategy=strategy,split=split,seed=full['seed'],class_id=cls,
                            support=int(mask.sum()),final_correct=int((final_pred[mask]==cls).sum()),policy_correct=int((prediction[mask]==cls).sum())))
    write_csv(output/'shared_vs_individual_seed_metrics.csv', rows)
    write_csv(output/'m1_per_class_counts.csv', per_class)
    summaries=[]
    for case, _, _, _ in CASES:
        for split in ('source_F','target_E'):
            for strategy in ('S','I','A','C'):
                group=[r for r in rows if (r['case'],r['split'],r['strategy'])==(case,split,strategy)]
                info=dict(case=case,split=split,strategy=strategy,n=3,selected_seeds=sum(r['selection_status']=='selected' for r in group),
                    all_risk_feasible=all(r['risk_feasible'] for r in group),all_saving_floor_pass=all(r['saving_floor_pass'] for r in group),
                    threshold_parameters=3 if strategy=='I' else 1,target_recalibrations=3 if strategy=='I' and split=='target_E' else 0,
                    search_candidates_total=sum(r.get('candidates',0) for r in group) if strategy=='I' else group[0].get('candidates',0))
                for metric in ('accuracy','reference_accuracy','balanced_accuracy','accuracy_drop','worst_class_accuracy_drop','cost_saving_fraction','early_fraction'):
                    values=[r[metric] for r in group if metric in r]
                    if len(values)==3:
                        info.update({metric+'_mean':float(np.mean(values)),metric+'_sample_sd':float(np.std(values,ddof=1)),metric+'_min':min(values),metric+'_max':max(values)})
                if strategy!='S' and all((case,split,s,r['seed']) in paired for r in group for s in ('S',strategy)):
                    diffs=[]
                    for r in group:
                        l, base, ids=paired[(case,split,'S',r['seed'])]
                        other_l, other, other_ids=paired[(case,split,strategy,r['seed'])]
                        if not np.array_equal(ids,other_ids) or not np.array_equal(l,other_l): raise ValueError('Pairing mismatch')
                        diffs.append(other.astype(float)-base.astype(float))
                    info.update(bootstrap_pair(l,diffs))
                    info['paired_direction']=strategy+' minus S'
                summaries.append(info)
    write_csv(output/'shared_vs_individual_summary.csv',summaries)
    save(output/'policy_selection.json',dict(status='completed',selections=selections))
    files=['shared_vs_individual_seed_metrics.csv','shared_vs_individual_summary.csv','m1_per_class_counts.csv','policy_selection.json']
    save(output/'m1_completed.json',dict(status='completed',finished_at=now(),expected_rows=72,
        files={name:sha(output/name) for name in files}, interpretation='Post-hoc fit/audit separation. I uses target fitting information. No human maintenance cost measurement.'))


def load_official(dataset):
    import torch
    from image_classification.data.cifar import DATASET_SPECS, _transforms
    spec=DATASET_SPECS[dataset]
    data=spec.dataset_class(root=ROOT/'data',train=False,download=False,transform=_transforms(spec)[1])
    if len(data)!=10000: raise ValueError('Expected 10000 official images')
    return torch.stack([data[i][0] for i in range(len(data))]),np.array(data.targets,dtype=np.int64)


def m2(output, authorize):
    if not authorize: raise PermissionError('Explicit authorization flag required')
    verify_inputs(output)
    from scripts.analysis.benchmark_early_exit_p8 import load_model, latency_statistics
    from scripts.analysis.benchmark_early_exit_p8_v3 import configure_numerics,time_mode,telemetry
    from scripts.analysis.analyze_early_exit_p5c import discover_new_runs
    import torch
    marker=ROOT/'artifacts/minimal_core_official_access_20260918_r1.json'
    if marker.exists() or (output/'official_started.json').exists(): raise FileExistsError('Official access already started; preserve failure/results')
    # Exclusive access marker before official images are instantiated; interrupted runs never auto-retry.
    with marker.open('x') as handle:
        json.dump(dict(output=str(output.relative_to(ROOT)),started_at=now(),protocol_sha256=sha(output/'protocol_manifest.json'),
            authorization='Explicit user-approved launch flag for CIFAR10/CIFAR100 six A3 checkpoints at frozen thresholds',models=A3),handle,indent=2)
    save(output/'official_started.json',read_json(marker))
    configure_numerics()
    torch.set_num_threads(1)
    device=torch.device('cuda')
    runs,_=discover_new_runs('p5c_r1')
    seed_rows, round_rows, correctness, class_rows=[],[],[],[]
    raw=output/'m2_raw'; raw.mkdir(exist_ok=False)
    for dataset,matrix in A3.items():
        inputs,labels=load_official(dataset)
        inputs=inputs.to(device)
        order=np.random.Generator(np.random.PCG64(20260918)).permutation(len(labels))[:1000]
        for seed in matrix['seeds']:
            print(f'M2 {dataset} seed{seed}: locked full test + singleton timing',flush=True)
            model,_=load_model(runs[(dataset,'A3',seed)],device)
            threshold=matrix['threshold']
            final_logits=np.empty((10000,10 if dataset=='cifar10' else 100),dtype=np.float32)
            exit_logits=np.empty_like(final_logits)
            routes=np.empty(10000,dtype=bool)
            policy_predictions=np.empty(10000,dtype=np.int64)
            check=dict(dataset=dataset,seed=seed,samples_checked=0,route_errors=0,prediction_errors=0,logit_tolerance_errors=0,max_abs_logit_difference=0.)
            with torch.inference_mode():
                for i in range(10000):
                    batch=inputs[i:i+1]
                    early_logit=model.forward_to_exit(batch,8)
                    final_logit=model.forward_to_exit(batch,None)
                    early=early_logit.float().softmax(1).amax(1)>=threshold
                    dynamic,path=model.forward_with_policy(batch,threshold,exit_position=8)
                    reference=torch.where(early[:,None],early_logit,final_logit)
                    diff=(dynamic-reference).abs()
                    check['route_errors']+=int(((path==0)!=early).sum())
                    check['prediction_errors']+=int((dynamic.argmax(1)!=reference.argmax(1)).sum())
                    check['logit_tolerance_errors']+=int((diff>1e-5+1e-4*reference.abs()).any(1).sum())
                    check['max_abs_logit_difference']=max(check['max_abs_logit_difference'],float(diff.max()))
                    check['samples_checked']+=1
                    final_logits[i]=final_logit.cpu().numpy()[0]
                    exit_logits[i]=early_logit.cpu().numpy()[0]
                    routes[i]=bool(early.item())
                    policy_predictions[i]=int(dynamic.argmax(1).item())
                    if i and i%2000==0: print(f'  audited {i}/10000',flush=True)
            correctness.append(check)
            save(output/'m2_correctness.json',correctness)
            np.savez_compressed(raw/f'{dataset}_s{seed}_predictions.npz',sample_ids=np.arange(10000),labels=labels,final_logits=final_logits,
                exit8_logits=exit_logits,early=routes,policy_predictions=policy_predictions)
            if any(check[k] for k in ('route_errors','prediction_errors','logit_tolerance_errors')):
                raise RuntimeError(f'Singleton correctness failed: {check}')
            early_cost,final_cost,head_cost=costs(dataset)
            record=dict(labels=labels,exit8_logits=exit_logits,final_logits=final_logits,exit_cost=early_cost/final_cost,head_cost=head_cost/final_cost)
            value=metrics(record,routes)
            checkpoint=next(r for r in read_json(output/'input_manifest.json')['a3_checkpoints'] if r['dataset']==dataset and r['seed']==seed)
            value.update(dataset=dataset,seed=seed,threshold=threshold,variant='A3',data_scope='all 10000 official test images',
                         exit8_accuracy=float((exit_logits.argmax(1)==labels).mean()), checkpoint_sha256=checkpoint['checkpoint_sha256'],
                         risk_budget_scope='historical CIFAR empirical budget; supplementary observed outcomes',
                         risk_feasible=risk(value,0 if dataset=='cifar10' else .04),saving_floor_pass=value['cost_saving_fraction']>=.15-TOL)
            seed_rows.append(value)
            for cls in sorted(set(map(int,labels))):
                mask=labels==cls
                class_rows.append(dict(dataset=dataset,seed=seed,class_id=cls,support=int(mask.sum()),
                    final_correct=int((final_logits.argmax(1)[mask]==cls).sum()),policy_correct=int((policy_predictions[mask]==cls).sum())))
            batches=[inputs[int(i):int(i)+1] for i in order]
            workload_metrics=metrics({**record,**{k:record[k][order] for k in ('labels','exit8_logits','final_logits')}},routes[order])
            np.savez_compressed(raw/f'{dataset}_s{seed}_workload.npz',sample_ids=order,expected_early=routes[order],labels=labels[order])
            for rnd in range(5):
                modes=['final_only','actual_dynamic']; random.Random(20260918+seed*100+rnd).shuffle(modes)
                pair=[]
                for mode_index,mode in enumerate(modes):
                    before=telemetry()
                    latencies,counts,memory=time_mode(model,batches,mode,threshold,100,1000,device)
                    if mode=='actual_dynamic' and not np.array_equal(counts,routes[order].astype(int)):
                        raise RuntimeError('Timed route IDs/counts differ from audited same-workload routes')
                    path=raw/f'{dataset}_s{seed}_r{rnd}_{mode}.npz'
                    arrays=dict(latency_ms=latencies,sample_ids=order)
                    if counts is not None: arrays['early_counts']=counts
                    np.savez_compressed(path,**arrays)
                    row=dict(dataset=dataset,seed=seed,round=rnd,mode=mode,mode_order=mode_index,threshold=threshold,batch_size=1,
                        workload_samples=1000,workload_unique_images=1000,workload_ids_sha256=array_hash(order),
                        workload_early_fraction=workload_metrics['early_fraction'],workload_mac_saving_fraction=workload_metrics['cost_saving_fraction'],
                        full_test_mac_saving_fraction=value['cost_saving_fraction'],raw_path=str(path.relative_to(output)),raw_sha256=sha(path),
                        **latency_statistics(latencies,1),**memory,telemetry_before=json.dumps(before),telemetry_after=json.dumps(telemetry()))
                    pair.append(row)
                final=next(r for r in pair if r['mode']=='final_only')['latency_mean_ms']
                dynamic=next(r for r in pair if r['mode']=='actual_dynamic')['latency_mean_ms']
                for row in pair: row['paired_latency_saving_fraction']=1-dynamic/final
                round_rows+=pair
                write_csv(output/'a3_latency_round_metrics.csv',round_rows)
            del model
            torch.cuda.empty_cache()
            write_csv(output/'a3_accuracy_mac_seed_metrics.csv',seed_rows)
            write_csv(output/'m2_per_class_counts.csv',class_rows)
        del inputs
    save(output/'m2_completed.json',dict(status='completed',finished_at=now(),official_models=6,official_images_per_model=10000,
        raw_latency_arrays=60,correctness=correctness,files={name:sha(output/name) for name in ['a3_accuracy_mac_seed_metrics.csv','a3_latency_round_metrics.csv','m2_per_class_counts.csv','m2_correctness.json']}))


def numeric_rows(path):
    with path.open() as f: return list(csv.DictReader(f))


def m3(output):
    verify_inputs(output)
    for stage in ('m1','m2'):
        receipt=read_json(output/f'{stage}_completed.json')
        for name,wanted in receipt['files'].items():
            if sha(output/name)!=wanted: raise ValueError('Stage output changed')
    rounds=numeric_rows(output/'a3_latency_round_metrics.csv')
    accuracy=numeric_rows(output/'a3_accuracy_mac_seed_metrics.csv')
    if len(rounds)!=60 or len(accuracy)!=6 or len(numeric_rows(output/'shared_vs_individual_seed_metrics.csv'))!=72:
        raise ValueError('Incomplete output row counts')
    joint=[]
    for acc in accuracy:
        with np.load(output/'m2_raw'/f"{acc['dataset']}_s{acc['seed']}_predictions.npz") as data:
            early, final, head=costs(acc['dataset'])
            record=dict(labels=data['labels'],exit8_logits=data['exit8_logits'],final_logits=data['final_logits'],exit_cost=early/final,head_cost=head/final)
            recomputed=metrics(record,data['early'])
            expected_predictions=np.where(data['early'],data['exit8_logits'].argmax(1),data['final_logits'].argmax(1))
            if not np.array_equal(data['sample_ids'],np.arange(10000)) or not np.array_equal(expected_predictions,data['policy_predictions']):
                raise ValueError('Full-test prediction/ID audit failed')
            for key in ('accuracy','reference_accuracy','accuracy_drop','worst_class_accuracy_drop','cost_saving_fraction','early_fraction'):
                if abs(recomputed[key]-float(acc[key]))>TOL:raise ValueError('Full-test metric audit failed')
            all_routes=data['early'].copy()
        group=[r for r in rounds if (r['dataset'],r['seed'])==(acc['dataset'],acc['seed'])]
        if len(group)!=10: raise ValueError('Incomplete timing seed')
        for row in group:
            path=output/row['raw_path']
            if sha(path)!=row['raw_sha256']: raise ValueError('Raw timing checksum mismatch')
            with np.load(path) as d:
                if d['latency_ms'].shape!=(1000,) or not np.isfinite(d['latency_ms']).all() or np.any(d['latency_ms']<=0): raise ValueError('Invalid raw timing')
                if abs(d['latency_ms'].mean()-float(row['latency_mean_ms']))>1e-9 or array_hash(d['sample_ids'])!=row['workload_ids_sha256']: raise ValueError('Timing summary/IDs mismatch')
                if row['mode']=='actual_dynamic' and not np.array_equal(d['early_counts'],all_routes[d['sample_ids']].astype(int)):
                    raise ValueError('Timed workload route audit failed')
        for rnd in range(5):
            pair=[r for r in group if int(r['round'])==rnd]
            if len(pair)!=2 or {r['mode'] for r in pair}!={'final_only','actual_dynamic'} or {int(r['mode_order']) for r in pair}!={0,1}:
                raise ValueError('Invalid paired timing round')
            final_ms=float(next(r for r in pair if r['mode']=='final_only')['latency_mean_ms'])
            dynamic_ms=float(next(r for r in pair if r['mode']=='actual_dynamic')['latency_mean_ms'])
            if any(abs(float(r['paired_latency_saving_fraction'])-(1-dynamic_ms/final_ms))>TOL for r in pair):
                raise ValueError('Paired timing saving audit failed')
        value=dict(acc)
        value['timing_scope']='1000 unique official images; 5 repetitions per mode; resident singleton FP32'
        value['seed_latency_saving_fraction']=float(np.mean([float(r['paired_latency_saving_fraction']) for r in group if r['mode']=='actual_dynamic']))
        for mode in ('final_only','actual_dynamic'):
            for key in ('latency_mean_ms','latency_p95_ms','throughput_samples_per_second'):
                value[mode+'_'+key]=float(np.mean([float(r[key]) for r in group if r['mode']==mode]))
        value['p95_scope']='mean of five per-round p95 values, not pooled population p95'
        value['timed_subset_early_fraction']=float(group[0]['workload_early_fraction'])
        value['timed_subset_mac_saving_fraction']=float(group[0]['workload_mac_saving_fraction'])
        joint.append(value)
    write_csv(output/'a3_joint_operating_points.csv',joint)
    summaries=[]
    for dataset in A3:
        rows=[r for r in joint if r['dataset']==dataset]
        info=dict(dataset=dataset,n=3)
        for key in ('accuracy','reference_accuracy','accuracy_drop','worst_class_accuracy_drop','cost_saving_fraction','seed_latency_saving_fraction',
                    'final_only_latency_mean_ms','actual_dynamic_latency_mean_ms','final_only_latency_p95_ms','actual_dynamic_latency_p95_ms'):
            vals=[float(r[key]) for r in rows]
            info[key+'_mean']=float(np.mean(vals));info[key+'_sample_sd']=float(np.std(vals,ddof=1))
        summaries.append(info)
    write_csv(output/'a3_joint_summary.csv',summaries)
    # One compact table per dataset; model-level units are the three training seeds.
    tex=['\\begin{tabular}{lrrrr}', '\\hline', 'Dataset & Policy (\\%) & MAC saved (\\%) & Latency saved (\\%) & Seed SD (pp) \\\\', '\\hline']
    for r in summaries:
        tex.append(f"{r['dataset']} & {100*r['accuracy_mean']:.2f} & {100*r['cost_saving_fraction_mean']:.2f} & {100*r['seed_latency_saving_fraction_mean']:.2f} & {100*r['seed_latency_saving_fraction_sample_sd']:.2f} \\\\")
    tex+=['\\hline','\\end{tabular}']
    (output/'a3_joint_summary.tex').write_text('\n'.join(tex)+'\n')
    # Discrete target-E points; no smoothed frontier, red indicates class-budget failure.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policy_rows=numeric_rows(output/'shared_vs_individual_seed_metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,3.4))
    for axis,(case,_,_,worst) in zip(axes,CASES):
        for strategy,marker in [('S','o'),('I','s'),('A','^'),('C','D')]:
            rs=[r for r in policy_rows if r['case']==case and r['split']=='target_E' and r['strategy']==strategy and r['selection_status']=='selected']
            for i,r in enumerate(rs):
                class_fail=float(r['worst_class_accuracy_drop'])>worst+TOL
                axis.scatter(100*float(r['cost_saving_fraction']),100*float(r['accuracy_drop']),marker=marker,
                    c='#b2182b' if class_fail else '#2166ac',s=40,label=strategy if i==0 else None,alpha=.75)
        axis.set_title(case.replace('_',' / '));axis.axhline(0,color='gray',lw=.7)
        axis.set_xlabel('Corrected MAC saving (%)');axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[0].set_ylabel('Policy accuracy drop (pp)')
    fig.tight_layout();fig.savefig(output/'shared_policy_tradeoffs.pdf',bbox_inches='tight');plt.close(fig)
    checks=read_json(output/'m2_correctness.json')
    failed=[dict(dataset=r['dataset'],seed=r['seed'],risk_feasible=r['risk_feasible'],saving_floor_pass=r['saving_floor_pass']) for r in joint if r['risk_feasible']!='True' or r['saving_floor_pass']!='True']
    save(output/'analysis_receipt.json',dict(status='passed',technical_audit=True,scientific_failed_points=failed,all_seeds_retained=True,
        m1_rows=72,m2_models=6,timing_rows=60,checked_sample_executions=sum(c['samples_checked'] for c in checks),
        correctness_errors={k:sum(c[k] for c in checks) for k in ('route_errors','prediction_errors','logit_tolerance_errors')},
        limitations=['post-hoc historical development exposure','I recalibrates target, shared methods do not','only three seeds',
            'official benchmarks exposed to other project methods','resident subset timing, full-test accuracy',
            'no population or simultaneous risk guarantee','no end-to-end service timing']))
    lines=['# Minimal core evidence — completed', '', 'All scientific negative results are retained. Technical completion does not imply constraint success.', '',
           'See shared_vs_individual_summary.csv and a3_joint_summary.csv; policy selection is post-hoc development analysis. I uses target fitting information.', '',
           '| Dataset | Full-test policy accuracy (%) | Full-test MAC saving (%) | Resident subset latency saving (%) |',
           '|---|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['dataset']} | {100*r['accuracy_mean']:.4f} | {100*r['cost_saving_fraction_mean']:.4f} | {100*r['seed_latency_saving_fraction_mean']:.4f} ± {100*r['seed_latency_saving_fraction_sample_sd']:.4f} |")
    lines+=['','Full test =10000 images/model; timed workload =1000 unique same-test images, repeated five rounds. These scopes are distinct.',
            'Official datasets were used only after explicit authorization for six fixed A3 checkpoints and two locked thresholds. No threshold search on official data.',
            'All graph/table claims must retain earlier large-batch slowdowns and P6/P7 stop/failure results.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','m1','m2','m3'],required=True)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--authorize-a3-cifar-official-test',action='store_true')
    a=p.parse_args()
    if a.stage=='m2':m2(a.output,a.authorize_a3_cifar_official_test)
    else:{'prepare':prepare,'m1':m1,'m3':m3}[a.stage](a.output)


if __name__=='__main__':main()
