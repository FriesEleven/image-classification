"""Development-only paper-readiness analyses and fail-stop serial stages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
AB = ROOT / 'artifacts/analyses/early_exit_p5ab_20260905_092909'
AC = ROOT / 'artifacts/analyses/early_exit_p5c_20260905_143000'


def read_json(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def corrected_saving(early_fraction, early_cost, head_cost):
    return 1 - early_fraction * early_cost - (1 - early_fraction) * (1 + head_cost)


def compute_matched(values, tolerance=.005):
    return len(values) == 3 and all(v is not None for v in values) and max(values) - min(values) <= tolerance + 1e-12


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def load_cohorts():
    import numpy as np
    cohorts = {}
    for key, item in read_json(AB / 'development_logits_manifest.json')['files'].items():
        path = ROOT / item['path']
        if sha(path) != item['sha256']:
            raise ValueError(f'Logits hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            record = {k: data[k] for k in data.files}
        cohort, seed = key.split('/')
        dataset = 'cifar100' if cohort.startswith('cifar100') else 'cifar10'
        early, final, head = (2682624, 6240128, 6400) if dataset == 'cifar100' else (2676864, 6124928, 640)
        record.update(seed=int(seed.removeprefix('seed')), exit_cost=early / final, head_cost=head / final)
        for required in ('labels', 'exit8_logits', 'final_logits'):
            if required not in record or len(record[required]) != item['samples']:
                raise ValueError(f'Invalid development cache: {key}/{required}')
        cohorts.setdefault(cohort, []).append(record)
    return cohorts


def compare(output):
    import numpy as np
    from scripts.analysis.run_early_exit_p5ab import route_metrics, score_values
    cohorts = load_cohorts()
    cases = [('cifar10', 'cifar10_source', 'cifar10_target', 0.),
             ('cifar100_strict', 'cifar100_source', 'cifar100_target', 0.),
             ('cifar100_relaxed', 'cifar100_source', 'cifar100_confirmation', .04)]
    rows, selections = [], []

    def metrics(record, scores, threshold):
        result = route_metrics(record, scores >= threshold)
        result['historical_no_overhead_saving'] = result['cost_saving_fraction']
        result['cost_saving_fraction'] = corrected_saving(result['route_fractions'][0], record['exit_cost'], record['head_cost'])
        result['expected_cost_fraction'] = 1 - result['cost_saving_fraction']
        return result

    def feasible(values, worst):
        return all(r['accuracy_drop'] <= 1e-12 and r['balanced_accuracy_drop'] <= 1e-12 and
                   r['worst_class_accuracy_drop'] <= worst + 1e-12 for r in values)

    for name, source, target, worst in cases:
        for method in ('msp', 'entropy', 'margin'):
            print(f'Comparing {name}/{method}', flush=True)
            source_scores = [score_values(method, r['exit8_logits']) for r in cohorts[source]]
            pooled = np.concatenate(source_scores)
            # Identical source-quantile grid for all scores; include all/none exits.
            thresholds = np.unique(np.r_[np.quantile(pooled, np.linspace(0, 1, 1001)),
                                         np.nextafter(pooled.max(), np.inf)])
            candidates = []
            for threshold in thresholds:
                vals = [metrics(r, s, threshold) for r, s in zip(cohorts[source], source_scores)]
                savings = [v['cost_saving_fraction'] for v in vals]
                candidates.append((float(threshold), vals, float(np.mean(savings)), min(savings)))
            risk_candidates = [c for c in candidates if feasible(c[1], worst)]
            chosen = [('same_risk', None, max(risk_candidates, key=lambda c: (c[3], c[2], c[0])) if risk_candidates else None)]
            for floor in (.10, .20, .30):
                near = [c for c in candidates if abs(c[2] - floor) <= .0025 + 1e-12]
                # Compute-only view: no risk claim. Equal tuning rule, source risk minimized.
                selected = min(near, key=lambda c: (max(v['worst_class_accuracy_drop'] for v in c[1]),
                               max(v['accuracy_drop'] for v in c[1]), abs(c[2] - floor), -c[0])) if near else None
                chosen.append(('matched_compute', floor, selected))
            target_scores = [score_values(method, r['exit8_logits']) for r in cohorts[target]]
            for view, floor, selected in chosen:
                selection = dict(case=name, method=method, view=view, requested_saving=floor,
                                 status='source_infeasible' if selected is None else 'selected')
                if selected is not None:
                    threshold, source_vals, mean_saving, _ = selected
                    target_vals = [metrics(r, s, threshold) for r, s in zip(cohorts[target], target_scores)]
                    selection.update(threshold=threshold, source_mean_saving=mean_saving,
                                     target_mean_saving=float(np.mean([v['cost_saving_fraction'] for v in target_vals])),
                                     target_risk_feasible=feasible(target_vals, worst),
                                     target_saving_floor_pass=all(v['cost_saving_fraction'] >= .15 for v in target_vals))
                    for split, records, vals in [('source', cohorts[source], source_vals), ('target', cohorts[target], target_vals)]:
                        for record, value in zip(records, vals):
                            rows.append(dict(case=name, method=method, view=view, requested_saving=floor,
                                             threshold=threshold, split=split, seed=record['seed'], **value))
                selections.append(selection)
    for selection in selections:
        if selection['view'] != 'matched_compute':
            continue
        peers = [s for s in selections if s['case'] == selection['case'] and s['view'] == selection['view'] and
                 s['requested_saving'] == selection['requested_saving']]
        selection['target_compute_matched'] = compute_matched([s.get('target_mean_saving') for s in peers])
    save(output / 'fair_comparison.json', dict(status='completed', selections=selections,
         interpretation='Exploratory development-only reanalysis after earlier outcomes were known. Empirical constraints, not population risk guarantees. Same images across some model seeds; no new-image holdout claim. MAC excludes score-operator cost; not wall-clock equivalence. Margin is top-two softmax probability margin. Historical PCEE/UCB adapters are not claimed as official baselines.'))
    write_csv(output / 'fair_comparison_seed_metrics.csv', rows)


def tradeoffs(output):
    import numpy as np
    with (AC / 'tables/variant_seed_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    runtimes = {}
    for path in (ROOT / 'artifacts/sweeps').glob('*/manifest.json'):
        data = read_json(path)
        for run in data.get('runs', []):
            if run.get('status') == 'completed' and run.get('return_code') == 0:
                runtimes.setdefault(run.get('experiment_id'), []).append((path, data, run))
    for row in rows:
        row['historical_policy_saving'] = row['policy_cost_saving_fraction']
        if row['policy_early_fraction']:
            early, final, head = (2682624, 6240128, 6400) if row['dataset'] == 'cifar100' else (2676864, 6124928, 640)
            row['corrected_policy_saving'] = corrected_saving(float(row['policy_early_fraction']), early/final, head/final)
        row['training_timing_status'] = 'unavailable'
        matches = runtimes.get(row['experiment_id'], [])
        if len(matches) == 1:
            path, manifest, run = matches[0]
            row.update(training_manifest=str(path.relative_to(ROOT)), training_manifest_sha256=sha(path),
                       training_gpu=manifest.get('runtime', {}).get('gpu', 'unavailable'))
            start, end = run.get('started_at'), run.get('finished_at')
            if start and end:
                seconds = (datetime.fromisoformat(end.replace('Z', '+00:00')) - datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()
                row.update(training_wall_seconds=seconds, training_timing_status='historical_descriptive_not_controlled')
        row['training_design'] = {'A0': 'baseline', 'A1': 'single_exit_no_KD', 'A2': 'single_exit_KD',
                                  'A3': 'two_exits_no_KD', 'A4': 'two_exits_KD'}[row['variant']]
    aggregates = []
    for dataset in ('cifar10', 'cifar100'):
        for variant in ('A0', 'A1', 'A2', 'A3', 'A4'):
            group = [r for r in rows if r['dataset'] == dataset and r['variant'] == variant]
            if len(group) != 3:
                raise ValueError(f'Missing ablation seeds: {dataset}/{variant}')
            for metric in ('final_validation_accuracy', 'policy_accuracy', 'corrected_policy_saving',
                           'parameters_total', 'parameters_exit_heads', 'training_wall_seconds'):
                values = [float(r[metric]) for r in group if r.get(metric) not in ('', None)]
                if len(values) == 3:
                    aggregates.append(dict(dataset=dataset, variant=variant, metric=metric, n=3,
                                           mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1))))
    write_csv(output / 'ablation_seed_tradeoffs.csv', rows)
    write_csv(output / 'ablation_aggregate_tradeoffs.csv', aggregates)
    save(output / 'ablation_scope.json', dict(status='completed', new_training_runs=0,
         note='All 30 A0-A4 seeds retained. Historical wall time is descriptive, not matched-hardware training speed. Training peak memory unavailable. No assertion that A3 dominates A4. Existing final/policy validation scopes remain distinct; this is not new official A3 evidence.'))


def latency_readme(rows):
    lines = ['# Independent P8-v3 latency replication', '',
             'Resident model-only FP32 timing, not end-to-end service latency. Positive saving is faster; negative is slowdown.',
             'Mean and sample SD use three training seeds, each averaging five paired rounds. No universal acceleration claim.', '',
             '| Device | Dataset | Batch | Saving (%) | Seed SD (pp) | Direction |',
             '|---|---|---:|---:|---:|---|']
    for row in rows:
        saving = row['actual_saving_percent_mean']
        direction = 'slowdown' if saving < 0 else 'positive point estimate' if saving > 0 else 'neutral'
        lines.append(f"| {row['device']} | {row['dataset']} | {row['batch_size']} | {saving:.3f} | {row['actual_saving_percent_sample_sd']:.3f} | {direction} |")
    return '\n'.join(lines) + '\n'


def latency_audit(output):
    from scripts.analysis.analyze_early_exit_p8_v3 import audit, aggregate, read_csv, write_figure, write_latex
    input_dir = output / 'latency'
    target = output / 'latency_report'
    target.mkdir(exist_ok=False)
    rows = read_csv(input_dir / 'tables/round_metrics.csv')
    receipt = audit(input_dir, rows)
    # Preserve audit failures too. This uses the historical stricter zero-argmax-error audit.
    save(target / 'analysis_receipt.json', receipt)
    if receipt['status'] != 'passed':
        raise RuntimeError(f"Latency audit failed: {receipt['errors']}")
    seeds, aggregates = aggregate(rows)
    write_csv(target / 'seed_summary.csv', seeds)
    write_csv(target / 'aggregate_summary.csv', aggregates)
    write_latex(target / 'aggregate_summary.tex', aggregates)
    write_figure(target, aggregates)
    (target / 'README.md').write_text(latency_readme(aggregates))


def run_stages(output, commands):
    state = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), stages=[])
    for name, command in commands:
        item = dict(name=name, command=command, status='running', started_at=datetime.now(timezone.utc).isoformat())
        state['stages'].append(item)
        save(output / 'batch_status.json', state)
        print(f'STAGE {name}: {command}', flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, check=False)
            item['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'Stage {name} exited {result.returncode}')
            item['status'] = 'completed'
        except BaseException as error:
            item.update(status='failed', error=str(error))
            state['status'] = 'failed'
            save(output / 'batch_status.json', state)
            raise
        item['finished_at'] = datetime.now(timezone.utc).isoformat()
    state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    save(output / 'batch_status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['comparison', 'tradeoffs', 'latency_audit'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'comparison': compare, 'tradeoffs': tradeoffs, 'latency_audit': latency_audit}[args.stage](args.output)


if __name__ == '__main__':
    main()
