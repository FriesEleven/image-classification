"""Audit P7 target runs and evaluate the source-frozen threshold without search."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _collect_logits
from scripts.analysis.analyze_early_exit_p6_source import configure_numerics, feasible, route_metrics
from scripts.launch_early_exit_p7_target import PROTOCOL, load_protocol, sha256, validated_plan


def execute(manifest_path, output, audit_only=False):
    protocol = load_protocol()
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'completed' and manifest['concurrent_jobs'] == 1
    expected = {r['experiment_id']: r for r in validated_plan()}
    runs = manifest['runs']
    assert len(runs) == 6 and {r['experiment_id'] for r in runs} == set(expected)
    receipts = []
    previous_end = None
    for run in runs:
        assert run['status'] == 'completed' and run['return_code'] == 0
        assert not run.get('termination_signal')
        assert run['resolved_config'] == expected[run['experiment_id']]['resolved_config']
        assert previous_end is None or run['started_at'] >= previous_end
        previous_end = run['finished_at']
        root = ROOT / 'artifacts/runs' / run['experiment_id']
        summary = json.loads((root / 'summary.json').read_text())
        assert summary == run['summary'] and summary['test_evaluated'] is False
        assert sha256(root / 'checkpoints/model_best.pth') == summary['best_checkpoint_sha256']
        assert sha256(root / 'split_indices.json') == summary['split_indices_sha256']
        rows = list(csv.DictReader((root / 'logs/training.csv').open()))
        assert [int(row['epoch']) for row in rows] == list(range(1, 101))
        evidence = ['summary.json', 'config.yaml', 'provenance.json', 'split_indices.json', 'logs/training.csv']
        evidence += ['checkpoints/' + name for name in ('model_best.pth', 'model_latest.pth', 'final.pth')]
        receipts.append({'run': run['experiment_id'], 'files': {name: sha256(root / name) for name in evidence}})
    audit = {'status': 'passed', 'manifest_sha256': sha256(manifest_path), 'runs': receipts,
             'protocol_sha256': sha256(PROTOCOL), 'official_test_accessed': False}
    if audit_only:
        print(json.dumps(audit, indent=2))
        return
    assert not output.exists(), 'Refusing to overwrite an existing evaluation'
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == protocol['target_training']['gpu']
    configure_numerics()
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lock = json.loads((ROOT / protocol['source_lock']['path']).read_text())
    cost = lock['cost_model']
    gate = protocol['target_evaluation_after_training']
    selection = {k: gate[k] for k in ('minimum_early_fraction_each_seed', 'maximum_early_fraction_each_seed', 'minimum_mac_saving_each_seed')}
    rows = []
    for seed in protocol['target_training']['seeds']:
        a0 = next(r for r in runs if r['seed'] == seed and r['resolved_config']['model_type'] == 'mobilenetv2')
        a3 = next(r for r in runs if r['seed'] == seed and r['resolved_config']['model_type'] == 'multi_exit')
        assert a0['summary']['split_indices_sha256'] == a3['summary']['split_indices_sha256']
        labels, values = _collect_logits(a3, torch.device('cuda'), split='calibration')
        assert len(values) == 3 and all(v.shape == (10000, 100) and np.isfinite(v).all() for v in values)
        assert labels.shape == (10000,) and np.all(np.bincount(labels, minlength=100) == 100)
        path = output / f'seed{seed}_logits.npz'
        np.savez_compressed(path, labels=labels, final_logits=values[0], exit8_logits=values[1], exit15_logits=values[2])
        record = {'labels': labels, 'final_logits': values[0], 'exit_logits': values[1], 'num_classes': 100,
                      'exit_cost_fraction': cost['deployable_exit_path_macs_including_head'] / cost['reference_final_path_macs'],
                      'fallback_cost_fraction': cost['fallback_path_macs'] / cost['reference_final_path_macs']}
        metrics = route_metrics(record, softmax_confidence(values[1]) >= gate['threshold'])
        row = dict(seed=seed, paired_final_gain=a3['summary']['best_validation_accuracy'] - a0['summary']['best_validation_accuracy'],
                   policy_gate_passed=feasible(metrics, gate['risk_budget'], selection), logits_sha256=sha256(path), **metrics)
        rows.append(row)
        print(json.dumps(row), flush=True)
    final_pass = np.mean([r['paired_final_gain'] for r in rows]) >= gate['final_head_gain_mean_minimum'] - 1e-12 and min(r['paired_final_gain'] for r in rows) >= gate['final_head_gain_each_seed_minimum'] - 1e-12
    result = {'status': 'ready_for_locked_test_design' if final_pass and all(r['policy_gate_passed'] for r in rows) else 'stop_without_test',
                  'final_head_gate_passed': bool(final_pass), 'threshold': gate['threshold'], 'threshold_candidates': 0,
                  'official_test_accessed': False, 'manifest_sha256': sha256(manifest_path), 'protocol_sha256': sha256(PROTOCOL), 'seeds': rows}
    (output / 'target_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    execute(args.manifest, args.output, args.audit_only)
