"""One-time P7 official validation evaluation; --verify-only never reads test data."""
import argparse
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from image_classification.data.imagenet100 import _transforms
from image_classification.models import build_model
from image_classification.selection.early_exit import softmax_confidence
from scripts.analysis.analyze_early_exit_p0 import _config
from scripts.analysis.analyze_early_exit_p6_source import configure_numerics, feasible, route_metrics
from scripts.launch_early_exit_p7_target import sha256

LOCK = ROOT / 'reports/experiments/2026-09-07-early-exit-p7-target-analysis/official_test_lock.json'
OUT = ROOT / 'artifacts/analyses/early_exit_p7_official_test'


def verify():
    lock = json.loads(LOCK.read_text())
    for name, checksum in lock['files'].items():
        if sha256(ROOT / name) != checksum:
            raise ValueError(f'Frozen input changed: {name}')
    if lock['threshold'] != .85 or lock['threshold_candidates'] != 0:
        raise ValueError('Policy changed')
    if OUT.exists() or (ROOT / 'data/imagenet100/locked_test').exists():
        raise FileExistsError('Existing official-test output/access marker; inspect before proceeding')
    print('Verified frozen evaluator, target evidence and six checkpoints; official data not read.', flush=True)
    return lock


def prepare(lock):
    for name, checksum in lock['archives'].items():
        if sha256(Path(name)) != checksum:
            raise ValueError('Official archive changed')
    devkit = next(Path(p) for p in lock['archives'] if 'devkit' in p)
    archive = next(Path(p) for p in lock['archives'] if 'img_val' in p)
    with tarfile.open(devkit) as tar:
        meta = loadmat(io.BytesIO(tar.extractfile('ILSVRC2012_devkit_t12/data/meta.mat').read()), squeeze_me=True)['synsets']
        mapping = {int(row[0]): str(row[1]) for row in meta if int(row[4]) == 0}
        ids = [int(v) for v in tar.extractfile('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt').read().split()]
    if len(ids) != 50000:
        raise ValueError('Expected 50,000 official validation labels')
    chosen = {f'ILSVRC2012_val_{i:08d}.JPEG': mapping[label] for i, label in enumerate(ids, 1) if mapping[label] in lock['classes']}
    if Counter(chosen.values()) != Counter({c: 50 for c in lock['classes']}):
        raise ValueError('Official subset must have exactly 50 samples per frozen class')
    destination = ROOT / 'data/imagenet100/locked_test'
    destination.mkdir()
    inventory = []
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name not in chosen:
                continue
            if not member.isfile():
                raise ValueError('Expected a regular image member')
            path = destination / chosen[member.name] / member.name
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as output:
                output.write(tar.extractfile(member).read())
            inventory.append({'path': str(path.relative_to(destination)), 'sha256': sha256(path)})
    if len(inventory) != 5000:
        raise ValueError('Incomplete official subset')
    (OUT / 'image_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return destination


def execute(lock):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3080 Ti':
        raise RuntimeError('RTX 3080 Ti required')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'started.json').write_text(json.dumps({'lock_sha256': sha256(LOCK), 'official_test_access_authorized_by_command': True}) + '\n')
    dataset = ImageFolder(prepare(lock), transform=_transforms()[1])
    if dataset.classes != lock['classes'] or len(dataset) != 5000:
        raise ValueError('Class/sample boundary mismatch')
    loader = DataLoader(dataset, batch_size=128, num_workers=12, shuffle=False, pin_memory=True)
    configure_numerics()
    torch.set_num_threads(1)
    device = torch.device('cuda')
    manifest = json.loads((ROOT / lock['manifest']).read_text())
    records = {}
    for run in manifest['runs']:
        config = _config(run['resolved_config'])
        model = build_model(config).to(device)
        checkpoint = ROOT / 'artifacts/runs' / run['experiment_id'] / 'checkpoints/model_best.pth'
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        model.eval()
        labels, heads = [], None
        with torch.inference_mode():
            for x, y in loader:
                values = model(x.to(device))
                values = values if isinstance(values, tuple) else (values,)
                if heads is None:
                    heads = [[] for _ in values]
                for chunks, value in zip(heads, values):
                    chunks.append(value.cpu().numpy())
                labels.append(y.numpy())
        labels = np.concatenate(labels)
        values = [np.concatenate(chunks) for chunks in heads]
        if not all(np.isfinite(v).all() for v in values):
            raise ValueError('Non-finite test logits')
        variant = 'A0' if len(values) == 1 else 'A3'
        path = OUT / f"seed{run['seed']}_{variant}.npz"
        np.savez_compressed(path, labels=labels, **{f'head{i}': v for i,v in enumerate(values)})
        record = {'accuracy': float((values[0].argmax(1) == labels).mean()), 'logits_sha256': sha256(path)}
        if variant == 'A3':
            cost = lock['cost_model']
            record['policy'] = route_metrics({'labels': labels, 'final_logits': values[0], 'exit_logits': values[1], 'num_classes': 100,
                'exit_cost_fraction': cost['deployable_exit_path_macs_including_head']/cost['reference_final_path_macs'],
                'fallback_cost_fraction': cost['fallback_path_macs']/cost['reference_final_path_macs']}, softmax_confidence(values[1]) >= lock['threshold'])
            record['policy_gate_passed'] = feasible(record['policy'], lock['risk_budget'], lock['selection'])
        records[f"{run['seed']}/{variant}"] = record
        print(f"Completed seed {run['seed']} {variant}", flush=True)
        del model
    result = {'status': 'completed', 'official_test_accessed': True, 'threshold_candidates': 0, 'threshold': .85,
              'all_policy_gates_passed': all(v.get('policy_gate_passed', True) for v in records.values()),
              'lock_sha256': sha256(LOCK), 'records': records}
    (OUT / 'test_results.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'completed.json').write_text(json.dumps({'results_sha256': sha256(OUT / 'test_results.json')}) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--authorize-official-test', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and not args.authorize_official_test:
        parser.error('Explicit --authorize-official-test is required')
    lock = verify()
    if not args.verify_only:
        execute(lock)
