"""Start one detached, serial, development-only readiness batch; never train."""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.analysis.paper_readiness import AB, AC, load_cohorts, read_json, run_stages, save, sha

OUTPUT = ROOT / 'artifacts/analyses/paper_readiness_20260908_r1'
LOG = ROOT / 'artifacts/launcher_logs/paper_readiness_20260908_r1.log'


def preflight():
    from scripts.analysis.benchmark_early_exit_p8_v3 import validate_protocol
    protocol = validate_protocol()
    fixed = {
        AB / 'development_logits_manifest.json': '95887ceae1d23e488d36ef3b6ea03f19ce33458c456edb13ab874cf31a519ab5',
        AC / 'tables/variant_seed_metrics.csv': '9e6e93a96d1bfe2952bafc0a826099cc0146e6105fac57c5cc074bfc47409bdb',
    }
    for path, expected in fixed.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    cohorts = load_cohorts()
    subprocess.run([sys.executable, 'scripts/launch_early_exit_p8_v3.py', '--dry-run'], cwd=ROOT, check=True)
    import torch
    from image_classification.data.cifar import DATASET_SPECS
    for dataset in ('cifar10', 'cifar100'):
        # Check local train archive integrity; never download or open official test.
        DATASET_SPECS[dataset].dataset_class(root=ROOT / 'data', train=True, download=False)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != protocol['gpu']:
        raise RuntimeError('This batch requires RTX 3080 Ti')
    if len(os.sched_getaffinity(0)) < 12:
        raise RuntimeError('Fewer than 12 CPU threads available')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError('Need at least 3 GiB free for this non-training batch')
    own = os.getpid()
    for line in subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or int(parts[0]) == own or not parts[1].startswith('python'):
            continue
        if any(name in parts[2] for name in ('scripts/train.py', 'run_baselines.py', 'benchmark_early_exit_p8',
                                            'launch_paper_readiness.py', 'paper_readiness.py --stage')):
            raise RuntimeError(f'Conflicting process: {line}')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(s.strip().isdigit() and int(s.strip()) != own for s in active.splitlines()):
        raise RuntimeError(f'GPU is in use: {active.strip()}')
    return dict(status='ready', development_cache_files=sum(map(len, cohorts.values())),
                checkpoints_verified=6, cpu_threads=12, gpu=protocol['gpu'],
                disk_free_gib=shutil.disk_usage(ROOT).free / 1024**3, new_training_runs=0,
                official_test_accessed=False, raw_latency_files_expected=1200)


def worker():
    # The parent keeps the shared P8 lock open and passes that same fd to us.
    # Direct --worker invocation is refused without the inherited lock descriptor.
    fd = int(os.environ['PAPER_READINESS_LOCK_FD'])
    os.fstat(fd)
    from image_classification.training.provenance import source_fingerprint
    receipt = read_json(OUTPUT / 'launch_receipt.json')
    if source_fingerprint() != receipt['source_fingerprint']:
        raise RuntimeError('Code changed between launch and worker start')
    py = sys.executable
    commands = [
        ('fair_comparison', [py, 'scripts/analysis/paper_readiness.py', '--stage', 'comparison', '--output', str(OUTPUT)]),
        ('a3_tradeoffs', [py, 'scripts/analysis/paper_readiness.py', '--stage', 'tradeoffs', '--output', str(OUTPUT)]),
        ('latency_replication', [py, 'scripts/analysis/benchmark_early_exit_p8_v3.py', '--output', str(OUTPUT / 'latency')]),
        ('latency_audit', [py, 'scripts/analysis/paper_readiness.py', '--stage', 'latency_audit', '--output', str(OUTPUT)]),
    ]
    run_stages(OUTPUT, commands)
    if source_fingerprint() != receipt['source_fingerprint']:
        state = read_json(OUTPUT / 'batch_status.json')
        state.update(status='failed', error='Source changed during batch')
        save(OUTPUT / 'batch_status.json', state)
        raise RuntimeError(state['error'])
    save(OUTPUT / 'output_hashes.json', {str(p.relative_to(OUTPUT)): sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()})
    print(f'COMPLETED: {OUTPUT}; inspect all results including unfavorable cases.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    if OUTPUT.exists() or LOG.exists():
        raise RuntimeError(f'Batch already reserved; preserve and inspect {OUTPUT} and {LOG}. No automatic rerun.')
    with (ROOT / 'artifacts/early_exit_p8_launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = preflight()
        if args.dry_run:
            print(checked)
            return
        from image_classification.training.provenance import source_fingerprint
        OUTPUT.mkdir(parents=True, exist_ok=False)
        save(OUTPUT / 'launch_receipt.json', dict(preflight=checked, source_fingerprint=source_fingerprint(),
             protocol=dict(scope='post-hoc development-only, frozen before this execution', methods=['msp', 'entropy', 'margin'],
                           source_quantiles=1001, compute_targets=[.10, .20, .30], source_tolerance=.0025,
                           target_cross_method_tolerance=.005, primary_latency_scenario='RTX3080Ti CIFAR10 batch1 resident FP32',
                           latency_scope='model-only resident inputs; excludes preprocessing, transfer and service queueing',
                           cpu_threads=12, p7='frozen, no optimization', retain_all_seeds=True),
             git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)))
        LOG.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='12', MKL_NUM_THREADS='12',
                   OPENBLAS_NUM_THREADS='1', PAPER_READINESS_LOCK_FD=str(lock.fileno()))
        with LOG.open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                                     env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
        print(f'PID: {child.pid}\nLog: {LOG}\nOutput: {OUTPUT}\nSerial background execution; do not launch other GPU/CPU experiments.')


if __name__ == '__main__':
    main()
