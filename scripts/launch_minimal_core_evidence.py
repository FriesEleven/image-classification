"""One detached serial M1/M2/M3 batch after frozen M0 input preparation."""
import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from scripts.analysis.minimal_core_evidence import OUT, verify_inputs, now
from scripts.analysis.paper_readiness import read_json,save,sha,run_stages

LOG=ROOT/'artifacts/launcher_logs/minimal_core_evidence_20260918_r1.log'


def verify_stage_files():
    if (OUT/'m1_completed.json').exists():
        for name,checksum in read_json(OUT/'m1_completed.json')['files'].items():
            if sha(OUT/name)!=checksum:raise ValueError('M1 output changed')


def conflicts():
    import torch
    import shutil
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3080 Ti':
        raise RuntimeError('RTX3080Ti required')
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need >=2GiB free')
    names=['scripts/train.py','run_baselines.py','benchmark_early_exit_p8','launch_paper_readiness.py',
           'minimal_core_evidence.py --stage','launch_minimal_core_evidence.py']
    for line in subprocess.check_output(['ps','-eo','pid=,comm=,args='],text=True).splitlines():
        parts=line.strip().split(None,2)
        if len(parts)==3 and int(parts[0])!=os.getpid() and parts[1].startswith('python') and any(name in parts[2] for name in names):
            raise RuntimeError(f'Conflicting experiment: {line}')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(s.strip().isdigit() and int(s.strip())!=os.getpid() for s in active.splitlines()):
        raise RuntimeError(f'GPU is busy: {active}')


def worker():
    fd=int(os.environ['MINIMAL_CORE_LOCK_FD']);os.fstat(fd)
    verify_inputs(OUT)
    verify_stage_files()
    launch=read_json(OUT/'launch_receipt.json')
    if sha(OUT/'protocol_manifest.json')!=launch['protocol_sha256']:raise ValueError('Protocol changed')
    py=sys.executable;script='scripts/analysis/minimal_core_evidence.py'
    commands=[]
    if not (OUT/'m1_completed.json').exists():commands.append(('M1_shared_policy',[py,script,'--stage','m1','--output',str(OUT)]))
    commands += [('M2_locked_A3',[py,script,'--stage','m2','--output',str(OUT),'--authorize-a3-cifar-official-test']),
                 ('M3_audit',[py,script,'--stage','m3','--output',str(OUT)])]
    run_stages(OUT,commands)
    save(OUT/'output_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_hashes.json'})
    destination=ROOT/'reports/experiments/2026-09-18-minimal-core-evidence'
    destination.mkdir(parents=True,exist_ok=False)
    import shutil
    files=['protocol_manifest.json','input_manifest.json','policy_selection.json','shared_vs_individual_seed_metrics.csv',
           'shared_vs_individual_summary.csv','m1_per_class_counts.csv','a3_accuracy_mac_seed_metrics.csv',
           'a3_latency_round_metrics.csv','a3_joint_operating_points.csv','a3_joint_summary.csv','m2_per_class_counts.csv',
           'analysis_receipt.json','output_hashes.json','README.md','m2_correctness.json','batch_status.json',
           'shared_policy_tradeoffs.pdf','a3_joint_summary.tex']
    for name in files:shutil.copy2(OUT/name,destination/name)
    print(f'COMPLETED; original: {OUT}; compact report: {destination}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--authorize-a3-cifar-official-test',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:worker();return
    if not a.dry_run and not a.authorize_a3_cifar_official_test:p.error('Explicit approved official assessment flag required')
    if not OUT.exists():raise FileNotFoundError('Run M0 --stage prepare first')
    if LOG.exists() or (OUT/'launch_receipt.json').exists() or (OUT/'official_started.json').exists() or (ROOT/'artifacts/minimal_core_official_access_20260918_r1.json').exists():
        raise FileExistsError('Batch already launched/accessed; inspect, do not automatically retry')
    with (ROOT/'artifacts/early_exit_p8_launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        conflicts();verify_inputs(OUT);verify_stage_files()
        if a.dry_run:
            print(dict(status='ready',checkpoints=6,caches=15,new_training_runs=0,official_images_not_loaded=True,
                m1_completed=(OUT/'m1_completed.json').exists(),rounds_per_seed=5,raw_timing_files=60));return
        save(OUT/'launch_receipt.json',dict(started_at=now(),protocol_sha256=sha(OUT/'protocol_manifest.json'),
            user_approved_official_assessment=True,output=str(OUT.relative_to(ROOT)),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
        LOG.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,MINIMAL_CORE_LOCK_FD=str(lock.fileno()),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with LOG.open('x') as handle:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=env,
                stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lock.fileno(),))
        print(f'PID: {child.pid}\nLog: {LOG}\nOutput: {OUT}\nOne serial batch; no training. Keep all scientific failures.')


if __name__=='__main__':main()
