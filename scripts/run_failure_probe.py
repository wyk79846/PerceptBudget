"""Collect resumable saved features, then train the isolated failure probe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.analyze_gripper_timing import save


def oom_count():
    return int(next(line.split()[1] for line in Path('/proc/vmstat').read_text().splitlines()
        if line.startswith('oom_kill ')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--collect-only',action='store_true',help='Stop after extraction; do not run the legacy pooled-feature trainer')
    parser.add_argument('--train-safe-paper',action='store_true',help='Use the official SAFE endpoint training recipe after collection')
    args=parser.parse_args()
    if args.collect_only and args.train_safe_paper:
        parser.error('--collect-only and --train-safe-paper cannot be combined')
    root=args.run_dir
    attempt=0
    while True:
        attempt+=1
        before=oom_count()
        save(root/'workflow.json',dict(status='collecting',attempt=attempt))
        result=subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/collect_safe_features.py'),
            '--run-dir',str(root)],cwd=PROJECT)
        if result.returncode==0:
            break
        progress_path=root/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        error=progress.get('error','')
        resource_error=any(word in error.lower() for word in
            ('available ram below600mib','outofmemoryerror','cuda out of memory'))
        resource_error|=result.returncode==-9 and oom_count()>before
        if not resource_error:
            save(root/'workflow.json',dict(status='invalid',stage='extraction',attempt=attempt,
                exit_code=result.returncode,error=error))
            raise SystemExit(result.returncode or 1)
        save(root/'workflow.json',dict(status='waiting_for_ram',attempt=attempt,error=error,
            note='Completed features retained; only unfinished records will resume'))
        print('RESOURCE_RESUME='+json.dumps(dict(attempt=attempt,error=error)),flush=True)
        time.sleep(30)
    if args.collect_only:
        save(root/'workflow.json',dict(status='completed',stage='feature_extraction',extraction_attempts=attempt))
        return
    save(root/'workflow.json',dict(status='training',extraction_attempts=attempt))
    command=([sys.executable,'-u',str(PROJECT/'scripts/train_safe_paper.py'),'--features',str(root)]
        if args.train_safe_paper else
        [sys.executable,'-u',str(PROJECT/'scripts/train_failure_probe.py'),'--run-dir',str(root)])
    result=subprocess.run(command,cwd=PROJECT)
    save(root/'workflow.json',dict(status='completed' if result.returncode==0 else 'invalid',
        stage='training',extraction_attempts=attempt,exit_code=result.returncode))
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
