#!/usr/bin/env python3
"""Finite root lease for one power-gating switch; no forced idle or clock writes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.gpu_railgate import state,set_enabled


def emit(status,**values):
    print(json.dumps(dict(status=status,**values)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    if os.geteuid()!=0:parser.error('Only the one-switch helper runs as root')
    os.umask(0o077)
    directory=Path('/run/perceptbudget-railgate');directory.mkdir(mode=0o700,exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid!=0 or directory.stat().st_mode&0o077:
        raise RuntimeError('Invalid root snapshot directory')
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=directory/'original.json'
        if args.restore:
            set_enabled(json.loads(snapshot.read_text())['railgate_enable'])
            snapshot.unlink();emit('RESTORED',actual=state());return
        if snapshot.exists():raise RuntimeError('Pending snapshot; run helper with --restore')
        saved=state()['railgate_enable']
        if saved not in ('0','1'):raise RuntimeError('Cannot read original railgate setting')
        with snapshot.open('x') as stream:
            json.dump({'railgate_enable':saved},stream);stream.flush();os.fsync(stream.fileno())
        def interrupted(signum,frame):raise KeyboardInterrupt(signum)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
        try:
            emit('READY',actual=state(),lease_seconds=180)
            end=time.monotonic()+180
            while time.monotonic()<end:
                ready,_,_=select.select([sys.stdin],[],[],min(.2,max(0.,end-time.monotonic())))
                if not ready:continue
                line=sys.stdin.readline()
                if not line or line.strip()=='STOP':break
                value=line.strip()
                try:
                    set_enabled(value)
                except OSError as exc:
                    emit('REJECTED',enabled=value,errno=exc.errno,error=str(exc),actual=state())
                    return
                emit('APPLIED',enabled=value,actual=state())
        finally:
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
            set_enabled(saved);snapshot.unlink();emit('RESTORED',actual=state())


if __name__=='__main__':main()
