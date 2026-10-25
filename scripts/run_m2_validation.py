#!/usr/bin/env python3
"""Three independent batches, resident model inside each geometry/session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def jobs(batches):
    return [dict(batch=batch, mode=mode, args=(['--full-service'] if mode=='full_continue'
            else ['--selected-fast-clocks', '--live-service']))
            for batch in batches for mode in ('full_continue', 'fast')]


FROZEN_KEYS=('checkpoint_provenance_revision','H','N','state_enabled','state_prompt_capacity',
             'rtc_beta','rtc_author_order','rtc_exact_flash_attention','rtc_fused_norm','source_packed_conditioning',
             'precision','flashrt_revision','checkpoint_profile','checkpoint_path','norm_stats_path')


def verified_service(directory,filename,config):
    directory=Path(directory).resolve()
    if directory.parent!=(Path(config['output_root'])/'M2').resolve():
        raise ValueError('Reused service must belong to M2')
    prior=json.loads((directory/'run.json').read_text())['config_snapshot']
    if any(prior.get(k)!=config.get(k) for k in FROZEN_KEYS):
        raise ValueError('Reused service configuration differs from current baseline')
    path=directory/filename
    if not path.exists():return False
    d=json.loads(path.read_text())
    if d.get('status')!='completed':return False
    if d.get('preflight_only') or not d.get('restored') or len(d.get('requests',[]))!=60:
        raise ValueError('Service completion/restore evidence incomplete')
    if len(d.get('conditions',[]))!=3 or any(c.get('services')!=20 or not c.get('restored')
            or c.get('batch_energy',{}).get('energy_j') is None
            or c.get('matched_hold',{}).get('energy',{}).get('energy_j') is None for c in d['conditions']):
        raise ValueError('Service conditions or long batch energy incomplete')
    seen=set()
    for r in d['requests']:
        key=(r['state_history'],r['index'])
        if key in seen or key[0] not in ('hot','short','long') or key[1] not in range(20) or r['profile']!='baseline':
            raise ValueError('Service request plan differs or contains duplicates')
        seen.add(key)
        if not r.get('frequency_check',{}).get('matches') or not r.get('frame_matches_reference') \
                or not r.get('state_matches_reference') or r.get('raw_action_max_abs_difference_vs_reference')!=0:
            raise ValueError('Service input/action/frequency checks incomplete')
        if r['requested_idle_seconds'] and not r.get('idle_frequency_check',{}).get('matches'):
            raise ValueError('Service idle clock checks incomplete')
    return True


def verified_continue(directory):
    path=Path(directory)/'continue_control.json'
    if not path.exists():return False
    d=json.loads(path.read_text())
    if d.get('status')!='completed':return False
    if len(d.get('windows',[]))!=60 or not d.get('all_final_states_equal') or d.get('preflight_only'):
        raise ValueError('Continue completion evidence incomplete')
    seen=set()
    for w in d['windows']:
        key=(w['requested_history_seconds'],w['repeat']);offset={0.:0,.2:3,1.:15}.get(key[0])
        if key in seen or key[1] not in range(20) or offset is None or w['profile']!='gpu_emc':
            raise ValueError('Continue window plan differs')
        seen.add(key)
        if (not w.get('restored') or not w['frequency_check']['matches'] or not w['refresh_frequency_check']['matches']
                or w['executed_steps']!=5 or w['deadline_overruns'] or w['committed_prefix_max_abs_error']!=0
                or [r['source_action_index'] for r in w['rows']]!=list(range(offset,offset+5))):
            raise ValueError('Continue correctness checks incomplete')
        for field in ('transient_continue','including_fresh_images'):
            power=w[field]
            if power['energy_j'] is None or power['power_coverage']['coverage_fraction']!=1 or power['power_coverage']['missing_count']:
                raise ValueError('Continue power coverage incomplete')
    return True


def resume_services(session,config,batches):
    session=Path(session).resolve()
    if session.parent!=(Path(config['output_root'])/'M2').resolve():
        raise ValueError('Resume source must be an M2 session')
    source=json.loads((session/'validation_index.json').read_text())
    if source.get('batches')!=list(batches):raise ValueError('Resume batch selection differs')
    out={}
    for job in source['jobs']:
        if not job.get('run'):continue
        if job['mode']=='full_continue':
            full=job.get('full_service_run',job['run'])
            if verified_service(full,'full_service.json',config):
                continuation=job.get('continue_run',job['run'])
                out[(job['batch'],job['mode'])]=dict(full_service_run=full,
                    continue_run=continuation if verified_continue(continuation) else None)
        elif job['mode']=='fast' and verified_service(job['run'],'selected_fast_service.json',config):
            out[(job['batch'],job['mode'])]=dict(run=job['run'])
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--all',action='store_true',help='Run all three independent batches sequentially')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume-session',type=Path,help='Reuse verified whole service results; retry incomplete Continue separately')
    args=parser.parse_args()
    batches=(1,2,3) if args.all else (args.batch,)
    if args.dry_run and not args.resume_session:
        print(json.dumps(dict(jobs=jobs(batches),model_loads=2*len(batches),
            service_requests=120*len(batches),continue_windows=60*len(batches),
            note='Full and Continue share one model; Fast geometry separate; no per-request reload')))
        return 0
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    config=load_config(repo_root=ROOT)
    reuse=resume_services(args.resume_session,config,batches) if args.resume_session else {}
    if args.dry_run:
        print(json.dumps(dict(jobs=jobs(batches),reuse=[dict(batch=k[0],mode=k[1],**v) for k,v in reuse.items()],
            full_service_rerun=not any(k[1]=='full_continue' for k in reuse),partial_continue_windows_merged=False)))
        return 0
    run=create_run(config,'M2','validation-session',repo_root=ROOT)
    print('index_dir='+str(run),flush=True)
    result=dict(status='running',jobs=[],batches=list(batches),planned_model_loads=2*len(batches),
                resume_session=None if args.resume_session is None else str(args.resume_session.resolve()))
    try:
        for job in jobs(batches):
            prior=reuse.get((job['batch'],job['mode']),{})
            if (job['mode']=='fast' and prior) or prior.get('continue_run'):
                row=dict(job,**prior)
                row.update(run=prior.get('run',prior.get('continue_run')),exit_code=0,reused_verified=True)
                result['jobs'].append(row)
                _write_json_atomically(run/'validation_index.json',result)
                print('REUSED '+str(job['batch'])+'/'+job['mode'],flush=True)
                continue
            subprocess.run(['sudo','-v'],check=True)
            label='batch'+str(job['batch'])+'_'+job['mode']
            log=run/(label+'.log')
            command=[config['python_executable'],'-u',str(ROOT/'scripts/profile_clock_screen.py'),
                     *job['args'],'--service-validation','--validation-batch',str(job['batch']),
                     '--service-profiles','baseline']
            if prior.get('full_service_run'):
                command=[config['python_executable'],'-u',str(ROOT/'scripts/profile_clock_screen.py'),
                    '--continue-history','--continue-validation','--validation-batch',str(job['batch'])]
                print('REUSING_FULL_SERVICE='+prior['full_service_run']+'; retry Continue only',flush=True)
            print('START '+label+' log='+str(log),flush=True)
            with log.open('w') as out:
                code=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT).returncode
            child_runs=[line.split('=',1)[1] for line in log.read_text().splitlines() if line.startswith('run_dir=')]
            child=Path(child_runs[-1]) if child_runs else None
            row=dict(job,log=str(log),exit_code=code,run=None if child is None else str(child))
            if prior.get('full_service_run'):
                row.update(full_service_run=prior['full_service_run'],continue_run=None if child is None else str(child),
                           full_service_reused=True)
            result['jobs'].append(row)
            _write_json_atomically(run/'validation_index.json',result)
            if code or child is None or json.loads((child/'summary.json').read_text())['status']!='completed':
                raise RuntimeError('Validation failed at '+label+'; inspect '+str(log)+' and child clock restoration')
            print('DONE '+label+' run='+str(child),flush=True)
        for batch in batches:
            full_job=next(r for r in result['jobs'] if r['batch']==batch and r['mode']=='full_continue')
            fast_job=next(r for r in result['jobs'] if r['batch']==batch and r['mode']=='fast')
            full=json.loads((Path(full_job.get('full_service_run',full_job['run']))/'full_service.json').read_text())
            fast=json.loads((Path(fast_job['run'])/'selected_fast_service.json').read_text())
            if any(full[k]!=fast[k] for k in ('initial_physics_sha256','old_target_sha256')):
                raise RuntimeError('Full/Fast calibration physics or old targets differ; do not compare costs')
            a=full['requests'][0]['measured'];b=fast['requests'][0]['measured']
            if a['frame_sha256']!=b['frame_sha256'] or a['raw_state']!=b['raw_state']:
                raise RuntimeError('Full/Fast live observations differ; not a paired cost table')
            fast_job['cross_mode_observation_pairing_passed']=True
        result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/'validation_index.json',result)
        meta=json.loads((run/'run.json').read_text());meta['status']=result['status']
        _write_json_atomically(run/'run.json',meta)
        print('M2_VALIDATION_STATUS='+result['status'],flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
