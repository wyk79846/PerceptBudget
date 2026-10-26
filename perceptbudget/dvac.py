"""Small read-only DVAC signal probe on the existing source-step RTC graph."""
from collections import deque
import hashlib
import json
from pathlib import Path
import statistics
import time


def variance_scores(clean, tail=5):
    import numpy as np
    x = np.asarray(clean, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 7 or not np.isfinite(x).all():
        raise ValueError('Require finite denoising_steps x horizon x7 clean estimates')
    if type(tail) is not int or not 2 <= tail <= x.shape[0]:
        raise ValueError('Tail must contain at least two available denoising estimates')
    return x[-tail:].var(axis=0, ddof=0).sum(axis=1)


def prefix_decision(scores, history, *, expired=0, alpha=2.):
    """Threshold from prior snapshots only; first snapshot initializes history."""
    import numpy as np
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError('Require finite nonnegative per-action variances')
    if type(expired) is not int or not 0 <= expired <= len(scores) or alpha<0:
        raise ValueError('Invalid consumed prefix or alpha')
    if not history:
        return dict(threshold=None,prefix_actions=None,remaining_actions=None,warmup=True)
    reference = np.concatenate([np.asarray(x,dtype=np.float64) for x in history])
    threshold = float(reference.mean()+alpha*reference.std(ddof=0))
    crossings = np.flatnonzero(scores>threshold)
    prefix = max(1,int(crossings[0])) if len(crossings) else len(scores)
    return dict(threshold=threshold,prefix_actions=prefix,
                remaining_actions=max(0,prefix-expired),warmup=False)


def probe(adapter, source_run, directory, task, stats, torch, *, metadata):
    import gc
    import numpy as np
    from perceptbudget.h50_checkpoint import image_input, normalize_state
    from perceptbudget.rtc_memory import memory_record
    from perceptbudget.experiment import _write_json_atomically
    source_run,directory = Path(source_run),Path(directory)
    requests_path = source_run/'rtc/q5_requests.json'
    requests = json.loads(requests_path.read_text())
    selected = [r for r in requests if r.get('observation_step') and 'actual_delay_steps' in r.get('rtc_guidance',{})]
    if len(selected) < 5:
        raise ValueError('DVAC probe requires five committed guided source requests')
    indices = np.linspace(0,len(selected)-1,5,dtype=int)
    selected = [selected[i] for i in indices]
    spool = source_run/'rtc/q5_frames_spool.npy'
    if spool.exists():
        frames = np.load(spool,mmap_mode='r',allow_pickle=False)
    else:
        with np.load(source_run/'rtc/q5_frames.npz') as saved:
            frames = saved['frames']
    cases=[]
    for r in selected:
        frame=frames[r['observation_step']]
        state=np.asarray(r['observation_raw_state'],dtype=np.float32)
        snapshot=dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],
                      raw_state=state,state=normalize_state(state,stats))
        valid=adapter._prepare_context(snapshot,task['instruction'],{},guided=True)
        if valid!=r['rtc_guidance']['valid_prefix_length']:
            raise RuntimeError('Reconstructed source context length differs')
        cases.append((r,snapshot,valid))
    adapter.step_bank=None
    gc.collect();torch.cuda.empty_cache()
    graph_audit=adapter.prepare_step_bank(sorted({c[2] for c in cases}))
    mem=memory_record(torch,'dvac_probe_ready')
    if mem['available_mib']<900:
        raise RuntimeError('DVAC probe RAM headroom below900MiB')
    result=dict(status='running',mode='dvac_probe',source_run=str(source_run),model_loads=1,
        expert_loads=1,H=50,N=5,tail_lengths=[3,5],alpha=2.,rolling_snapshots=5,
        graph_audit=graph_audit,memory=mem,rows=[],policy_changed=False,clocks_changed=False,
        action_dimensions='First7 model-normalized action dimensions; excludes25 padded dimensions',
        clean_definition='Pre-guidance clean=x+(1-tau)*(-expert_velocity); input x follows RTC-guided trajectory',
        beta0_scope='Diagnostic zero guidance gain on same graph; VJP is still computed; not a faster solver',
        timing_scope='Fixed context expert/RTC replay, no VLM; record-on includes GPU copy, host transfer and variance/threshold calculation',
        paper='https://arxiv.org/html/2606.03847v1',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (source_run/'run.json',requests_path)},source_frame_storage=str(spool if spool.exists() else source_run/'rtc/q5_frames.npz'))
    histories={(beta,tail):deque(maxlen=5) for beta in (5.,0.) for tail in (3,5)}
    curves=[]
    try:
        for case_index,(source,snapshot,valid) in enumerate(cases):
            adapter._prepare_context(snapshot,task['instruction'],{},guided=True)
            expert,graph=adapter.step_bank.select(valid)
            expert.refresh_context(adapter.pipeline.attn.enc_K,adapter.pipeline.attn.enc_V,valid)
            solver=source['rtc_solver_inputs']
            noise=torch.as_tensor(solver['noise'],device='cuda',dtype=torch.float32)
            previous=torch.as_tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
            delay,executed=solver['delay'],solver['executed']
            expired=source['rtc_guidance']['actual_delay_steps']
            for beta in (5.,0.):
                graph.beta.fill_(beta)
                graph.replay(noise,previous,delay,executed)
                torch.cuda.synchronize()
                timings=[]
                last_clean=None
                scores={}
                decisions={}
                for repeat in range(3):
                    measurements={}
                    raw_off=raw_on=None
                    # Alternate ordering to reduce systematic warm/cold bias.
                    for record_on in ((False,True) if repeat%2==0 else (True,False)):
                        torch.cuda.synchronize();start=time.perf_counter()
                        clean=[] if record_on else None
                        raw,norms=graph.replay(noise,previous,delay,executed,clean_trace=clean)
                        if record_on:
                            values=torch.stack(clean).cpu().numpy()
                            for tail in (3,5):
                                scores[tail]=variance_scores(values,tail)
                                decisions[tail]=prefix_decision(scores[tail],histories[beta,tail],expired=expired)
                        torch.cuda.synchronize()
                        measurements['record_on_ms' if record_on else 'record_off_ms']=(time.perf_counter()-start)*1000
                        saved=raw.detach().cpu().numpy().copy()
                        if record_on:
                            raw_on=saved;last_clean=values.copy()
                        else:raw_off=saved
                    delta=float(np.max(np.abs(raw_on-raw_off)))
                    if delta!=0:
                        raise RuntimeError('DVAC recording changed final actions')
                    timings.append(dict(repeat=repeat,raw_max_abs=delta,**measurements))
                row=dict(case=case_index,source_prediction_id=source['prediction_id'],
                    observation_step=source['observation_step'],source_commit_step=source.get('ready_observed_step'),
                    valid_prefix=valid,beta=beta,delay=delay,executed=executed,expired=expired,
                    timings=timings,results={})
                for tail in (3,5):
                    v=scores[tail]
                    row['results'][str(tail)]=dict(**decisions[tail],
                        mean_variance=float(v.mean()),max_variance=float(v.max()),
                        expired_prefix_mean=float(v[:expired].mean()) if expired else None,
                        strong_prefix_mean=float(v[:delay].mean()) if delay else None,
                        available_tail_mean=float(v[max(delay,expired):].mean()),scores=v.tolist())
                    histories[beta,tail].append(v.copy())
                curves.append(last_clean)
                result['rows'].append(row)
                _write_json_atomically(directory/'dvac_probe.json',result)
                print('DVAC_CASE='+json.dumps(dict(case=case_index,beta=beta,step=row['observation_step'],
                    med_off_ms=statistics.median(x['record_off_ms'] for x in timings),
                    med_on_ms=statistics.median(x['record_on_ms'] for x in timings),
                    candidate_prefix={t:r['prefix_actions'] for t,r in row['results'].items()})),flush=True)
                graph.beta.fill_(5.)
        np.savez_compressed(directory/'clean_estimates.npz',clean=np.stack(curves),
            case=[r['case'] for r in result['rows']],beta=[r['beta'] for r in result['rows']])
        normal=[t for r in result['rows'] if r['beta']==5. for t in r['timings']]
        result.update(status='completed',action_parity_passed=True,
            median_record_off_ms=statistics.median(r['record_off_ms'] for r in normal),
            median_record_on_ms=statistics.median(r['record_on_ms'] for r in normal),
            median_paired_overhead_ms=statistics.median(r['record_on_ms']-r['record_off_ms'] for r in normal),
            scope='Five saved inputs from one successful Spatial0 episode; no new rollout or scheduling changes')
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for graph in adapter.step_bank.entries.values():
            # Entries retain (bound_expert, graph) pairs in the current bank.
            graph = graph[1] if isinstance(graph,tuple) else graph
            graph.beta.fill_(5.)
        _write_json_atomically(directory/'dvac_probe.json',result)
        _write_json_atomically(directory/'summary.json',result)
    return result
