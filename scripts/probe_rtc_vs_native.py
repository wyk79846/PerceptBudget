"""Same-input N5/state timing diagnosis; no rollout or weight updates."""
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from scripts.probe_rtc_step import timed
from perceptbudget.experiment import create_run, load_config


def main():
    config = load_config(repo_root=ROOT)
    config.update(H=20, N=5, state_enabled=True)
    config['screen'].update(seed=7, init_state_index=0, suites=['libero_spatial'])
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'rtc-vs-native-timing', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                          PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        base._patch_egl_cleanup()
        torch, _ = base._check_cuda_and_native(config)
        torch.use_deterministic_algorithms(True)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.rtc_rollout import RTCGuidanceAdapter
        from flash_rt.subgraphs.pi05.context_action import enable
        model = base._load_and_check_model(config, torch)
        enable(model)
        task = base._load_tasks([0], ['libero_spatial'])[0]
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        try:
            obs, _ = base._prepare_env_episode(env, task, np)
            stats = load_libero_norm_stats(config['norm_stats_path'])
            snap = base._make_snapshot(obs, stats, True)
            def native():
                return base._infer_actions(model, snap, task['instruction'], 20, {})
            native()
            result = {'scope': 'same-input inference only; no async rollout', 'H':20, 'N':5,
                      'state':True, 'native_full':timed(torch, native)}
            pipe, pl = model._pipe, model._pipe.pipeline
            def context():
                with torch.cuda.stream(pipe._graph_torch_stream):
                    pl._context_graph.replay(pl._graph_stream)
                pipe._graph_torch_stream.synchronize()
            result['context_graph_only'] = timed(torch, context)
            adapter = RTCGuidanceAdapter(model, torch, horizon=20, num_steps=5,
                attention_backend='flash-varlen', audit_reference=False)
            previous = pipe._noise_out.float().cpu().numpy().copy()[5:]
            def rtc():
                return adapter.predict(snap, task['instruction'], previous,
                                       delay=5, executed=5, request={})
            rtc()
            result['rtc_full'] = timed(torch, rtc)
            result['rtc_expert_graph'] = timed(torch, adapter.graph.graph.replay)
            # Same Torch expert without input-gradient guidance: isolate the
            # VJP increment from the native-vs-Torch implementation difference.
            noise = torch.randn(20, 32, device='cuda', dtype=torch.bfloat16).float()
            def forward_only():
                with torch.no_grad():
                    return adapter.expert.forward(noise)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward_only()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward_only()
            result['torch_expert_forward_only_graph'] = timed(torch, graph.replay)
            result['forward_output_finite'] = bool(torch.isfinite(output).all())
            result['status'] = 'completed'
            base._write_json_atomic(run/'summary.json', result)
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
