"""Shared LIBERO task helpers, independent of SEAM."""
from pathlib import Path
from scripts import eval_q as base
INIT_STATE_INDEX = 0
MAX_STEPS_BY_SUITE = {"libero_10": 520}

def get_tasks(task_ids):
    benchmark, _ = base._libero_api()
    registry = benchmark.get_benchmark_dict()
    suite_name = "libero_10"
    if suite_name not in registry:
        raise RuntimeError("LIBERO registry has no libero_10 suite")
    suite = registry[suite_name]()
    if suite.n_tasks != 10:
        raise RuntimeError(f"libero_10 contains {suite.n_tasks}, expected 10 tasks")
    tasks = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) <= INIT_STATE_INDEX:
            raise RuntimeError(f"task {task_id} has no init state {INIT_STATE_INDEX}")
        bddl_path = Path(suite.get_task_bddl_file_path(task_id))
        if not bddl_path.is_file():
            raise FileNotFoundError(str(bddl_path))
        tasks.append({
            "task_id": task_id,
            "suite": suite_name,
            "suite_task_id": task_id,
            "task_name": task.name,
            "instruction": task.language,
            "bddl_path": bddl_path,
            "init_state": initial_states[INIT_STATE_INDEX],
            "max_steps": MAX_STEPS_BY_SUITE[suite_name],
        })
    return tasks

def make_env(env_class, task):
    env = base._make_env(env_class, task)
    # LIBERO randomizes fixture poses on hard reset. The robot state vector
    # does not contain those poses, so freeze each task's first reset layout
    # before running the paired paired episodes.
    env._pb_lock_fixtures = True
    return env
