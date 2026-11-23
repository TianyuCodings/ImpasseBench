"""Run the monitor study on SWE-bench Verified (paper Section 4 and its appendix).

Two studies share the 102 main tasks (``resources/verified_cases.json``):

* ``reminder`` -- clean (149 tasks) plus six persistent barriers. Arm ``off``
  runs the monitor silently, arm ``soft`` shows its reminders and ticket
  template. An accepted escalation is followed by one scripted operator
  restoration (:mod:`.resume`).
* ``retry`` -- the six remaining conditions. Arm ``soft`` adds bounded
  automatic retry; no restoration.

Both arms use Qwen3-Coder-30B-A3B on Bedrock with the escalate interface and
an 80-step budget. Example::

    python -m impassebench.monitor.experiment run --study reminder --arm soft \\
        --shard 0/3 --out runs/monitor
    python -m impassebench.monitor.experiment grade --study reminder --out runs/monitor
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .grade import atomic, grade, load_oracle, read
from .hooks import MonitorHooks

RESOURCES = Path(__file__).resolve().parent / 'resources'
MAX_STEPS = 80
RETRY = dict(retry_max_attempts=3, retry_backoff_s=1.0)


def load_study(study):
    cases = read(RESOURCES / 'verified_cases.json')[study]
    tasks = {t['instance_id']: t for t in read(RESOURCES / 'verified_tasks.json')['tasks']}
    return cases, tasks


def shard_filter(cases, tasks, shard):
    """All cases of a task land on the same shard, so its images are built once."""
    index, count = (int(x) for x in shard.split('/'))
    assert 0 <= index < count
    order = {iid: i for i, iid in enumerate(tasks)}
    return [c for c in cases if order[c['instance_id']] % count == index]


def hooks_for(study, arm):
    arms = read(RESOURCES / 'verified_cases.json')[study]['arms'][arm]
    return lambda: MonitorHooks(monitor_mode=arms['monitor'], retry_mode=arms['retry'], **RETRY)


def run_case(case, task, arm_dir, study, arm, make_agent):
    from ..harness.rollout import run_rollout
    from .resume import resume
    path = arm_dir / 'results' / (case['case_id'] + '.json')
    if path.exists():
        return read(path)
    result = dict(case, arm=arm, status='error')
    try:
        meta = run_rollout(task, case['condition'], make_agent(), interface='escalate',
                           max_steps=MAX_STEPS, out_dir=arm_dir / 'runs', hooks=hooks_for(study, arm)())
        result.update(status='completed', run_id=meta['run_id'], termination=meta['termination'],
                      n_steps=meta['n_steps'], error=meta.get('error'))
        cfg = read(RESOURCES / 'verified_cases.json')[study]
        if (cfg['operator_restoration'] and meta['termination'] == 'escalate'
                and not meta.get('error') and meta.get('artifact_observer_ok')):
            result['restoration'] = resume(arm_dir / 'runs' / meta['run_id'], task, make_agent(),
                                           arm_dir / 'restored', hooks_factory=hooks_for(study, arm),
                                           case_id=case['case_id'])
    except Exception as exc:
        result['error'] = repr(exc)
    atomic(path, result)
    return result


def cmd_run(args):
    from .qwen_bedrock import QwenConverseAgent
    cases, tasks = load_study(args.study)
    cases = shard_filter(cases['cases'], tasks, args.shard)
    oracle = {r['instance_id']: r for r in load_oracle()}
    arms = ['off', 'soft'] if args.arm == 'both' else [args.arm]
    for arm in arms:
        arm_dir = args.out / args.study / arm
        (arm_dir / 'results').mkdir(parents=True, exist_ok=True)
        jobs = []
        for case in cases:
            task = dict(tasks[case['instance_id']])
            task.setdefault('patch', oracle[task['instance_id']]['patch'])
            jobs.append((case, task))
        with ThreadPoolExecutor(args.parallel) as pool:
            futures = [pool.submit(run_case, c, t, arm_dir, args.study, arm, QwenConverseAgent) for c, t in jobs]
            for n, f in enumerate(as_completed(futures), 1):
                r = f.result()
                print(f"[{arm} {n}/{len(jobs)}] {r['instance_id']} {r['condition']} -> "
                      f"{r.get('termination') or r['status']}", flush=True)
    return 0


def cmd_grade(args):
    cases, _ = load_study(args.study)
    by_instance = {}
    for arm in ('off', 'soft'):
        arm_dir = args.out / args.study / arm
        for case in cases['cases']:
            rp = arm_dir / 'results' / (case['case_id'] + '.json')
            if not rp.exists():
                continue
            r = read(rp)
            if r.get('status') != 'completed':
                continue
            job = dict(case_id=case['case_id'], instance_id=case['instance_id'],
                       condition=case['condition'], arm=arm, phase='original',
                       run_dir=str(arm_dir / 'runs' / r['run_id']))
            by_instance.setdefault((arm, 'original', case['instance_id']), []).append(job)
            rr = r.get('restoration') or {}
            if rr.get('status') == 'completed':
                by_instance.setdefault((arm, 'resumed', case['instance_id']), []).append(
                    dict(job, phase='resumed', parent_run_dir=job['run_dir'],
                         run_dir=str(arm_dir / 'restored' / rr['run_id'])))
    outcomes = []
    for (arm, phase, iid), jobs in sorted(by_instance.items()):
        for i, job in enumerate(jobs):  # one patch per instance per grading batch
            target = args.out / args.study / 'grades' / arm / phase / f"{job['case_id']}"
            if (target / 'outcomes.json').exists():
                outcomes += read(target / 'outcomes.json')
                continue
            outcomes += grade([job], target, test_reset=(args.study == 'retry'), parallel=1)
    atomic(args.out / args.study / 'GRADES.json', outcomes)
    print(json.dumps({'graded': len(outcomes), 'resolved': sum(o['resolved'] is True for o in outcomes),
                      'unknown': sum(o['resolved'] is None for o in outcomes)}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run trajectories (and restorations) for one shard')
    r.add_argument('--study', choices=['reminder', 'retry'], required=True)
    r.add_argument('--arm', choices=['off', 'soft', 'both'], default='both')
    r.add_argument('--shard', default='0/1', help='i/n: run every n-th task starting at i')
    r.add_argument('--parallel', type=int, default=4)
    r.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('grade', help='grade finished trajectories with SWE-bench')
    g.add_argument('--study', choices=['reminder', 'retry'], required=True)
    g.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == 'run' else cmd_grade(args)


if __name__ == '__main__':
    sys.exit(main())
