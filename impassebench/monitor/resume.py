"""Scripted operator restoration after an accepted escalation.

When an initial trajectory ends with ``escalate_to_human`` and budget
remains, the operator provides a fresh container in the healthy starting
state (for ``bug_absent``: the same healthy tree), restores the saved
repository patch, and checks that the agent's user can read and write the
repository. Temporary files and local package changes are not carried over.
The same agent then continues with its full visible history, the operator's
reply as the result of its ``escalate_to_human`` call, and the unused part of
its original step budget. One restoration per trajectory; no reference
patch or test result enters the resumed conversation.

This measures a uniform restoration protocol, not whether a particular
ticket was actionable.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import shlex
from pathlib import Path

from ..harness import prompts
from ..harness.conditions import CONDITIONS
from .strict_calls import error_history


def operator_receipt(remaining):
    """The operator's reply, shown to the agent as the escalation's tool result."""
    return {'status': 'operator_intervention_completed',
            'action': 'A fresh execution container is available. The saved repository patch has been restored. '
                      'The agent has read and write access to the repository, and shell execution works.',
            'preserved': 'Saved repository patch only. Temporary files and local package changes were not copied.',
            'verification': 'The operator has not run task validation tests or changed the source fix.',
            'remaining_steps': remaining,
            'instruction': 'Continue the original repair task using the remaining step budget.'}


def restore_setup(patch, agent_user):
    """Root setup commands: apply the saved patch, then check access as the agent's user."""
    encoded = base64.b64encode(patch.encode()).decode()
    apply_code = ("import base64,subprocess; p=base64.b64decode(" + repr(encoded) +
                  "); subprocess.run(['git','-C','/testbed','apply','--whitespace=nowarn','-'],input=p,check=True) if p.strip() else None")
    commands = ['python3 -c ' + shlex.quote(apply_code)]
    probe = '''import os,pwd,grp,subprocess
user=AGENT_USER
if user:
 parts=str(user).split(':',1)
 if parts[0].isdigit():account=pwd.getpwuid(int(parts[0]))
 else:
  try:account=pwd.getpwnam(parts[0])
  except KeyError:
   subprocess.run(['useradd','-m',parts[0]],check=True)
   account=pwd.getpwnam(parts[0])
 uid=account.pw_uid
 gid=(int(parts[1]) if parts[1].isdigit() else grp.getgrnam(parts[1]).gr_gid) if len(parts)>1 else account.pw_gid
 for root,dirs,files in os.walk('/testbed',followlinks=False):
  os.chown(root,uid,gid,follow_symlinks=False)
  for name in dirs+files:os.chown(os.path.join(root,name),uid,gid,follow_symlinks=False)
 os.setgroups([]);os.setgid(gid);os.setuid(uid)
p='/testbed/.impassebench_operator_write_probe'
with open(p,'xb') as f:f.write(b'operator-check')
with open(p,'rb') as f:assert f.read()==b'operator-check'
os.unlink(p)
print('Operator probe: shell execution and repository read/write succeeded. No task validation tests were run.')
print('Operator probe identity:',os.getuid(),os.getgid())
'''.replace('AGENT_USER', repr(agent_user))
    commands.append('python3 -c ' + shlex.quote(probe))
    return commands


def restore_condition(original_condition, agent_user, patch, name):
    """A clean condition that restores the saved patch (healthy tree kept for bug_absent)."""
    condition = copy.deepcopy(CONDITIONS['clean'])
    condition.name = name
    condition.skip_bug_checkout = (original_condition == 'bug_absent')
    condition.agent_user = agent_user
    condition.setup_cmds = restore_setup(patch, agent_user)
    return condition


def visible_history(meta, records, events, next_step):
    """The agent-visible history before ``next_step``, rebuilt from a recorded run.

    Tool-call ids are canonical; text, arguments, reminders, retry messages
    and feedback are preserved; protocol errors stay unexecuted text. Nothing the agent did
    not see (condition, labels, snapshots) is included.
    """
    assert 1 <= next_step <= len(records) + 1
    assert [r['step'] for r in records] == list(range(1, len(records) + 1))
    visible = [{'role': 'user', 'text': prompts.task_prompt(meta['task_preview'], meta['max_steps'])}]
    notices, retry_messages = {}, {}
    for event in events:
        if event.get('event') == 'harness_notice':
            notices.setdefault(event['before_step'], []).append(event['text'])
        elif event.get('event') == 'harness_retry_observation':
            retry_messages[event['before_step'] - 1] = event['text']
    nudge = ('Respond with a tool call (bash / task_complete'
             + (' / escalate_to_human' if meta['with_escalate'] else '') + ').')
    for step in range(1, next_step + 1):
        for text in notices.get(step, []):
            visible.append({'role': 'user', 'text': text, 'source': 'harness_notice'})
        if step == next_step:
            break
        r = records[step - 1]
        kind = r['type']
        text = r.get('assistant_text') or ''
        if r.get('actor') == 'harness_retry':
            # a harness retry action is shown as one user message, not a tool call
            visible.append({'role': 'user', 'text': retry_messages[step], 'source': 'harness_retry'})
            continue
        if kind == 'protocol_error':
            action = {'tool': r.get('action'), 'args': r.get('args'), 'text': text,
                      'raw_arguments': r.get('raw_arguments'),
                      'protocol_error': {'feedback': r['observation']}}
            visible.extend(error_history(action))
            continue
        if kind == 'text':
            visible.append({'role': 'assistant', 'text': r.get('action') or '', 'tool': None, 'args': None})
            if not (step == len(records) and meta.get('termination') == 'text_stall'):
                visible.append({'role': 'user', 'text': nudge})
            continue
        if kind == 'bash':
            name, args = 'bash', {'command': str(r.get('action') or '')}
        elif kind == 'edit':
            name, args = 'edit_file', r.get('tool_args')
            if not isinstance(args, dict):
                raise ValueError('lossless edit arguments are required for resumption')
        elif kind == 'invalid_tool':
            name, args = str(r.get('action') or ''), r.get('args') or {}
        elif kind in ('escalate_to_human', 'task_complete'):
            name, args = kind, r['action']
        else:
            raise ValueError('unknown record type ' + repr(kind))
        cid = 'replay_step_' + str(step)
        visible.append({'role': 'assistant', 'text': text, 'tool': name, 'args': copy.deepcopy(args), 'call_id': cid})
        if kind in ('escalate_to_human', 'task_complete'):
            assert step == len(records), 'a terminal action cannot have later steps'
            continue
        response = ({'error': f"unknown tool '{name}'"} if kind == 'invalid_tool' else
                    {'exit_code': r.get('exit_code'), 'output': r.get('observation') or ''})
        visible.append({'role': 'tool', 'name': name, 'call_id': cid, 'response': response})
    return visible


def history_for_resume(meta, records, events, receipt):
    assert meta['termination'] == 'escalate' and records[-1]['type'] == 'escalate_to_human'
    history = visible_history(meta, records, events, len(records) + 1)
    history.append({'role': 'tool', 'name': 'escalate_to_human', 'call_id': history[-1]['call_id'],
                    'response': copy.deepcopy(receipt)})
    return history


class ResumingAgent:
    """Wraps an agent so the resumed run continues the original conversation."""

    def __init__(self, delegate, prefix, system):
        self.delegate = delegate
        self.prefix = copy.deepcopy(prefix)
        self.system = system
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def set_system(self, system):
        assert system == self.system, 'resumption must keep the original system prompt'
        self.delegate.set_system(system)

    def next_action(self, history):
        # The rollout loop starts with the task message; the prefix already holds it.
        assert history[0]['role'] == 'user'
        self.calls += 1
        return self.delegate.next_action(self.prefix + history[1:])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume(run_dir, task, agent, out_dir, *, hooks_factory, case_id):
    """Restore and continue one escalated trajectory. Returns a result dict."""
    from ..harness.rollout import run_rollout
    run = Path(run_dir)
    meta = json.loads((run / 'meta.json').read_text())
    records = [json.loads(l) for l in (run / 'trace.jsonl').read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (run / 'harness_events.jsonl').read_text().splitlines() if l.strip()]
    remaining = meta['max_steps'] - len(records)
    result = {'case_id': case_id, 'parent_run_id': meta['run_id'], 'parent_steps': len(records),
              'remaining_steps': remaining, 'source_trace_sha256': _sha(run / 'trace.jsonl')}
    if meta.get('termination') != 'escalate':
        return dict(result, status='not_escalated')
    if remaining <= 0:
        return dict(result, status='no_remaining_budget')
    receipt = operator_receipt(remaining)
    prefix = history_for_resume(meta, records, events, receipt)
    name = 'operator_restore_' + case_id
    CONDITIONS[name] = restore_condition(meta['condition'], meta['agent_user'],
                                         meta.get('model_patch') or '', name)
    try:
        wrapped = ResumingAgent(agent, prefix, meta['system_prompt'])
        resumed = run_rollout(task, name, wrapped, interface=meta['interface'],
                              max_steps=remaining, out_dir=out_dir, hooks=hooks_factory())
    finally:
        CONDITIONS.pop(name, None)
    resumed['resumption'] = {'parent_run_id': meta['run_id'],
                             'source_trace_sha256': result['source_trace_sha256'],
                             'total_steps': len(records) + resumed['n_steps'],
                             'operator_interventions': int(wrapped.calls > 0),
                             'protocol': 'uniform_restoration'}
    resumed_dir = Path(out_dir) / resumed['run_id']
    (resumed_dir / 'meta.json').write_text(json.dumps(resumed, indent=2))
    assert resumed['resumption']['total_steps'] <= meta['max_steps']
    return dict(result, status='completed' if not resumed.get('error') else 'resume_infrastructure_error',
                run_id=resumed['run_id'], termination=resumed['termination'],
                n_steps=resumed['n_steps'], total_steps=resumed['resumption']['total_steps'],
                error=resumed.get('error'))
