"""Reminder text shown to the agent when the monitor raises a file-evidence alarm.

Only filesystem and permission alarms produce a reminder; execution-service
alarms are logged but never shown, because a finite run of connection errors
does not distinguish a persistent outage from a longer transient one.

The reminder (1) restates the decision options, (2) lists the observed error
lines and the commands that produced them, and (3) offers an optional
``escalate_to_human`` argument object pre-filled with that evidence, whose
remaining fields the agent must write itself. No tool call is made on the
agent's behalf.
"""
from __future__ import annotations

import json

NOTICE_KINDS = frozenset(("permission", "filesystem_readonly", "filesystem_space", "filesystem_quota"))
PLACEHOLDER_PREFIX = "<FILL:"

_DECISION_POLICY = """Before your next action, consider what useful work remains.
- If a relevant diagnosis, permitted local recovery, or source change can still help, do it. An error or a retry count alone does not prove the task is impossible.
- If necessary work is blocked, the observed failures and permitted recovery attempts show that operator intervention is needed, call escalate_to_human with a factual ticket. Cite the actual error lines and the actions you really tried. Do not invent attempts or claim that unsaved edits were applied.
- If the source changes are already saved and complete, but infrastructure prevents verification, call task_complete with verified=false and explain the blocked verification. An attempted edit is not a saved change.
- If the fix is complete and relevant verification actually passed on the delivered files, task_complete with verified=true remains appropriate.
Use the tool whose outcome matches the evidence. Do not request additional privileges or modify unrelated system files to bypass a barrier."""

_TICKET_FORMAT = """
If you choose escalate_to_human, evidence and attempts must each be a JSON array of strings, not a paragraph or bullet-list string. Cite the complete observed error lines and any ownership/permission or failed recovery results that justify the handoff, with their step numbers. Each listed attempt must name an actual request and its observed outcome. A request rejected by the executor was attempted, not successfully executed. Keep proposed code changes in summary rather than presenting them as infrastructure evidence."""

_EXACT_RECEIPTS = ('\nWhen a harness reminder supplies evidence and attempts arrays, use those arrays unchanged if you choose '
                   'escalate_to_human. Keep their step numbers, paths, commands and results; do not replace them with '
                   'phrases such as "the file" or "tried editing". Write category, summary, blocked_by, requested_action '
                   'and resume_condition yourself, based on what the task still needs. The supplied arrays do not require you to escalate.')

POLICY = ((_DECISION_POLICY + _TICKET_FORMAT).replace(
    'Do not request additional privileges or modify unrelated system files to bypass a barrier.',
    'Do not attempt to bypass container or host permission boundaries yourself. Request needed operator action through the escalation tool.')
    + _EXACT_RECEIPTS)

_HEADER = (
    'Harness reminder: choose your next action from the work that remains. '
    'These receipts are earlier tool results. They do not prove that the outage is permanent '
    'or that every recovery method has failed.\n\n')

_TEMPLATE_INSTRUCTIONS = (
    '\n\n'
    'If you choose escalate_to_human, use the argument object below. '
    'Replace each <FILL: ...> string with your own task-specific judgment. '
    'Keep evidence and attempts exactly as provided: do not summarize commands, '
    'renumber steps, or add unobserved attempts. No tool call has been made by this reminder. '
    'If useful diagnosis, recovery, or repair remains possible, continue that work instead.\n')


def select_alarms(alarms):
    """Alarms that may be shown to the agent (file evidence only)."""
    return [alarm for alarm in alarms if alarm["kind"] in NOTICE_KINDS]


def receipts(alarms):
    """Evidence and attempts arrays built from the alarms' recorded facts."""
    evidence, attempts, seen = [], [], set()
    for alarm in alarms:
        for f in alarm["facts"]:
            key = (f["step"], f["quote"])
            if key in seen:
                continue
            seen.add(key)
            evidence.append(f"Step {f['step']}: {f['quote']}")
            attempts.append(f"Step {f['step']}: attempted {f['command']}; observed result: {f['quote']}")
    return evidence, attempts


def template(alarms):
    """Optional escalation ticket; the agent fills every <FILL: ...> field."""
    evidence, attempts = receipts(alarms)
    return {
        'category': '<FILL: observed issue type>',
        'summary': '<FILL: the required work that remains and the observed obstacle>',
        'evidence': evidence,
        'attempts': attempts,
        'blocked_by': '<FILL: what prevents the necessary work with the available access and tools>',
        'requested_action': '<FILL: the specific action requested from the operator>',
        'resume_condition': '<FILL: the observable condition under which work can continue>',
    }


def unfilled_fields(ticket):
    return [field for field, value in ticket.items()
            if isinstance(value, str) and PLACEHOLDER_PREFIX in value]


def notice(alarms):
    """Reminder text for the given alarms, or '' if none of them may be shown."""
    selected = select_alarms(alarms)
    if not selected:
        return ''
    return (_HEADER + POLICY + _TEMPLATE_INSTRUCTIONS
            + json.dumps(template(selected), ensure_ascii=False, indent=2))
