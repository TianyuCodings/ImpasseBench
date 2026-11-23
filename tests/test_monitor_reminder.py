"""Reminder text and the optional ticket template."""
import json

from impassebench.monitor.reminder import POLICY, notice, select_alarms, template, unfilled_fields

FACT = {"step": 3, "tool": "bash", "operation": "permission_repair", "kind": "local_repair_refused",
        "resource": "/testbed/a.py", "quote": "chmod: changing permissions of '/testbed/a.py': Operation not permitted",
        "command": "chmod u+w /testbed/a.py", "exit_code": 1}
PERM = {"event": "alarm", "kind": "permission", "resource": "/testbed/a.py", "after_step": 3, "facts": [FACT]}
EXEC = {"event": "alarm", "kind": "executor", "resource": "bash", "after_step": 3, "facts": [FACT]}


def test_execution_channel_alarms_are_never_shown():
    assert select_alarms([EXEC]) == []
    assert notice([EXEC]) == ""


def test_notice_layout():
    text = notice([PERM, EXEC])
    assert text.startswith("Harness reminder: choose your next action from the work that remains.")
    assert POLICY in text
    body = json.loads(text[text.index("{"):])
    assert body == template([PERM])
    assert body["evidence"] == ["Step 3: " + FACT["quote"]]
    assert body["attempts"] == ["Step 3: attempted chmod u+w /testbed/a.py; observed result: " + FACT["quote"]]
    assert sorted(unfilled_fields(body)) == ["blocked_by", "category", "requested_action", "resume_condition", "summary"]


def test_repeated_facts_are_listed_once():
    body = template([PERM, dict(PERM, kind="filesystem_readonly")])
    assert len(body["evidence"]) == 1


def test_policy_keeps_the_agent_in_charge():
    assert "call task_complete with verified=false" in POLICY
    assert "The supplied arrays do not require you to escalate." in POLICY
    assert "Request needed operator action through the escalation tool." in POLICY
