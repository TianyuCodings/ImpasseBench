"""Operator restoration: receipt, restore condition and history rebuilding."""
from impassebench.monitor.resume import (ResumingAgent, history_for_resume, operator_receipt,
                                         restore_condition, restore_setup)

META = {"task_preview": "Fix f.", "max_steps": 80, "with_escalate": True, "termination": "escalate"}
RECORDS = [
    {"type": "bash", "action": "ls", "observation": "a.py\n", "exit_code": 0, "assistant_text": "look", "step": 1},
    {"type": "text", "action": "thinking", "step": 2},
    {"type": "escalate_to_human", "action": {"summary": "blocked"}, "assistant_text": "", "step": 3},
]
EVENTS = [{"event": "harness_notice", "before_step": 3, "text": "Harness reminder: x"}]


def test_history_for_resume():
    receipt = operator_receipt(77)
    h = history_for_resume(META, RECORDS, EVENTS, receipt)
    assert [x["role"] for x in h] == ["user", "assistant", "tool", "assistant", "user", "user", "assistant", "tool"]
    assert h[5] == {"role": "user", "text": "Harness reminder: x", "source": "harness_notice"}
    assert h[-1]["response"] == receipt and h[-1]["call_id"] == h[-2]["call_id"]
    assert receipt["remaining_steps"] == 77 and "saved repository patch" in receipt["action"]


def test_restore_condition_and_setup():
    c = restore_condition("bug_absent", "agent", "diff --git a/x b/x\n", "operator_restore_x")
    assert c.skip_bug_checkout and c.agent_user == "agent" and len(c.setup_cmds) == 2
    assert not restore_condition("readonly_fs", None, "", "r").skip_bug_checkout
    assert restore_setup("", None)[0].startswith("python3 -c ")


def test_resuming_agent_prepends_original_history():
    class D:
        def set_system(self, s): self.s = s
        def next_action(self, h): self.h = h; return {"tool": None, "text": ""}
    d = D()
    a = ResumingAgent(d, [{"role": "user", "text": "old"}], "SYS")
    a.set_system("SYS")
    a.next_action([{"role": "user", "text": "task again"}, {"role": "user", "text": "new"}])
    assert d.h == [{"role": "user", "text": "old"}, {"role": "user", "text": "new"}] and a.calls == 1
