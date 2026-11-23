"""Prompts and tool schemas are the ones the released trajectories were run with."""
import hashlib
import json

import pytest

from impassebench.harness import prompts

# sha256 of the system prompt recorded in every released trajectory
RECORDED_SYSTEM_PROMPT_SHA256 = {
    "base": "975b8f3245764fadd876b24143c182cc5e553733fbccea98ff9bcf9e9d22784b",
    "escalate": "6b1a11937ad7fbbe4c108650540490e0ccec921147e1a48d2d4bc7c3c1b0a5c3",
}


@pytest.mark.parametrize("interface", prompts.INTERFACES)
def test_system_prompt_matches_recorded(interface):
    sp = prompts.system_prompt(interface)
    assert hashlib.sha256(sp.encode()).hexdigest() == RECORDED_SYSTEM_PROMPT_SHA256[interface]
    assert ("escalate_to_human" in sp) == (interface == "escalate")


def test_tools_per_interface():
    names = {i: [t["name"] for t in prompts.tool_declarations(i)] for i in prompts.INTERFACES}
    assert names["base"] == ["bash", "task_complete", "edit_file"]
    assert names["escalate"] == ["bash", "task_complete", "edit_file", "escalate_to_human"]
    esc = prompts.ESCALATE_TOOL["parameters"]["required"]
    assert esc == ["category", "summary", "evidence", "attempts",
                   "blocked_by", "requested_action", "resume_condition"]
    json.dumps(prompts.tool_declarations("escalate"))  # serializable


def test_task_prompt():
    t = prompts.task_prompt("Something is broken.", 80)
    assert "<issue>\nSomething is broken.\n</issue>" in t
    assert "budget of 80 tool calls" in t


def test_unknown_interface():
    with pytest.raises(ValueError):
        prompts.system_prompt("tools")
