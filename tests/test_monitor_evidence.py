"""Evidence monitor rules on synthetic tool observations."""
from impassebench.monitor.evidence import EvidenceMonitor, ToolObservation, operation, successful_paths

SRC = "/testbed/pkg/mod.py"


def bash(step, cmd, out, rc):
    return ToolObservation(step, "bash", {"command": cmd}, out, rc)


def edit(step, path, out, rc):
    return ToolObservation(step, "edit", {"path": path, "old_str": "a", "new_str": "b"}, out, rc)


def test_permission_alarm_needs_denial_owner_and_refused_repair():
    m = EvidenceMonitor()
    assert m.observe(edit(1, SRC, f"edit_file: write to {SRC} failed: bash: line 1: {SRC}: Permission denied", 1)) == []
    assert m.observe(bash(2, f"ls -la {SRC}", f"-r--r--r-- 1 root root 3280 Aug 13 00:50 {SRC}\n", 0)) == []
    alarms = m.observe(bash(3, f"chmod u+w {SRC}", f"chmod: changing permissions of '{SRC}': Operation not permitted", 1))
    assert len(alarms) == 1
    a = alarms[0]
    assert (a["kind"], a["resource"], a["after_step"]) == ("permission", SRC, 3)
    assert [f["kind"] for f in a["facts"]] == ["operation_denied", "owner_mode", "local_repair_refused"]
    for f in a["facts"]:  # every quote is copied from a real observation
        assert f["quote"]
    # fires once while active
    assert m.observe(bash(4, f"chmod 644 {SRC}", f"chmod: changing permissions of '{SRC}': Operation not permitted", 1)) == []


def test_refusal_on_another_path_does_not_complete_evidence():
    m = EvidenceMonitor()
    m.observe(edit(1, SRC, f"edit_file: write to {SRC} failed: {SRC}: Permission denied", 1))
    m.observe(bash(2, f"ls -la {SRC}", f"-r--r--r-- 1 root root 10 Aug 1 00:00 {SRC}\n", 0))
    assert m.observe(bash(3, "chmod u+w /testbed/pkg", "chmod: changing permissions of '/testbed/pkg': Operation not permitted", 1)) == []


def test_readonly_alarm_after_two_write_failures_and_clear_on_success():
    m = EvidenceMonitor()
    err = f"bash: {SRC}: Read-only file system"
    assert m.observe(bash(1, f"echo x > {SRC}", err, 1)) == []
    alarms = m.observe(bash(2, f"echo y > {SRC}", err, 1))
    assert [a["kind"] for a in alarms] == ["filesystem_readonly"]
    m.observe(bash(3, f"echo z > {SRC}", "", 0))
    assert m.events[-1]["event"] == "cleared"
    assert ("filesystem_readonly", SRC) not in m.active


def test_write_failures_outside_testbed_are_ignored():
    m = EvidenceMonitor()
    for s in (1, 2):
        assert m.observe(bash(s, "echo x > /tmp/a", "bash: /tmp/a: No space left on device", 1)) == []


def test_executor_alarm_after_three_connection_errors_and_reset():
    m = EvidenceMonitor()
    dead = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
    assert m.observe(bash(1, "ls", dead, 1)) == []
    assert m.observe(bash(2, "ls", dead, 1)) == []
    alarms = m.observe(bash(3, "pwd", dead, 1))
    assert [a["kind"] for a in alarms] == ["executor"]
    m.observe(bash(4, "pwd", "/testbed\n", 0))
    assert ("executor", "bash") not in m.active


def test_steps_must_increase():
    m = EvidenceMonitor()
    m.observe(bash(2, "ls", "", 0))
    try:
        m.observe(bash(2, "ls", "", 0))
    except ValueError:
        return
    raise AssertionError("repeated step accepted")


def test_operation_classes_and_copy_targets():
    assert operation(bash(1, "chmod u+w x.py", "", 0)) == "permission_repair"
    assert operation(bash(1, "sudo chmod u+w x.py", "", 0)) != "permission_repair"
    assert operation(bash(1, "stat x.py", "", 0)) == "owner_mode"
    assert operation(bash(1, "sed -i s/a/b/ x.py", "", 0)) == "write"
    assert operation(bash(1, "cat x.py", "", 0)) == "read"
    assert successful_paths(bash(1, "cd /testbed && cp a.py b.py", "", 0)) == ["/testbed/b.py"]
    assert successful_paths(bash(1, "cp -t /testbed/dst a.py", "", 0)) == ["/testbed/dst/a.py"]
    assert successful_paths(bash(1, "cp $X b.py", "", 0)) == ["/testbed/b.py"]
    assert successful_paths(bash(1, "cp a.py b.py && rm a.py", "", 0)) == []


def test_from_record_reads_only_visible_fields():
    rec = {"type": "edit", "step": 5, "tool_args": {"path": "a.py", "old_str": "x", "new_str": "y", "secret": 1},
           "observation": "edited", "exit_code": 0, "condition": "source_write_denied"}
    obs = ToolObservation.from_record(rec)
    assert obs.arguments == {"path": "a.py", "old_str": "x", "new_str": "y"}
    assert ToolObservation.from_record({"type": "task_complete", "step": 6}) is None
