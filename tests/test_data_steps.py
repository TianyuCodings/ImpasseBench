"""Step and record encoding used by the released parquet files (no pyarrow needed)."""
import json

from impassebench import data as D

BASH = {"type": "bash", "action": "ls -la", "observation": "total 0\n", "exit_code": 0,
        "wall_time_s": 0.12, "truncated": False, "assistant_text": "", "step": 1}
BASH_FAULT = {"type": "bash", "action": "git status", "observation": "fatal: I/O error\n",
              "exit_code": 128, "wall_time_s": 0.05, "truncated": False, "fault_injected": True,
              "assistant_text": "Checking state.", "step": 2}
EDIT = {"type": "edit", "action": "edit pkg/mod.py", "observation": "edited pkg/mod.py (+1 lines)",
        "exit_code": 0, "assistant_text": "", "step": 3}
DONE = {"type": "task_complete",
        "action": {"summary": "Fixed it.", "verified": False, "unverified_reason": "tests unreadable"},
        "assistant_text": "", "step": 4}
ESC = {"type": "escalate_to_human",
       "action": {"category": "filesystem", "summary": "s", "evidence": ["EROFS"], "attempts": ["touch"],
                  "blocked_by": "b", "requested_action": "r", "resume_condition": "c"},
       "assistant_text": "", "step": 5}
TEXT = {"type": "text", "action": "I will now inspect the repository.", "step": 6}
INVALID = {"type": "invalid_tool", "action": "unknown_tool", "args": {"x": 1},
           "assistant_text": "", "step": 7}

ALL = [BASH, BASH_FAULT, EDIT, DONE, ESC, TEXT, INVALID]


def test_steps_round_trip_exactly():
    for step in ALL:
        enc = D.encode_step(step)
        assert set(enc) == set(D.STEP_FIELDS) | {"extra"}
        assert enc["extra"] is None
        dec = D.decode_step(enc)
        assert dec == step
        # key order matches the harness, so trace.jsonl lines are byte-identical
        assert json.dumps(dec) == json.dumps(step)


def test_terminal_arguments_are_json_encoded():
    enc = D.encode_step(DONE)
    assert json.loads(enc["action"]) == DONE["action"]
    assert enc["observation"] is None
    assert json.loads(D.encode_step(INVALID)["args"]) == {"x": 1}


def test_unusual_values_survive_via_extra():
    odd = [
        {"type": "bash", "action": ["not", "a", "string"], "step": 1},
        {"type": "bash", "action": "true", "exit_code": None, "step": 2},
        {"type": "bash", "action": "true", "wall_time_s": 1, "step": 3},       # int, not float
        {"type": "bash", "action": "true", "exit_code": True, "step": 4},      # bool, not int
        {"type": "bash", "action": "true", "new_field": {"a": [1, 2]}, "step": 5},
        {"type": "task_complete", "action": "plain string", "step": 6},
    ]
    for step in odd:
        assert D.decode_step(D.encode_step(step)) == step


def test_record_round_trip_and_raw_view():
    rec = {
        "traj_id": "m/escalate/inst__x/readonly_fs", "model": "m", "model_id": "m-id",
        "interface": "escalate", "with_escalate": True, "with_edit": True,
        "instance_id": "inst__x", "repo": "swesmith/inst", "condition": "readonly_fs",
        "category": "persistent_barrier", "max_steps": 80, "n_steps": len(ALL),
        "termination": "escalate", "total_tokens": 1234, "sampling": "temperature=0.7",
        "started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:05:00",
        "system_prompt": "You are ...", "final_patch": "",
        "runtime_fault": {"kind": "sandbox_flaky", "first_failed_step": 4},
        "base_image": "img", "base_image_digest": "img@sha256:00",
        "steps": ALL, "labels": {"C": 1, "Q": 1, "t_star": 2},
    }
    row = D.encode_record(rec)
    assert isinstance(row["runtime_fault"], str)
    back = D.decode_record(row)
    assert back == rec
    meta, trace = D.record_to_raw(back)
    assert trace == ALL
    assert meta["with_escalate"] is True and meta["model"] == "m-id" and meta["model_patch"] == ""


def test_run_dir_round_trip(tmp_path):
    rec = {
        "traj_id": "m/base/inst__y/clean", "model": "m", "model_id": "m-id", "interface": "base",
        "with_escalate": False, "with_edit": True, "instance_id": "inst__y", "repo": "r",
        "condition": "clean", "category": "control", "max_steps": 80, "n_steps": 2,
        "termination": "task_complete", "total_tokens": 10, "sampling": "temperature=0.7",
        "started_at": "a", "ended_at": "b", "system_prompt": "p", "final_patch": "diff --git a/x b/x\n",
        "runtime_fault": None, "base_image": "i", "base_image_digest": "d",
        "steps": [BASH, DONE], "labels": None,
    }
    run_dir = D.write_run_dir(rec, tmp_path)
    assert (run_dir / "diff.patch").read_text() == rec["final_patch"]
    raw = (run_dir / "trace.jsonl").read_text()
    assert raw == json.dumps(BASH) + "\n" + json.dumps(DONE) + "\n"
    back = D.read_run_dir(run_dir, model="m")
    assert back["traj_id"] == rec["traj_id"] and back["steps"] == rec["steps"]
    assert back["category"] == "control" and back["final_patch"] == rec["final_patch"]
    assert [r["traj_id"] for r in D.iter_run_dirs(tmp_path, model="m")] == [rec["traj_id"]]
