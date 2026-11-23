"""Loaders over a tiny synthetic dataset laid out like the released one."""
import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from impassebench import data as D  # noqa: E402


def _record(model, interface, inst, cond, n=2):
    steps = [{"type": "bash", "action": f"echo {i}", "observation": f"{i}\n", "exit_code": 0,
              "wall_time_s": 0.1, "truncated": False, "assistant_text": "", "step": i + 1}
             for i in range(n - 1)]
    steps.append({"type": "task_complete", "action": {"summary": "done", "verified": True},
                  "assistant_text": "", "step": n})
    return {
        "traj_id": f"{model}/{interface}/{inst}/{cond}", "model": model, "model_id": model + "-id",
        "interface": interface, "with_escalate": interface == "escalate", "with_edit": True,
        "instance_id": inst, "repo": "swesmith/r", "condition": cond,
        "category": D.CONDITION_CATEGORY[cond], "max_steps": 80, "n_steps": n,
        "termination": "task_complete", "total_tokens": 100, "sampling": "temperature=0.7",
        "started_at": "s", "ended_at": "e", "system_prompt": "p", "final_patch": "",
        "runtime_fault": None, "base_image": "img", "base_image_digest": "img@sha256:0",
        "steps": steps, "labels": {"C": 1, "Q": None, "t_star": None, "Y": 1},
    }


@pytest.fixture()
def dataset(tmp_path):
    recs = [_record(m, i, inst, c)
            for m in ("model-a", "model-b") for i in ("base", "escalate")
            for inst in ("t1", "t2") for c in ("clean", "readonly_fs")]
    for m in ("model-a", "model-b"):
        for i in ("base", "escalate"):
            rows = [D.encode_record(r) for r in recs if r["model"] == m and r["interface"] == i]
            path = tmp_path / "data" / "trajectories" / m / f"{i}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
    flat = [{"traj_id": r["traj_id"], "model": r["model"], "condition": r["condition"], **r["labels"]}
            for r in recs]
    pq.write_table(pa.Table.from_pylist(flat), tmp_path / "data" / "labels.parquet")
    with open(tmp_path / "tasks.jsonl", "w") as fh:
        for t in ("t1", "t2"):
            fh.write(json.dumps({"instance_id": t, "conditions": ["clean", "readonly_fs"]}) + "\n")
    (tmp_path / "conditions.json").write_text(json.dumps([{"id": "clean"}, {"id": "readonly_fs"}]))
    (tmp_path / "models.json").write_text(json.dumps([{"slug": "model-a"}, {"slug": "model-b"}]))
    with open(tmp_path / "fault_setups.jsonl", "w") as fh:
        fh.write(json.dumps({"instance_id": "t1", "condition": "readonly_fs", "docker_run_args": ["--read-only"]}) + "\n")
    return tmp_path, {r["traj_id"]: r for r in recs}


def test_iter_trajectories_restores_records(dataset):
    root, recs = dataset
    got = {r["traj_id"]: r for r in D.iter_trajectories(root)}
    assert set(got) == set(recs)
    for tid, rec in recs.items():
        assert got[tid]["steps"] == rec["steps"]
        assert got[tid]["labels"] == rec["labels"]
        assert {k: got[tid][k] for k in D.RECORD_FIELDS} == {k: rec[k] for k in D.RECORD_FIELDS}


def test_filters_and_single_lookup(dataset):
    root, recs = dataset
    sel = list(D.iter_trajectories(root, models=["model-b"], interfaces=["escalate"],
                                   conditions=["readonly_fs"], instances=["t2"]))
    assert [r["traj_id"] for r in sel] == ["model-b/escalate/t2/readonly_fs"]
    rec = D.load_trajectory("model-a/base/t1/clean", root)
    assert rec["steps"] == recs["model-a/base/t1/clean"]["steps"]
    with pytest.raises(KeyError):
        D.load_trajectory("model-a/base/t9/clean", root)
    light = next(D.iter_trajectories(root, models=["model-a"], with_steps=False))
    assert "steps" not in light


def test_tables(dataset):
    root, recs = dataset
    assert set(D.load_tasks(root)) == {"t1", "t2"}
    assert [c["id"] for c in D.load_conditions(root)] == ["clean", "readonly_fs"]
    assert len(D.load_models(root)) == 2
    assert D.load_fault_setups(root)[("t1", "readonly_fs")]["docker_run_args"] == ["--read-only"]
    labels = D.load_labels(root)
    assert len(labels) == len(recs) and {r["traj_id"] for r in labels} == set(recs)
