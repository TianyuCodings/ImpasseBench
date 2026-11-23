"""Patch composition for grading ``bug_absent`` on SWE-bench Verified.

A ``bug_absent`` task starts from a healthy tree (the reference fix already
applied), so the agent's patch is relative to that tree. To grade it with the
official harness, which starts from the buggy base commit, the delivered tree
is rebuilt on a disposable copy of the original image: reference patch first
(the healthy starting state), then the agent patch, even if it reverts the
fix. Both intermediate states are checked against the run's filesystem
snapshots (see :mod:`.artifacts`), so a patch that does not reproduce the
observed files fails closed. No test result is used to choose a patch.

This file is also executed inside the task image (standard library only and
compatible with old Python versions); keep it self-contained.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def git(tree, *args, payload=None):
    command = ['git', '-C', str(tree), '-c', 'core.fileMode=false',
               '-c', 'core.hooksPath=/dev/null'] + list(args)
    result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
    return result.stdout


def file_value(path):
    if path.is_symlink():
        return {'type': 'link', 'target': os.readlink(path)}
    if path.is_file():
        return {'type': 'file', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if not path.exists():
        return None
    raise ValueError('Unsupported tracked artifact type: %s' % path)


def tracked(tree):
    return {p.decode() for p in git(tree, 'ls-files', '-z').split(b'\0') if p}


def content_matches(tree, expected, paths):
    mismatches = []
    for name in sorted(paths):
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        if file_value(tree / path) != expected.get(name):
            mismatches.append(name)
    if mismatches:
        raise ValueError('Artifact content mismatch: ' + ', '.join(mismatches[:10]))


def compose(tree, reference_patch, agent_patch, healthy_files=None, delivered_files=None):
    """Compose original -> healthy -> delivered in ``tree`` (a disposable checkout)."""
    tree = Path(tree)
    assert isinstance(reference_patch, str) and reference_patch.strip()
    assert isinstance(agent_patch, str)
    assert git(tree, 'rev-parse', '--is-inside-work-tree').strip() == b'true'
    # image-build edits belong to the starting state, never to the agent patch
    git(tree, 'add', '-A')
    original = git(tree, 'write-tree').decode().strip()
    original_paths = tracked(tree)
    ref = reference_patch if reference_patch.endswith('\n') else reference_patch + '\n'
    git(tree, 'apply', '--check', '--whitespace=nowarn', '-', payload=ref.encode())
    git(tree, 'apply', '--whitespace=nowarn', '-', payload=ref.encode())
    git(tree, 'add', '-A')
    healthy_paths = tracked(tree)
    healthy_tree = git(tree, 'write-tree').decode().strip()
    if healthy_files is not None:
        content_matches(tree, healthy_files, original_paths | healthy_paths)
    if agent_patch.strip():
        git(tree, 'apply', '--check', '--whitespace=nowarn', '-', payload=agent_patch.encode())
        git(tree, 'apply', '--whitespace=nowarn', '-', payload=agent_patch.encode())
    git(tree, 'add', '-A')
    delivered = git(tree, 'write-tree').decode().strip()
    delivered_paths = tracked(tree)
    if delivered_files is not None:
        content_matches(tree, delivered_files, original_paths | healthy_paths | delivered_paths)
    patch = git(tree, 'diff', '--binary', '--no-ext-diff', original, delivered).decode()
    return {'composed_patch': patch, 'original_tree': original, 'healthy_tree': healthy_tree,
            'delivered_tree': delivered,
            'reference_patch_sha256': hashlib.sha256(reference_patch.encode()).hexdigest(),
            'agent_patch_sha256': hashlib.sha256(agent_patch.encode()).hexdigest(),
            'composed_patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
            'healthy_content_checked': healthy_files is not None,
            'delivered_content_checked': delivered_files is not None,
            'scoring_scope': 'Preservation of a healthy starting task; not a newly solved bug.'}


def snapshot_maps(run):
    """Initial and final /testbed contents reconstructed from the snapshot files."""
    events = sorted((Path(run) / 'artifacts').glob('*.json'))
    assert events, 'Missing artifact observations'
    state, healthy, hashes = {}, None, {}
    for index, path in enumerate(events):
        event = json.loads(path.read_text())
        assert event['index'] == index and not event.get('errors') and not event.get('error')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for change in event['changes']:
            name = change['path']
            if not name.startswith('/testbed/') or name.startswith('/testbed/.git/') or name == '/testbed/.git':
                continue
            name = name[len('/testbed/'):]
            after = change['after']
            if after.get('type') == 'file':
                state[name] = {'type': 'file', 'sha256': after['sha256']}
            elif after.get('type') == 'link':
                state[name] = {'type': 'link', 'target': after['target']}
            else:
                state.pop(name, None)
        if index == 0:
            assert event['actor'] == 'baseline' and event['label'] == 'before_fault_setup'
            healthy = dict(state)
    return healthy, state, hashes


def compose_artifact(job, dataset_row):
    """Composed patch for an initial ``bug_absent`` trajectory, or raise."""
    assert job['condition'] == 'bug_absent'
    if job['phase'] != 'original':
        raise NotImplementedError('resumed bug_absent trajectories are not part of the study')
    run = Path(job['run_dir'])
    meta = json.loads((run / 'meta.json').read_text())
    assert meta['instance_id'] == dataset_row['instance_id'] == job['instance_id']
    assert meta['condition'] == 'bug_absent' and not meta.get('resumption')
    assert meta['artifact_observer_ok'] and meta['baseline_dirt_paths'] == []
    healthy, delivered, event_hashes = snapshot_maps(run)
    assert healthy and isinstance(delivered, dict)
    image = meta['base_image']
    inspected = subprocess.run(['docker', 'image', 'inspect', image], universal_newlines=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
    info = json.loads(inspected.stdout)[0]
    assert meta['base_image_digest'] and meta['base_image_digest'] in info.get('RepoDigests', [])
    request = {'reference_patch': dataset_row['patch'], 'agent_patch': meta['model_patch'],
               'healthy_files': healthy, 'delivered_files': delivered}
    name = 'impassebench-compose-' + uuid.uuid4().hex[:20]
    code = Path(__file__).read_text()
    command = ['docker', 'run', '--rm', '--pull=never', '--name', name,
               '--network', 'none', '--cap-drop=ALL', '--security-opt', 'no-new-privileges',
               '--memory=2g', '--cpus=2', '--pids-limit=256', '-i', '--entrypoint',
               '/opt/miniconda3/envs/testbed/bin/python', info['Id'], '-c', code]
    try:
        result = subprocess.run(command, input=json.dumps(request), universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    composed = json.loads(result.stdout)
    assert composed['healthy_content_checked'] and composed['delivered_content_checked']
    patch = composed.pop('composed_patch')
    composed.update(original_image_id=info['Id'], original_image_digest=meta['base_image_digest'],
                    artifact_event_sha256=event_hashes,
                    composition_code_sha256=hashlib.sha256(code.encode()).hexdigest())
    return patch, composed


if __name__ == '__main__':
    request = json.load(sys.stdin)
    print(json.dumps(compose(Path('/testbed'), request['reference_patch'], request['agent_patch'],
                             healthy_files=request['healthy_files'],
                             delivered_files=request['delivered_files'])))
