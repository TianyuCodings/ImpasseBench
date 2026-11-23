"""Observer-only filesystem snapshots, attributed to the agent or the harness.

After the container starts, after fault setup, and after every ``bash`` /
``edit_file`` step, the harness records which files under /testbed and /tmp
(plus every path Docker reports as changed) were created, modified or
removed, with their mode, owner and content hash. The snapshots are written
to ``<run_dir>/artifacts/NNNN.json``; they are never shown to the agent.
They let an evaluator reconstruct the initial and delivered file contents
(used to grade ``bug_absent`` against its healthy starting state).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess


def collect_snapshot(request):
    # Runs inside the container (its source is sent to python3 there).
    import hashlib
    import os
    import stat
    result, errors = {}, []
    previous = request.get('previous', {})
    paths = set(request.get('layer_paths', []))
    for root in request['roots']:
        for directory, dirs, files in os.walk(root, followlinks=False,
                onerror=lambda e: errors.append(str(e))):
            paths.add(directory)
            paths.update(os.path.join(directory, p) for p in dirs + files)
    for path in sorted(paths):
        try:
            st = os.lstat(path)
            kind = ('file' if stat.S_ISREG(st.st_mode) else 'link' if stat.S_ISLNK(st.st_mode)
                    else 'dir' if stat.S_ISDIR(st.st_mode) else 'special')
            entry = {'type': kind, 'mode': stat.S_IMODE(st.st_mode), 'uid': st.st_uid, 'gid': st.st_gid}
            if kind == 'file':
                key = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode]
                old = previous.get(path, {})
                if old.get('stat_key') == key:
                    entry['sha256'] = old['sha256']
                else:
                    h = hashlib.sha256()
                    with open(path, 'rb') as f:
                        for block in iter(lambda: f.read(1024 * 1024), b''):
                            h.update(block)
                    entry['sha256'] = h.hexdigest()
                entry['stat_key'] = key
                entry['size'] = st.st_size
            elif kind == 'link':
                entry['target'] = os.readlink(path)
            result[path] = entry
        except FileNotFoundError:
            pass  # a deleted Docker-layer path is represented by its absence
        except OSError as exc:
            errors.append(f'{path}: {exc}')
    return {'files': result, 'errors': errors}


def semantic(entry):
    return {k: v for k, v in (entry or {}).items() if k != 'stat_key'}


def changes(before, after):
    return [{'path': p, 'before': semantic(before.get(p)), 'after': semantic(after.get(p))}
            for p in sorted(before.keys() | after.keys())
            if semantic(before.get(p)) != semantic(after.get(p))]


class ArtifactObserver:
    def __init__(self, container_name, output_dir):
        self.container_name = container_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.previous = {}
        self.counter = 0

    def snapshot(self, actor, label):
        diff = subprocess.run(['docker', 'diff', self.container_name],
                              capture_output=True, text=True, timeout=120)
        if diff.returncode:
            raise RuntimeError(diff.stderr)
        layer = [line[2:] for line in diff.stdout.splitlines() if len(line) > 2]
        request = {'roots': ['/testbed', '/tmp'], 'layer_paths': layer,
                   'previous': self.previous}
        program = inspect.getsource(collect_snapshot) + '\nimport sys,json\nprint(json.dumps(collect_snapshot(json.load(sys.stdin))))'
        p = subprocess.run(['docker', 'exec', '-i', self.container_name, 'python3', '-c', program],
                           input=json.dumps(request), capture_output=True, text=True, timeout=180)
        if p.returncode:
            raise RuntimeError(p.stderr)
        value = json.loads(p.stdout)
        snapshot = value['files']
        event = {'index': self.counter, 'actor': actor, 'label': label,
                 'changes': changes(self.previous, snapshot), 'errors': value['errors'],
                 'scope': '/testbed, /tmp, and Docker-layer changed paths; no symlink traversal',
                 'snapshot_sha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()}
        if value['errors']:
            event['error'] = 'Incomplete filesystem snapshot'
        (self.output_dir / f'{self.counter:04d}.json').write_text(json.dumps(event, indent=2) + '\n')
        # The first snapshot plus each diff suffice to reconstruct contents;
        # stat keys only serve as a hash cache.
        self.previous = snapshot
        self.counter += 1
        return event
