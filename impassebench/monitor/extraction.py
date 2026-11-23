"""Patch extraction that does not require the faulted tree to be writable.

Normal extraction runs unchanged. If it reports an error (for example the
git index cannot be written under a read-only or full filesystem), the
stopped run's container is committed to a temporary image and the patch is
extracted from a fresh container of that image. The agent's own container is
never made writable. Docker commits omit mounted data, so the fallback is
refused when the repository or the baseline gitdir is a mount.
"""
from __future__ import annotations

import json
import subprocess
import uuid

from ..harness.container import ContainerSession

_PROTECTED = ("/testbed", "/opt/t2e_baseline.git")


class SnapshotExtractionSession(ContainerSession):
    patch_extraction_meta = None

    def git_diff(self):
        patch, error = super().git_diff()
        self.patch_extraction_meta = {"method": "original_container", "initial_error": error}
        if error is None:
            return patch, None
        suffix = uuid.uuid4().hex
        tag, name = "impassebench-extract:" + suffix, "impassebench_extract_" + suffix
        snapshot = None
        copied = ContainerSession(image=tag, name=name)
        try:
            inspection = subprocess.run(["docker", "inspect", self.name], capture_output=True,
                                        text=True, check=True, timeout=30)
            for mount in json.loads(inspection.stdout)[0]["Mounts"]:
                target = mount["Destination"].rstrip("/") or "/"
                for root in _PROTECTED:
                    if (target == "/" or target == root or root.startswith(target + "/")
                            or target.startswith(root + "/")):
                        raise RuntimeError("snapshot extraction cannot capture a mounted repository: " + target)
            result = subprocess.run(["docker", "commit", "--pause=true", self.name, tag],
                                    capture_output=True, text=True, check=True, timeout=300)
            snapshot = result.stdout.strip()
            copied.start()
            extracted, snapshot_error = ContainerSession.git_diff(copied)
            self.patch_extraction_meta.update(method="writable_snapshot_after_terminal",
                                              snapshot_image_id=snapshot,
                                              original_container_unchanged=True,
                                              snapshot_error=snapshot_error)
            if snapshot_error:
                return patch, error + "; writable snapshot: " + snapshot_error
            return extracted, None
        except Exception as exc:
            self.patch_extraction_meta["snapshot_error"] = repr(exc)
            return patch, error + "; snapshot extraction failed: " + repr(exc)
        finally:
            copied.stop()
            if snapshot:
                try:
                    subprocess.run(["docker", "image", "rm", tag], capture_output=True, timeout=60)
                except Exception as exc:
                    self.patch_extraction_meta["cleanup_error"] = repr(exc)
