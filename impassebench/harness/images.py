"""Pull the Docker images of the benchmark tasks ahead of a run.

Pulls are bounded in parallelism and retried with backoff (Docker Hub
rate-limits anonymous pulls; ``docker login`` raises the quota). Images that
are already present are skipped.

    impassebench pull-images                 # all 10 repository images
    impassebench pull-images --tasks <id>,<id> --parallel 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .rollout import image_for
from .tasks import load_tasks


def have_image(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True, timeout=60).returncode == 0


def pull(image: str, retries: int = 4) -> tuple[str, bool, str]:
    if have_image(image):
        return image, True, "present"
    delay = 30
    err = ""
    for _ in range(retries):
        p = subprocess.run(["docker", "pull", "--platform", "linux/amd64", "-q", image],
                           capture_output=True, text=True)
        if p.returncode == 0:
            return image, True, "pulled"
        err = (p.stderr or p.stdout).strip()[-300:]
        time.sleep(delay)
        delay *= 2
    return image, False, err


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="impassebench pull-images", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all", help="comma-separated instance ids, or 'all'")
    ap.add_argument("--task-file", default=None)
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args(argv)

    tasks = load_tasks(args.task_file)
    if args.tasks != "all":
        wanted = set(s.strip() for s in args.tasks.split(","))
        tasks = [t for t in tasks if t["instance_id"] in wanted]
    images = sorted({image_for(t) for t in tasks})
    print(f"{len(images)} images", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for f in as_completed([pool.submit(pull, im) for im in images]):
            image, ok, info = f.result()
            failed += not ok
            print(f"  {'ok ' if ok else 'FAIL'} {image} ({info})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
