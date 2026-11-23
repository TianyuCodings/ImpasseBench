"""Command-line entry point: ``impassebench <command> [options]``."""

import importlib
import sys

COMMANDS = {
    "pull-images": ("impassebench.harness.images", "Pull the Docker images for the benchmark tasks."),
    "run": ("impassebench.harness.run", "Run agents on tasks x conditions and record trajectories."),
    "annotate": ("impassebench.annotate.cli", "Label recorded trajectories (C, Q, t*, N_post, L, S, U, Y, Z)."),
    "grade": ("impassebench.annotate.grade", "Grade final patches in a healthy container (A)."),
    "monitor": ("impassebench.monitor.experiment", "Run the evidence-monitor harness experiments."),
}


def _usage() -> str:
    lines = ["usage: impassebench <command> [options]", "", "commands:"]
    lines += [f"  {name:<12} {doc}" for name, (_, doc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command '{name}'\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
