"""Rule-based trajectory annotator for ImpasseBench.

    from impassebench.annotate import annotate, load_tasks
    tasks = load_tasks()
    labels = annotate(record, tasks[record["instance_id"]])

No LLM judge is involved: every label is computed from the recorded tool
calls, observations and final report. See docs/metrics.md.
"""

from .core import (ANNOTATOR_VERSION, CATEGORY, CONDITIONS, LABEL_FIELDS,
                   annotate)
from .io import load_run_dir, load_tasks

__all__ = ["ANNOTATOR_VERSION", "CATEGORY", "CONDITIONS", "LABEL_FIELDS",
           "annotate", "load_run_dir", "load_tasks"]
