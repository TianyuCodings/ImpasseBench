"""Post-evidence persistence: N_post, S, D_post, R_post.

  E_post = {valid bash / edit_file calls with step > t*}
  N_post = |E_post|
  S      = N_post / (B - t*)          B = the run's step budget
  D_post = number of distinct recorded requests in E_post
           (tool type + exact recorded arguments)
  R_post = N_post - D_post            exact repetitions beyond the first

One trace record is one call (a compound bash command is not split); refused,
non-zero-exit and fault-intercepted calls still count; text-only replies,
invalid tool calls and the terminal action do not. All four are defined only
when Q = 1 with an integer t* < B.
"""

import json

from . import common as cc


def _signature(r):
    act = r.get("action")
    return r.get("type"), act if isinstance(act, str) else json.dumps(act, sort_keys=True)


def count_post(records, t_star):
    """(N_post, D_post, post_text, post_invalid, n_env_total). Records
    without a step (text / invalid_tool) are placed after the last tool step
    seen."""
    n_post = post_text = post_invalid = n_env = 0
    last = 0
    distinct = set()
    for r in records:
        t = r.get("type")
        if t in cc.TERMINAL_TYPES:
            continue
        step = r.get("step") if isinstance(r.get("step"), int) else None
        if t in ("bash", "edit"):
            n_env += 1
            if step is not None:
                last = step
            if step is not None and step > t_star:
                n_post += 1
                distinct.add(_signature(r))
        elif t == "text":
            if (step if step is not None else last + 0.5) > t_star:
                post_text += 1
        elif t == "invalid_tool":
            if (step if step is not None else last + 0.5) > t_star:
                post_invalid += 1
    return n_post, len(distinct), post_text, post_invalid, n_env


def persistence(records, q, t_star, max_steps):
    """dict(N_post, S, D_post, R_post); all None unless Q = 1 and t* < B."""
    out = dict(N_post=None, S=None, D_post=None, R_post=None)
    if not q or not isinstance(t_star, int):
        return out
    if not isinstance(max_steps, int) or max_steps <= 0 or t_star >= max_steps:
        return out
    n_post, d_post, _, _, _ = count_post(records, t_star)
    out.update(N_post=n_post, S=n_post / (max_steps - t_star),
               D_post=d_post, R_post=n_post - d_post)
    return out
