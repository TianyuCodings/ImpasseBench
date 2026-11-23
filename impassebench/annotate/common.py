#!/usr/bin/env python3
"""Shared trace, text, shell and Python-analysis helpers for the annotator.

Everything here is mechanical and auditable:

  * step validation: a record that is not a trace step is skipped for
    labelling but counted, so a partly unreadable trace is flagged rather
    than scored as if it were complete;
  * the terminal record as a list of (field, text) pairs and as a parsed
    record (kind, exact-bool `verified`), read from the trace itself;
  * sentence-level vocabulary matching with negation / hypothetical /
    fix-claim guards on the model's own prose; evidence items are exempt
    from the guards ONLY when they are verbatim quotes of this rollout's
    observations (the field name alone proves nothing);
  * shell parsing: heredoc bodies, the python the agent actually executed,
    source-file writes issued from the shell or from python (with a
    "confirmed" bit -- a write behind `false &&`, inside quotes, or whose
    own status is masked is NOT a confirmed edit), scratch files keyed by
    their ABSOLUTE path resolved against the shell's cwd (no basename
    matching), and exit-status masking;
  * an AST view of executed python: which statements run at module level
    (dead `if False:` bodies and uncalled function bodies excluded; pytest
    entry points included on demand) so that a call or attribute use can
    be told apart from a method object, an import, a string or a comment.
"""

import ast
import json
import os
import re
import shlex
import warnings

TESTBED = "/testbed"

TERMINAL_TYPES = ("task_complete", "escalate", "escalate_to_human")
TRACE_TYPES = {"bash", "edit", "task_complete", "escalate", "escalate_to_human", "text", "invalid_tool"}
# ticket / completion fields, in the order they are read. `evidence` items
# are meant to be verbatim quotes the model selected from its observations.
TICKET_TEXT_FIELDS = ("reason", "unverified_reason", "summary", "category",
                      "blocked_by", "resume_condition", "requested_action", "attempts")
QUOTE_FIELDS = ("evidence",)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------
def _valid_record(r):
    if not isinstance(r, dict):
        return False
    t = r.get("type")
    if t not in TRACE_TYPES:
        return False
    if t in ("bash", "edit") and not isinstance(r.get("step"), int):
        return False
    if t in TERMINAL_TYPES and not isinstance(r.get("action"), (dict, type(None))):
        return False
    return True


def validate_steps(steps):
    """(records, status) for a list of already-parsed trace rows.

    status: 'ok' | 'empty' | 'corrupt:<bad>/<total>'. Rows that are not a
    trace record (unknown type, tool record without a step) are skipped for
    labelling but COUNTED, so a partly unreadable trace is flagged instead
    of being scored as if it were complete."""
    recs, bad, total = [], 0, 0
    for r in steps or []:
        total += 1
        if not _valid_record(r):
            bad += 1
            continue
        recs.append(r)
    if total == 0:
        return [], "empty"
    if not recs:
        return [], f"corrupt:{bad}/{total}"
    return recs, ("ok" if not bad else f"corrupt:{bad}/{total}")


def read_trace_lines(lines):
    """Parse trace.jsonl lines into rows. Blank lines are ignored; a line
    that is not JSON is kept as a non-record so validate_steps counts it."""
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            out.append(None)
    return out


def terminal_record(records):
    """The rollout's own terminal action, read from the trace:
    dict(kind, step, action, verified, verified_present, verified_raw).
    kind: 'task_complete' | 'escalate' | None. `verified` is the exact
    boolean from task_complete (None when absent or not a JSON boolean --
    strings, 0/1 and None are schema errors, not states)."""
    for r in records:
        t = r.get("type")
        if t not in TERMINAL_TYPES:
            continue
        a = r.get("action") or {}
        if not isinstance(a, dict):
            a = {}
        kind = "task_complete" if t == "task_complete" else "escalate"
        raw = a.get("verified", None) if kind == "task_complete" else None
        present = kind == "task_complete" and "verified" in a
        ver = raw if isinstance(raw, bool) else None
        return dict(kind=kind, step=r.get("step"), action=a, verified=ver,
                    verified_present=present, verified_raw=raw)
    return dict(kind=None, step=None, action={}, verified=None, verified_present=False, verified_raw=None)


def terminal_fields(records):
    """(kind, fields): kind in {'task_complete','escalate',None}; fields is an
    ordered list of (name, text). List-valued ticket fields contribute one
    entry per item (evidence[0], evidence[1], ...)."""
    for r in records:
        t = r.get("type")
        if t not in TERMINAL_TYPES:
            continue
        kind = "task_complete" if t == "task_complete" else "escalate"
        a = r.get("action") or {}
        if not isinstance(a, dict):
            a = {}
        fields = []
        for k in TICKET_TEXT_FIELDS:
            v = a.get(k)
            if isinstance(v, list):
                fields.extend((f"{k}[{i}]", str(e)) for i, e in enumerate(v) if e)
            elif v:
                fields.append((k, str(v)))
        ev = a.get("evidence")
        if isinstance(ev, list):
            fields.extend((f"evidence[{i}]", str(e)) for i, e in enumerate(ev) if e)
        elif ev:
            fields.append(("evidence", str(ev)))
        if r.get("assistant_text"):
            fields.append(("assistant_text", str(r["assistant_text"])))
        return kind, fields
    return None, []


def is_quote_field(name):
    return name.split("[")[0] in QUOTE_FIELDS


def observation_corpus(records):
    return "\n".join(str(r.get("observation") or "") for r in records if r.get("type") in ("bash", "edit"))


def verbatim_units(fields, obs_corpus):
    """The sentences of quote fields that really are observation quotes:
    normalized sentence (>= 12 chars, quotes/backticks stripped) is a
    substring of this rollout's normalized observations. Only THOSE
    sentences are matched guard-free; a hypothesis written after a real
    quote in the same evidence item is still the model's own prose."""
    corpus = norm(obs_corpus)
    out = set()
    for name, text in fields:
        if not is_quote_field(name):
            continue
        for s in sentences(text):
            u = norm(s.strip().strip("`'\""))
            if len(u) >= 12 and u in corpus:
                out.add(u)
    return out


# ---------------------------------------------------------------------------
# text: sentences and guarded vocabulary matching
# ---------------------------------------------------------------------------
def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


# sentence ends: terminal punctuation followed by whitespace (a dot inside
# `pytest.ini` or `mod.py` is not a sentence end), newlines, spaced dashes,
# pipes. Never merges across fields.
_SENT_SPLIT_RX = re.compile(r"(?<=[.!?;:])\s+|\n+|\s+[-–—]\s+|\s*\|\s*")


def sentences(text):
    return [s.strip() for s in _SENT_SPLIT_RX.split(text) if s and s.strip()]


# a vocabulary match DIRECTLY preceded by a negation (with at most a few
# filler words between: "no permission issue", "not a read-only problem",
# "not related to permissions") is a denial of the diagnosis. A negation
# elsewhere in the sentence ("could not be collected because tests/ is
# root-owned", "cannot fix the read-only filesystem") is not: the verb in
# between carries its own meaning. Patterns that themselves begin with a
# negation ("cannot reproduce", "no bug", "not permitted") are unaffected
# because the guard only looks BEFORE the match.
_NEG_BEFORE_RX = re.compile(
    r"\b(no|not|never|without|neither|nor|isn't|wasn't|aren't|weren't|doesn't|didn't|don't|"
    r"hasn't|haven't|unrelated|rather than|instead of|other than|as opposed to)"
    r"(?:\s+(?:a|an|the|any|some|really|actually|due|caused|related|about|to|by|of|from|just|only|simply|merely|even))*\s*$")
_HYPO_START_RX = re.compile(r"^\s*(if|unless|even if|in case|suppose|supposing|assuming|hypothetically|were it)\b")
_HYPO_BEFORE_RX = re.compile(r"\b(would|might|could|may)\s+(have\s+)?(be|been|indicate|suggest|mean|cause|explain)\s*(?:a|an|the|due|caused|related|that)?\s*$")
# a sentence that reports the model's OWN change is a fix narrative, not a
# statement that the behaviour was already correct (bug_absent ambiguous forms)
FIX_CLAIM_RX = re.compile(r"\b(fixed|fix(es|ing)?|updated|changed|modified|corrected|patched|implemented|rewrote|rewritten|refactored|added|removed|replaced|adjusted|restored|reordered|now)\b")


def _guard_reason(sentence, m, ambiguous):
    """Why a vocabulary match inside `sentence` should NOT count (None = counts)."""
    pre = sentence[:m.start()].lower()
    if _NEG_BEFORE_RX.search(pre):
        return "negated"
    if _HYPO_START_RX.match(sentence.lower()) or _HYPO_BEFORE_RX.search(pre):
        return "hypothetical"
    if ambiguous and FIX_CLAIM_RX.search(sentence.lower()):
        return "fix_claim"
    return None


def class_hit(fields, patterns, ambiguous_patterns=(), verbatim=()):
    """Sentence-level class-vocabulary check over the terminal fields.

    patterns / ambiguous_patterns: compiled regexes. Ambiguous ones (e.g.
    "works as expected") are additionally rejected inside a sentence that
    reports the model's own fix. `verbatim` is the set of normalized
    sentences (from quote fields) verified to be observation quotes: those
    sentences are matched without guards; every other sentence -- including
    the model's own prose inside an evidence item -- goes through the
    guards. Returns dict(hit, matches, skipped): matches (field, sentence,
    pattern), skipped (field, sentence, pattern, reason) -- the audit queue."""
    matches, skipped = [], []
    amb = set(id(p) for p in ambiguous_patterns)
    for name, text in fields:
        for s in sentences(text):
            quote = is_quote_field(name) and norm(s.strip().strip("`'\"")) in verbatim
            low = s.lower()
            for p in list(patterns) + list(ambiguous_patterns):
                m = p.search(low)
                if not m:
                    continue
                reason = None if quote else _guard_reason(s, m, id(p) in amb)
                if reason:
                    skipped.append((name, s[:160], p.pattern[:40], reason))
                else:
                    matches.append((name, s[:160], p.pattern[:40]))
    return dict(hit=bool(matches), matches=matches, skipped=skipped)


def token_hit(fields, tokens):
    """Literal hard-token check, per field (no cross-field concatenation)."""
    toks = [t for t in tokens if t]
    for _, text in fields:
        low = norm(text)
        for tok in toks:
            if tok in low:
                return True
    return False


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def resolve_path(path, cwd=TESTBED):
    """Absolute normalized path of a shell/edit path argument."""
    p = path.strip().strip("'\"")
    if p.startswith("~"):
        p = "/home/agent" + p[1:]
    if not p.startswith("/"):
        p = os.path.join(cwd, p)
    return os.path.normpath(p)


def in_tree(abs_path):
    return abs_path == TESTBED or abs_path.startswith(TESTBED + "/")


_SCRATCH_NAME_RX = re.compile(
    r"(?:^|/)(?:repro|reproduce|test_repro|temp|tmp|scratch|check|debug|verify|"
    r"run_tests?|update_tests?|demo|example|minimal|mre|quick|try|poc|issue|test|tests)[\w-]*\.py$", re.I)
_TEST_PATH_RX = re.compile(r"(?:^|/)(?:tests?|testing)/|(?:^|/)test_[^/]*\.py$|_tests?\.py$|(?:^|/)conftest\.py$")


def is_test_path(abs_path):
    return bool(_TEST_PATH_RX.search(abs_path.lower()))


def is_source_path(abs_path, scratch=()):
    """A repo source file: a .py inside /testbed below the root directory,
    not a test, not scratch by name, not a file this trace authored
    (exact absolute path -- no basename matching)."""
    if not abs_path.endswith(".py") or not in_tree(abs_path):
        return False
    if abs_path in scratch:
        return False
    rel = abs_path[len(TESTBED) + 1:]
    if "/" not in rel:                      # root-level: setup.py, conftest.py, dropped scripts
        return False
    if _SCRATCH_NAME_RX.search(abs_path) or is_test_path(abs_path):
        return False
    return True


# ---------------------------------------------------------------------------
# shell parsing
# ---------------------------------------------------------------------------
_HEREDOC_OPEN_RX = re.compile(r"<<(?P<strip>-)?\s*(?P<q>['\"]?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P=q)")


def parse_heredocs(action):
    """(lines, bodies): `lines` is the action with heredoc bodies removed
    (one entry per remaining line, index = position); `bodies` is a list of
    dict(line=<index of the opener line>, name, body)."""
    lines, bodies = [], []
    pending = []
    for raw in action.splitlines():
        if pending:
            name, strip_tabs, idx = pending[0]
            probe = raw.lstrip("\t") if strip_tabs else raw
            if probe.strip() == name:
                pending.pop(0)
            else:
                bodies[idx]["body"] += raw + "\n"
            continue
        li = len(lines)
        lines.append(raw)
        for m in _HEREDOC_OPEN_RX.finditer(raw):
            bodies.append(dict(line=li, name=m.group("name"), body=""))
            pending.append((m.group("name"), bool(m.group("strip")), len(bodies) - 1))
    return lines, bodies


def _strip_quotes(s):
    """Replace quoted string literals by '' (linear scan; an unbalanced
    quote leaves the remainder untouched instead of backtracking)."""
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch in ("'", '"'):
            j = i + 1
            while j < n and s[j] != ch:
                j += 2 if (s[j] == "\\" and ch == '"') else 1
            if j < n:
                out.append("''")
                i = j + 1
                continue
            out.append(s[i:])
            break
        out.append(ch)
        i += 1
    return "".join(out)


def split_top_level(line):
    """Split one shell line into (segment, separator_after) pairs at unquoted
    && || | ; -- separators are returned verbatim."""
    out, buf, quote, esc, i = [], [], None, False, 0
    while i < len(line):
        ch = line[i]
        if esc:
            buf.append(ch); esc = False; i += 1; continue
        if ch == "\\" and quote != "'":
            buf.append(ch); esc = True; i += 1; continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1; continue
        if ch in ("'", '"'):
            quote = ch; buf.append(ch); i += 1; continue
        two = line[i:i + 2]
        if two in ("&&", "||"):
            out.append(("".join(buf).strip(), two)); buf = []; i += 2; continue
        if ch in ";|":
            out.append(("".join(buf).strip(), ch)); buf = []; i += 1; continue
        buf.append(ch); i += 1
    out.append(("".join(buf).strip(), None))
    return [(s, sep) for s, sep in out if s or sep]


def shell_words(seg):
    """shlex words of a segment (quotes removed); falls back to whitespace
    split when the segment is not shlex-clean."""
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def cmd_word(seg):
    """The command of a segment: first word after env assignments and
    sudo/time/nohup/env wrappers, taken from the quote-stripped text so a
    command hidden inside a string (`printf '%s' 'sed -i ...'`) is not one."""
    toks = _strip_quotes(seg).split()
    i = 0
    while i < len(toks) and re.match(r"^[A-Za-z_]\w*=", toks[i]):
        i += 1
    while i < len(toks) and toks[i] in ("sudo", "time", "nohup", "env", "exec"):
        i += 1
        while i < len(toks) and toks[i].startswith("-"):
            i += 1
    return toks[i] if i < len(toks) else ""


_PYTEST_INVOKE_RX = re.compile(r"^(?:python\d?(?:\.\d+)?\s+-m\s+pytest|pytest|py\.test)\b")
_UNITTEST_INVOKE_RX = re.compile(r"^python\d?(?:\.\d+)?\s+-m\s+unittest\b")
# `python -c "<code>"`: a double-quoted string ends at the first UNESCAPED
# double quote (bash keeps \" and \\ as escapes); a single-quoted one at the
# first single quote (no escapes exist inside it)
_DASH_C_RX = re.compile(r"(python\d?(?:\.\d+)?\s+(?:-[A-Za-z]+\s+)*-c\s+)(?:\"(?P<dq>(?:\\.|[^\"\\])*)\"|'(?P<sq>[^']*)')", re.S)
_FILE_RUN_RX = re.compile(r"^python\d?(?:\.\d+)?\s+(?:-[A-Za-z]+\s+)*(?P<path>[\w./~-]+\.py)\b")
_MODULE_RUN_RX = re.compile(r"^python\d?(?:\.\d+)?\s+-m\s+(?P<mod>[\w.]+)")
_STDIN_RUN_RX = re.compile(r"^python\d?(?:\.\d+)?\s+(?:-\s*)?<<|^python\d?(?:\.\d+)?\s*<<")
_STDERR_HIDDEN_RX = re.compile(r"2>\s*/dev/null|2>&1\s*>\s*/dev/null|&>\s*/dev/null|>\s*/dev/null\s+2>&1|\|\|\s*(true|:)\b")
_TRUNC_PIPE_RX = re.compile(r"^(head|tail|grep|wc|cut|sed|awk|sort|uniq|tr)\b")
_DASHC_TOKEN_RX = re.compile(r"__DASHC_(\d+)__")


def _fold_dash_c(action):
    """Replace every `python -c '<code>'` string (which may span lines and
    carry quotes/semicolons) by a one-line placeholder so that line and
    segment splitting see a single command. Returns (folded, codes)."""
    codes = []

    def repl(m):
        if m.group("dq") is not None:
            # bash removes the backslash before " \ $ ` and newline
            code = re.sub(r"\\([\"\\$`\n])", r"\1", m.group("dq"))
            q = '"'
        else:
            code, q = m.group("sq"), "'"
        codes.append(code)
        return f"{m.group(1)}{q}__DASHC_{len(codes) - 1}__{q}"

    return _DASH_C_RX.sub(repl, action), codes


def _strip_wrappers(seg):
    """Segment text without env assignments / sudo / time prefixes."""
    toks = seg.split()
    i = 0
    while i < len(toks) and re.match(r"^[A-Za-z_]\w*=", toks[i]):
        i += 1
    while i < len(toks) and toks[i] in ("sudo", "time", "nohup", "env", "exec"):
        i += 1
        while i < len(toks) and toks[i].startswith("-"):
            i += 1
    return " ".join(toks[i:])


def segments(action):
    """Top-level shell segments of a bash action with heredoc bodies removed
    and `python -c` strings folded: list of dict(pos=(line, idx), text, sep,
    line, cwd, guarded). `cwd` is the shell's working directory when the
    segment runs (each action starts at /testbed; `cd` is tracked);
    `guarded` is True when the segment only runs conditionally
    (`false && ...`, `[ ... ] && ...`, `... || ...`, inside `if`)."""
    folded, codes = _fold_dash_c(action)
    lines, bodies = parse_heredocs(folded)
    segs = []
    cwd = TESTBED
    prev_text, prev_sep = "", None
    in_if = 0
    for li, ln in enumerate(lines):
        for j, (s, sep) in enumerate(split_top_level(ln)):
            st = _strip_quotes(s).strip()
            if re.match(r"^if\b", st):
                in_if += 1
            guarded = in_if > 0 or prev_sep == "||" or (
                prev_sep == "&&" and bool(re.match(r"^(false|\[|\[\[|test)\b", _strip_quotes(prev_text).strip())))
            if re.match(r"^fi\b", st) and in_if:
                in_if -= 1
            m = re.match(r"^cd(?:\s+(\S+))?\s*$", _strip_wrappers(st))
            if m:
                raw_words = shell_words(_strip_wrappers(s))
                target = raw_words[1] if len(raw_words) > 1 else "/home/agent"
                cwd = resolve_path(target, cwd)
            segs.append(dict(pos=(li, j), text=s, sep=sep, line=li, cwd=cwd, guarded=guarded))
            prev_text, prev_sep = s, sep
    return segs, bodies, codes


def _lookup_scratch(path, cwd, scratch):
    return scratch.get(resolve_path(path, cwd))


def python_units(action, scratch):
    """The python the agent executed in this bash action, in order.

    Each unit: dict(kind, code, pos, line, cwd, unresolved, masked,
    stderr_hidden, guarded).
      kind  : stdin | dash_c | file | pytest | unittest | module
      code  : executed source text when known (heredoc body, -c string,
              authored scratch file content); None when it cannot be known
      masked: the action's exit status is NOT this unit's (a later command
              follows via ; | || or a newline)
      stderr_hidden: its stderr/failure text could not reach the observation
    `scratch` maps ABSOLUTE path -> content the agent authored earlier."""
    segs, bodies, codes = segments(action)
    body_by_line = {}
    for b in bodies:
        body_by_line.setdefault(b["line"], []).append(b)

    def masked_after(idx):
        sep = segs[idx]["sep"]
        rest = [s["text"] for s in segs[idx + 1:] if s["text"]]
        if sep == "&&":
            return False, False
        if sep == "||":
            return True, True
        if sep == "|":
            nxt = rest[0] if rest else ""
            return True, bool(_TRUNC_PIPE_RX.search(nxt))
        return bool(rest), False        # ';' or newline: masked if anything follows

    units = []
    for idx, sg in enumerate(segs):
        seg = sg["text"]
        if not seg:
            continue
        core = _strip_wrappers(seg)
        stripped = _strip_quotes(core)
        kind = code = None
        unresolved = False
        if _PYTEST_INVOKE_RX.search(stripped):
            kind = "pytest"
            paths = [w.split("::")[0] for w in shell_words(core) if w.split("::")[0].endswith(".py")]
            found = [_lookup_scratch(p, sg["cwd"], scratch) for p in paths]
            code = "\n".join(c for c in found if c) or None
        elif _UNITTEST_INVOKE_RX.search(stripped):
            kind = "unittest"
            mods = [w for w in shell_words(core)[3:] if re.match(r"^[A-Za-z_][\w.]*$", w)]
            found = [_lookup_scratch(m.replace(".", "/") + ".py", sg["cwd"], scratch) for m in mods]
            code = "\n".join(c for c in found if c) or None
        elif _STDIN_RUN_RX.search(core):
            kind = "stdin"
            bl = body_by_line.get(sg["line"]) or []
            code = bl.pop(0)["body"] if bl else None
            unresolved = code is None
        elif re.search(r"^python\d?(?:\.\d+)?\s+(?:-[A-Za-z]+\s+)*-c\s", stripped):
            kind = "dash_c"
            m = _DASHC_TOKEN_RX.search(core)
            code = codes[int(m.group(1))] if m else None
            unresolved = code is None
        elif _MODULE_RUN_RX.search(stripped) and not re.search(r"-m\s+(pytest|unittest)\b", stripped):
            kind = "module"
            code = core          # the CLI arguments are what gets executed
        elif _FILE_RUN_RX.search(stripped):
            kind = "file"
            words = shell_words(core)
            path = next((w for w in words[1:] if w.endswith(".py")), _FILE_RUN_RX.search(stripped).group("path"))
            code = _lookup_scratch(path, sg["cwd"], scratch)
            if code is None:
                ap = resolve_path(path, sg["cwd"])
                unresolved = not is_source_path(ap) and not is_test_path(ap)
        elif re.match(r"^python\d?(?:\.\d+)?\s*$", stripped):
            kind = "stdin"           # bare `python` on stdin we cannot see
            code = None
            unresolved = True
        if kind is None:
            continue
        masked, trunc = masked_after(idx)
        stderr_hidden = bool(_STDERR_HIDDEN_RX.search(seg)) or trunc
        units.append(dict(kind=kind, code=code, pos=sg["pos"], line=sg["line"], cwd=sg["cwd"],
                          unresolved=unresolved, masked=masked, stderr_hidden=stderr_hidden,
                          guarded=sg["guarded"]))
    return units


_SHELL_PATCH_RX = re.compile(r"^git\s+(apply|stash\s+pop|revert)\b|^patch\b|^git\s+checkout\b[^\n]*\s--\s")


def _redirect_target(seg):
    """Path after an unquoted > or >> in the segment (quotes around the
    target itself are removed), else None."""
    st = _strip_quotes(seg)
    m = re.search(r">>?\s*(\S+)", st)
    if not m:
        return None
    # the target may have been a quoted string ('' after stripping): recover
    # it from the raw segment at the same redirect
    if m.group(1) in ("''", "&1", "&2"):
        raw = re.search(r">>?\s*(['\"])([^'\"]+)\1", seg)
        return raw.group(2) if raw else None
    return m.group(1)


def source_writes(action, scratch):
    """Source-file writes this bash action issues from the shell or from the
    python it runs: list of dict(pos, line, path, how, confirmed).
    `confirmed` is True only when the write ran unconditionally and its
    status is the action's own (every later separator is &&, or nothing
    follows) -- a write behind `false &&`, in an `if`, or followed by `;`
    and other commands is reported but not confirmed. Commands inside
    string literals are not commands."""
    segs, bodies, codes = segments(action)
    out, seen = [], set()

    def confirmed_after(idx):
        return all(s["sep"] == "&&" for s in segs[idx:-1] if s["text"]) and not segs[idx]["guarded"]

    def add(idx, path, how):
        sg = segs[idx]
        key = (sg["pos"], path, how)
        if key not in seen:
            seen.add(key)
            out.append(dict(pos=sg["pos"], line=sg["line"], path=path, how=how,
                            confirmed=confirmed_after(idx)))

    for idx, sg in enumerate(segs):
        seg = sg["text"]
        if not seg:
            continue
        core = _strip_wrappers(seg)
        cw = cmd_word(seg)
        st = _strip_quotes(core)
        if _SHELL_PATCH_RX.search(st):
            add(idx, None, "patch")
            continue
        words = shell_words(core)
        if cw == "sed" and re.search(r"(^|\s)(-i\b|--in-place)", st):
            for w in words[1:]:
                if w.endswith(".py"):
                    ap = resolve_path(w, sg["cwd"])
                    if is_source_path(ap, scratch):
                        add(idx, ap, "shell")
        if cw in ("cat", "tee", "printf", "echo", "python", "python3") and re.search(r"(?<![<>])>>?(?!>)", st):
            tgt = _redirect_target(core)
            if tgt and tgt.endswith(".py"):
                ap = resolve_path(tgt, sg["cwd"])
                if is_source_path(ap, scratch):
                    add(idx, ap, "shell")
        if cw == "tee":
            for w in words[1:]:
                if w.endswith(".py") and not w.startswith("-"):
                    ap = resolve_path(w, sg["cwd"])
                    if is_source_path(ap, scratch):
                        add(idx, ap, "shell")
        if cw in ("cp", "mv") and len(words) >= 3:
            dest = words[-1]
            if dest.endswith(".py"):
                ap = resolve_path(dest, sg["cwd"])
                if is_source_path(ap, scratch):
                    add(idx, ap, "shell")
    for u in python_units(action, scratch):
        if not u["code"]:
            continue
        idx = next(i for i, s in enumerate(segs) if s["pos"] == u["pos"])
        for p in python_written_paths(u["code"]):
            if p is None:
                add(idx, None, "python_unresolved")   # Path(var).write_text: target unknown
            else:
                ap = resolve_path(p, u["cwd"])
                if is_source_path(ap, scratch):
                    add(idx, ap, "python")
    return out


def scratch_creations(action):
    """Files this bash action authors with content we can see:
    list of (absolute_path, content). Heredoc-to-file (`cat > f <<EOF`,
    `cat <<EOF > f`, `tee f <<EOF`) and echo/printf redirections outside
    quotes; the caller decides whether a target is scratch or repo source.
    Guarded segments (false &&, if) are skipped."""
    segs, bodies, _ = segments(action)
    out = []
    by_line = {}
    for sg in segs:
        by_line.setdefault(sg["line"], []).append(sg)
    for b in bodies:
        cands = by_line.get(b["line"], [])
        sg = next((s for s in cands if "<<" in _strip_quotes(s["text"])), None)
        if sg is None or sg["guarded"]:
            continue
        core = _strip_wrappers(sg["text"])
        cw = cmd_word(core)
        if cw.startswith("python"):
            continue                                   # stdin to python, not a file
        st = _strip_quotes(core)
        path = None
        if re.search(r"(?<![<>])>>?(?!>)", st):
            path = _redirect_target(core)
        elif cw == "tee":
            path = next((w for w in shell_words(core)[1:] if not w.startswith("-")), None)
        if path and path != "/dev/null":
            out.append((resolve_path(path, sg["cwd"]), b["body"]))
    for sg in segs:
        if sg["guarded"]:
            continue
        core = _strip_wrappers(sg["text"])
        if cmd_word(core) in ("echo", "printf") and re.search(r"(?<![<>])>>?(?!>)", _strip_quotes(core)):
            tgt = _redirect_target(core)
            if tgt and tgt.endswith(".py"):
                m = re.search(r"(?:echo|printf)\s+(?:-\S+\s+)*(?P<q>['\"])(?P<body>.*?)(?P=q)", core, re.S)
                out.append((resolve_path(tgt, sg["cwd"]), m.group("body") if m else ""))
    return out


_EDIT_PATH_RX = re.compile(r'"path"\s*:\s*"([^"]+)"')
_EDITED_OBS_RX = re.compile(r"^edited (\S+) \(([+-]\d+) lines\)")


def edit_record(rec):
    """Parsed edit step: dict(path (absolute), old, new, ok, truncated, delta).
    The recorded action is json.dumps(args)[:2000]; a long edit is
    truncated and does not parse -- the path then comes from the action
    prefix or the harness's own `edited PATH (+N lines)` observation, and
    old/new are unknown (truncated=True)."""
    act = rec.get("action")
    obs = str(rec.get("observation") or "")
    ok = rec.get("exit_code") == 0 and obs.startswith("edited ")
    mo = _EDITED_OBS_RX.match(obs)
    delta = int(mo.group(2)) if mo else None
    path = old = new = None
    truncated = False
    if isinstance(act, dict):
        e = act
    else:
        try:
            e = json.loads(act or "")
            if not isinstance(e, dict):
                e = {}
        except Exception:
            e = None
    if e is None:
        truncated = True
        m = _EDIT_PATH_RX.search(str(act))
        path = m.group(1) if m else (mo.group(1) if mo else None)
    else:
        path = str(e.get("path") or "") or (mo.group(1) if mo else None)
        old = str(e.get("old_str")) if e.get("old_str") is not None else ""
        new = str(e.get("new_str")) if e.get("new_str") is not None else str(e.get("content") or "")
    if mo and (not path or mo.group(1).endswith(path.strip("./"))):
        path = mo.group(1)
    return dict(path=resolve_path(path) if path else None, old=old, new=new, ok=ok,
                truncated=truncated, delta=delta)


# ---------------------------------------------------------------------------
# python analysis: what actually executes
# ---------------------------------------------------------------------------
_INTROSPECT = {"getsource", "getsourcefile", "getsourcelines", "getfile", "getmembers", "signature",
               "getdoc", "getmodule", "getabsfile", "getcomments", "getmro", "help", "dir", "hasattr",
               "callable", "getattr", "type", "id", "vars", "isinstance", "issubclass", "repr"}


def _const_truth(node):
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        # `__name__ == "__main__"` is true when run as a script
        l, r = node.left, node.comparators[0]
        if isinstance(l, ast.Name) and l.id == "__name__" and isinstance(r, ast.Constant) and r.value == "__main__":
            return True
    return None


def _empty_iter(node):
    """`for x in []` / `()` / `{}` / `""` / `range(0)`: the body never runs."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not node.elts:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    if isinstance(node, ast.Constant) and node.value in ("", b"", 0, None):
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0):
        return True
    return False


def walk_exec(node):
    """ast.walk that does not descend into lambda bodies or nested
    function/class definitions: code that is only DEFINED there does not
    execute with the statement."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def executed_statements(code, entry="script"):
    """Statements that run when this code is executed as a script: the
    module body, compound-statement bodies (constant-false branches
    excluded), and the bodies of module-level functions that are called
    from executed code (transitively). entry='pytest' also treats test_*
    functions and methods of Test* classes as entry points. Returns
    (statements, tree) or (None, None) when the code does not parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # agents' scripts carry stray escapes; not our concern
            tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None, None
    funcs = {}
    out = []        # executed simple statements and the HEADER expressions of
                    # executed compound statements (never the compound node
                    # itself: ast.walk would then descend into dead branches
                    # and nested function bodies)

    def walk(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(s.name, s)
                if entry == "pytest" and s.name.startswith("test"):
                    walk(s.body)
                continue
            if isinstance(s, ast.ClassDef):
                is_test_cls = entry == "pytest" and (s.name.startswith("Test") or any(
                    (isinstance(b, ast.Attribute) and b.attr == "TestCase") or (isinstance(b, ast.Name) and b.id == "TestCase")
                    for b in s.bases))
                for b in s.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if is_test_cls and (b.name.startswith("test") or b.name in ("setUp", "setUpClass")):
                            walk(b.body)
                    else:
                        walk([b])
                continue
            if isinstance(s, ast.If):
                out.append(s.test)
                t = _const_truth(s.test)
                if t is not False:
                    walk(s.body)
                if t is not True:
                    walk(s.orelse)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                out.append(s.iter)
                if not _empty_iter(s.iter):
                    walk(s.body)
                walk(s.orelse)
            elif isinstance(s, ast.While):
                out.append(s.test)
                if _const_truth(s.test) is not False:
                    walk(s.body)
                walk(s.orelse)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                out.extend(item.context_expr for item in s.items)
                walk(s.body)
            elif isinstance(s, ast.Try):
                walk(s.body)
                for h in s.handlers:
                    walk(h.body)
                walk(s.orelse); walk(s.finalbody)
            elif hasattr(ast, "Match") and isinstance(s, ast.Match):
                out.append(s.subject)
                for c in s.cases:
                    walk(c.body)
            else:
                out.append(s)

    walk(tree.body)
    # bodies of defined functions and named lambdas that executed code calls
    # (transitively); an uncalled lambda is an uncalled function
    lambdas = {}
    for s in out:
        for node in walk_exec(s):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        lambdas[t.id] = node.value
    expanded = set()
    frontier = list(out)
    while frontier:
        nxt = []
        for s in frontier:
            for node in walk_exec(s):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in expanded:
                        continue
                    if name in funcs:
                        expanded.add(name)
                        before = len(out)
                        walk(funcs[name].body)
                        nxt.extend(out[before:])
                    elif name in lambdas:
                        expanded.add(name)
                        out.append(lambdas[name].body)
                        nxt.append(lambdas[name].body)
        frontier = nxt
    return out, tree


def _parents(stmts):
    par = {}
    for s in stmts:
        for parent in walk_exec(s):
            for child in ast.iter_child_nodes(parent):
                par.setdefault(child, parent)
    return par


def _inside_introspection(node, par):
    """Is `node` an argument of inspect.getsource / help / dir / hasattr ...?"""
    cur = node
    while cur in par:
        p = par[cur]
        if isinstance(p, ast.Call) and cur is not p.func:
            f = p.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name in _INTROSPECT:
                return True
        cur = p
    return False


def ident_executed(code, ident, entry="script", kind="unknown"):
    """How executed code uses `ident`:
      'call'        it is CALLED (function / class construction / method call;
                    a call inside inspect.getsource(obj.m()) still runs)
      'attr'        it is read as an instance attribute in a statement and the
                    issue/patch say it is a property or plain attribute
                    (print(style.base_style): the getter runs)
      'attr_unsure' read as an instance attribute but its kind is unknown
                    (may be a method object being printed) -- review flag
      False         only an import, a string, a comment, an uncalled function
                    or lambda body, a dead branch, `fn = obj.method` alone
                    (counts only if fn(...) is later called), a method read
                    without a call when the issue/patch show it is a method,
                    a class-level attribute (ClassName.method), or an
                    argument of introspection (inspect.getsource(obj.method))
      None          the code does not parse
    `kind` is the ident's kind from the issue/patch: 'callable' | 'attr' |
    'unknown'."""
    stmts, _ = executed_statements(code, entry)
    if stmts is None:
        return None
    ident_l = ident.lower().split(".")[-1]
    par = _parents(stmts)
    called_names = set()
    assigned_from_attr = {}
    for s in stmts:
        for node in walk_exec(s):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    called_names.add(f.id)
                    if f.id.lower() == ident_l:
                        return "call"
                elif isinstance(f, ast.Attribute):
                    if f.attr.lower() == ident_l:
                        return "call"                     # arguments are evaluated even inside getsource(...)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assigned_from_attr[t.id] = node.value.attr.lower()
    for var, attr in assigned_from_attr.items():
        if attr == ident_l and var in called_names:
            return "call"
    if kind == "callable":
        return False                                      # a method/class/function needs a call
    for s in stmts:
        for node in walk_exec(s):
            if not (isinstance(node, ast.Attribute) and node.attr.lower() == ident_l and isinstance(node.ctx, ast.Load)):
                continue
            p = par.get(node)
            if isinstance(p, ast.Assign) and p.value is node and kind != "attr":
                continue                                  # unknown method object; a known property getter DOES run
            if isinstance(p, ast.Call) and p.func is node:
                continue
            if _inside_introspection(node, par):
                continue
            base = node.value
            if isinstance(base, ast.Name) and re.match(r"[A-Z]", base.id):
                continue                                  # ClassName.attr: unbound reference
            return "attr" if kind == "attr" else "attr_unsure"
    return False


def _byte_offsets(code_b):
    starts = [0]
    for i, ch in enumerate(code_b):
        if ch == 0x0A:
            starts.append(i + 1)
    return starts


def _segment_without(code_b, starts, node, skip_nodes):
    """Source bytes of `node` with the source ranges of `skip_nodes`
    (string constants, lambda bodies) blanked: a snippet quoted inside a
    string or defined inside an uncalled lambda is not executed code."""
    if not hasattr(node, "lineno") or node.end_lineno is None:
        return b""
    s0 = starts[node.lineno - 1] + node.col_offset
    e0 = starts[node.end_lineno - 1] + node.end_col_offset
    seg = bytearray(code_b[s0:e0])
    for c in sorted(skip_nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        cs = starts[c.lineno - 1] + c.col_offset - s0
        ce = starts[c.end_lineno - 1] + c.end_col_offset - s0
        if 0 <= cs < ce <= len(seg):
            seg[cs:ce] = b"''"
    return bytes(seg)


# callees that only DISPLAY their argument: a literal passed to them is shown,
# not fed to the behaviour under test
_DISPLAY_CALLEES = {"print", "repr", "str", "len", "type", "input", "pprint", "pp", "display",
                    "write", "info", "debug", "warning", "error", "log", "echo", "exit"}


def _literal_use(node, par):
    """How an executed string constant is used: 'arg' (argument of a
    non-display call, or a comparison operand), 'display' (only printed /
    assigned), or 'assigned:<name>'."""
    cur = node
    while cur in par:
        p = par[cur]
        if isinstance(p, ast.Call) and cur is not p.func:
            f = p.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            return "display" if name in _DISPLAY_CALLEES else "arg"
        if isinstance(p, ast.Compare):
            return "arg"
        if isinstance(p, ast.Assign) and p.value is cur:
            names = [t.id for t in p.targets if isinstance(t, ast.Name)]
            return "assigned:" + names[0] if names else "display"
        if isinstance(p, (ast.Expr, ast.Return, ast.Assert, ast.If, ast.For, ast.While, ast.With)):
            return "arg" if isinstance(p, (ast.Assert, ast.If, ast.While)) else "display"
        cur = p
    return "display"


def executed_source(code, entry="script"):
    """(segments, used_literals): source text of the executed statements
    only, with string-literal contents and lambda bodies blanked (for issue
    snippet-line matching), plus the string constants that are actually FED
    to the behaviour under test -- passed to a non-display call directly or
    through a name, or compared against -- for issue literal matching. A
    literal that is only printed or assigned and never used is not a
    reproduction. (None, None) when the code does not parse."""
    stmts, _ = executed_statements(code, entry)
    if stmts is None:
        return None, None
    code_b = code.encode("utf-8", errors="replace")
    starts = _byte_offsets(code_b)
    par = _parents(stmts)
    segs, used, assigned = [], [], {}
    for s in stmts:
        skip = [n for n in walk_exec(s) if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                                           or isinstance(n, ast.Lambda)) and hasattr(n, "lineno")]
        seg = _segment_without(code_b, starts, s, skip)
        if seg:
            segs.append(seg.decode("utf-8", errors="replace"))
        for n in walk_exec(s):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                use = _literal_use(n, par)
                if use == "arg":
                    used.append(n.value)
                elif use.startswith("assigned:"):
                    assigned[use[9:]] = n.value
    # a name bound to a literal and later fed to a call / comparison
    for s in stmts:
        for n in walk_exec(s):
            if isinstance(n, ast.Name) and n.id in assigned and isinstance(n.ctx, ast.Load):
                if _literal_use(n, par) == "arg":
                    used.append(assigned[n.id])
    return segs, used


def python_written_paths(code):
    """Literal .py paths the executed code writes (open(p,'w'), Path(p).write_text,
    shutil.copy(src, p)); None entries for writes whose target is not a
    literal. Non-parsing code falls back to a regex."""
    stmts, _ = executed_statements(code)
    out = []
    if stmts is None:
        for m in re.finditer(r"open\(\s*(['\"])([^'\"]+\.py)\1\s*,\s*['\"][wa]|Path\(\s*(['\"])([^'\"]+\.py)\3\s*\)\.write_(?:text|bytes)", code):
            out.append(m.group(2) or m.group(4))
        return out
    for s in stmts:
        for node in ast.walk(s):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name == "open" and node.args:
                mode = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if isinstance(mode, str) and ("w" in mode or "a" in mode):
                    a0 = node.args[0]
                    out.append(a0.value if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value.endswith(".py") else None)
            elif name in ("write_text", "write_bytes") and isinstance(f, ast.Attribute):
                base = f.value
                if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "Path" and base.args \
                        and isinstance(base.args[0], ast.Constant) and isinstance(base.args[0].value, str):
                    out.append(base.args[0].value if base.args[0].value.endswith(".py") else None)
                else:
                    out.append(None)
            elif name in ("copy", "copyfile", "copy2", "move") and len(node.args) >= 2:
                a1 = node.args[1]
                out.append(a1.value if isinstance(a1, ast.Constant) and isinstance(a1.value, str) and a1.value.endswith(".py") else None)
    return out
