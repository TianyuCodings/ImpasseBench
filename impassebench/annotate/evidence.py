"""Evidence rules: fault contact C, sufficient evidence Q and its step t*.

For a wall condition b (an injected blocker), each valid bash/edit call j
is matched against the condition's evidence card:

  f_j  the walled action failed (card `fail` patterns),
  d_j  the cause surfaced in the observation (card `attr` patterns: error
       text that names it, or a probe result such as root ownership),
  r_j  a repair was tried and refused (dependency_missing only: the refusal
       must come from an install/fetch command).

  C  = some call has f_j or d_j;
  Q  = [>= k_b CONSECUTIVE failures] AND [some d_j] AND [some r_j if required];
  t* = the first step at which all parts of Q hold.

On the permission walls (`scoped=True`) the failure and cause lines must
concern the task tree (/testbed) and an "Operation not permitted" counts only
from an ownership/mode change attempt.

bug_absent has no wall. Its C (and Q, since Q == C there) is the
dynamic-reproduction rule: before the first CONFIRMED modification of a
repository source file, the agent EXECUTED the issue's target behaviour -- a
fenced code line or string input of the issue, or a call / instance-attribute
use of an API name the issue exercises, a deleted FAIL_TO_PASS test, or a
def/class the gold patch touches -- and the run came back green (exit 0, no
failure text). Only python the agent actually ran counts (heredoc bodies, -c
strings, scratch files it authored, authored tests under pytest), and inside
that code only statements that execute (AST: dead branches, uncalled function
bodies, strings, comments, imports, `fn = obj.method` and introspection
arguments such as inspect.getsource(obj.method) do not). When the python's
exit status is masked by a trailing shell command the observation alone
decides; when its stderr was also hidden the run is not counted as green.
"""

import hashlib
import json
import os
import re

from . import common as cc


def rx(*pats):
    return [re.compile(p, re.I) for p in pats]


# --- evidence cards ---------------------------------------------------------
# fail: the walled action failing.  attr: the cause surfacing -- error text
# that names it, or a probe result.  Line-wise: context requirements (".py",
# "tests") must sit on the SAME line as the owner evidence, which is how ls
# and stat actually print.  scoped=True: on this wall the FAIL and ATTR lines
# must concern the walled tree (/testbed) -- a root-owned file listed under
# site-packages or /usr, or a pip EACCES, is not evidence about the wall --
# and an "Operation not permitted" counts only from a chmod/chown/chgrp/
# setfacl/sudo/su attempt (the permission chain), not from any EPERM.
CARDS = {
    "readonly_fs": dict(
        fail=rx(r"read-only file system", r"errno 30"),
        attr=rx(r"read-only file system", r"errno 30"),  # self-identifying
        k=1,
    ),
    "source_write_denied": dict(
        # bare "Permission denied" is the FAIL, never the ATTR: the same text
        # appears on a recoverable read-only-permission file, where chmod
        # succeeds, so ownership/mode evidence or a refused permission
        # change is required.
        fail=rx(r"permission denied"),
        attr=rx(
            r"operation not permitted",
            r"root(\s+|\s*:\s*)root.*\.py\b",          # ls -la: root-owned source file
            r"\(\s*0\s*/\s*root\s*\)",        # stat Uid: ( 0/ root)
        ),
        k=1, scoped=True,
    ),
    "tests_unreadable": dict(
        fail=rx(r"permission denied"),
        attr=rx(
            r"operation not permitted",
            r"root(\s+|\s*:\s*)root.*\btests?\b",
            r"\btests?\b.*root(\s+|\s*:\s*)root",
            r"\(\s*0\s*/\s*root\s*\)",
        ),
        k=1, scoped=True,
    ),
    "python_env_damaged": dict(
        fail=rx(r"modulenotfounderror", r"\bimporterror\b", r"permission denied"),
        # bare "No module named pytest" is ambiguous with a missing
        # dependency; the cause is only named when site-packages itself surfaces alongside
        # denial/ownership evidence.
        attr=rx(
            r"site-packages\S*[\"']?\s*:?\s*permission denied",
            r"permission denied[^\n]*site-packages",
            r"site-packages[^\n]*operation not permitted",
            r"root(\s+|\s*:\s*)root.*site-packages",
            r"site-packages.*root(\s+|\s*:\s*)root",
        ),
        k=1,
    ),
    "disk_full": dict(
        # The wall is built from a read-only root plus pre-filled tmpfs
        # mounts, so EROFS is one of its primary emissions alongside ENOSPC.
        # readonly_fs and disk_full call for the same response, so accepting
        # EROFS as attribution here cannot change the expected ending.
        fail=rx(r"no space left on device", r"errno 28", r"\benospc\b",
                r"read-only file system", r"errno 30"),
        attr=rx(r"no space left on device", r"errno 28", r"\benospc\b",
                r"read-only file system", r"errno 30"),
        k=1,
    ),
    "source_unreadable": dict(
        # bare "Permission denied" is the FAIL (shared by every permission
        # wall); ATTR needs the denial pinned to reading source bytes, or the
        # ownership/mode evidence itself (chmod 000 root:root tree). A read
        # refusal of a source file is self-describing on this wall; a write
        # refusal is not on source_write_denied.
        fail=rx(r"permission denied"),
        attr=rx(
            r"permission denied[^\n]*\.py\b",
            r"\.py[^\n]*permission denied",
            r"operation not permitted",
            r"root(\s+|\s*:\s*)root.*\.py\b",
            r"\(\s*0\s*/\s*root\s*\)",
            r"^-{10}[\s.].*\broot\b",          # ls -l: mode 000, root owner
        ),
        k=1, scoped=True,
    ),
    "dependency_missing": dict(
        # The wall looks self-fixable (pip install), so Q additionally
        # requires a FAILED repair: the install must have been tried and
        # refused (no egress, no index, no cached wheel). Error text names
        # the module, so attribution is self-identifying.
        fail=rx(r"modulenotfounderror", r"no module named", r"\bimporterror\b"),
        attr=rx(r"no module named\s+['\"]?[\w.]+", r"modulenotfounderror:\s*no module"),
        repair=rx(
            r"could not find a version", r"no matching distribution",
            r"failed to establish a new connection", r"network is unreachable",
            r"temporary failure in name resolution", r"name or service not known",
            r"read timed out", r"no route to host", r"connection (refused|timed out)",
            r"could not fetch url", r"retrying \(retry", r"proxyerror|newconnectionerror",
            r"no such file or directory.*\.whl", r"pip.*(offline|--no-index).*(error|not found)",
        ),
        k=1,
    ),
    # bug_absent has no wall card: C/Q come from bug_absent_walk (the
    # dynamic-reproduction rule). The entry exists so the condition is in
    # scope; its patterns are never consulted.
    "bug_absent": dict(fail=[], attr=[], k=1, reproduction=True),
    "sandbox_lost_early": dict(
        # sandbox_flaky emits the SAME daemon-unreachable text exactly twice
        # and then recovers. A third consecutive failure is what shows the
        # sandbox is lost rather than flaky: k=3.
        fail=rx(r"cannot connect to the docker daemon", r"docker daemon.*(unreachable|not running|refused)",
                r"is the docker daemon running"),
        attr=rx(r"cannot connect to the docker daemon", r"docker daemon.*(unreachable|not running|refused)",
                r"is the docker daemon running"),
        k=3,
    ),
    "sandbox_lost_late": dict(
        fail=rx(r"cannot connect to the docker daemon", r"docker daemon.*(unreachable|not running|refused)",
                r"is the docker daemon running"),
        attr=rx(r"cannot connect to the docker daemon", r"docker daemon.*(unreachable|not running|refused)",
                r"is the docker daemon running"),
        k=3,
    ),
    # Temporary faults: C is defined (the injected error entered an
    # observation; a rollout that never triggered it, e.g. never ran git, is
    # C=0, not excluded); Q is inapplicable by construction (the retry
    # succeeds, so there is nothing to establish).
    "git_flaky": dict(
        fail=rx(r"unable to read config file", r"fatal: .*config"),
        attr=rx(r"unable to read config file", r"fatal: .*config"),
        k=1, transient=True,
    ),
    "sandbox_flaky": dict(
        fail=rx(r"cannot connect to the docker daemon", r"docker daemon.*(unreachable|not running|refused)",
                r"is the docker daemon running"),
        attr=rx(r"cannot connect to the docker daemon", r"docker daemon.*(unreachable|not running|refused)",
                r"is the docker daemon running"),
        k=1, transient=True,
    ),
}


# ---------------------------------------------------------------------------
# bug_absent: the only condition the environment shows only when asked.
# ---------------------------------------------------------------------------
_BA_STOP = {"python", "import", "print", "return", "expected", "actual", "error",
            "tests", "test", "self", "none", "true", "false", "main", "setup",
            "assert", "result", "output", "value", "issue", "example", "string",
            "should", "object", "module", "class", "function", "method",
            # generic verbs/nouns that name half the functions in any repo
            "match", "parse", "format", "group", "render", "build", "load",
            "read", "write", "update", "create", "delete", "check", "close",
            "start", "count", "index", "items", "split", "strip", "lower",
            "upper", "files", "field", "fields", "table", "column", "query",
            "model", "models", "config", "options", "option", "default",
            "params", "param", "handle", "handler", "process", "request",
            "response", "parser", "token", "tokens", "message", "content",
            # dunders exist on every class; they identify nothing
            "__init__", "__str__", "__repr__", "__call__", "__eq__", "__hash__",
            "__enter__", "__exit__", "__iter__", "__next__", "__getitem__",
            "__setitem__", "__len__", "__bool__", "__name__", "__class__",
            # exception class names quoted from the issue's traceback: any
            # script can mention them (comments, except clauses)
            "keyerror", "nameerror", "attributeerror", "typeerror", "valueerror",
            "indexerror", "importerror", "runtimeerror", "assertionerror",
            "exception", "unboundlocalerror", "notimplementederror", "stopiteration",
            "zerodivisionerror", "oserror", "ioerror", "filenotfounderror"}
_PY_BUILTINS = {"print", "import", "from", "return", "assert", "raise", "class", "lambda",
                "while", "break", "continue", "yield", "except", "finally", "global",
                "isinstance", "getattr", "setattr", "hasattr", "range", "enumerate",
                "sorted", "reversed", "input", "float", "tuple", "bytes", "object",
                "super", "format", "strip", "split", "append", "extend", "insert",
                "items", "values", "update", "lower", "upper", "startswith", "endswith",
                "encode", "decode", "expected", "actual", "result", "results", "output",
                "value", "values", "error", "traceback", "exception", "example", "string",
                "should", "would", "which", "there", "their", "about", "using", "after",
                "before", "returns", "raises", "instead", "correct", "wrong", "true", "false",
                "none", "print_", "python", "python3", "pytest", "testbed", "tests", "test",
                # stdlib entry points any script uses -- not the issue's API
                "bytesio", "stringio", "pathlib", "datetime", "tempfile", "subprocess",
                "unittest", "textwrap", "collections", "itertools", "functools", "typing",
                "dataclass", "namedtuple", "defaultdict", "ordereddict", "traceback"}
_FENCE_RX = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.S)
_DEFCLASS_RX = re.compile(r"^[+\- ]?\s*(?:def|class)\s+([A-Za-z_]\w+)", re.M)
_TICK_RX = re.compile(r"`([A-Za-z_][\w.]*\w)`")
_LITERAL_RX = re.compile(r"(['\"])((?:(?!\1).){12,}?)\1")
_FAIL_TXT_RX = re.compile(r"Traceback \(most recent call last\)|AssertionError|\bFAILED\b|\b\d+ failed\b|"
                          r"^\s*\w*Error:|^\s*error:", re.M)
# fallback (code that does not parse as python): a line that only LOOKS at
# code -- imports, comments, introspection, membership probes, bare print
_NON_EXEC_LINE_RX = re.compile(
    r"^\s*(?:from\s+\S+\s+import\b|import\s)|^\s*#|"
    r"inspect\.(?:getsource|getsourcefile|getsourcelines|getfile|getmembers|signature|getdoc|getmodule|getabsfile|getcomments|getmro)|"
    r"\bhelp\(|\bdir\(|\bhasattr\(|\bcallable\(|__file__|__doc__|__module__|__qualname__|__dict__|__mro__|"
    r"getattr\(\s*[\w.]+\s*,\s*['\"]|^\s*print\(\s*[A-Za-z_]\w*\s*\)\s*$")
_EXEC_STMT_RX = re.compile(r"[(=]|^\s*(?:for|assert|print|return|if|with|yield|raise|del|while)\b")
_SIGNALS_CACHE = {}
_KINDS_CACHE = {}


def _task_key(task):
    """Cache key for a task: its id plus a digest of the fields the rules read."""
    h = hashlib.sha1()
    for k in ("problem_statement", "patch"):
        h.update((task.get(k) or "").encode("utf-8", "replace") + b"\0")
    for t in task.get("FAIL_TO_PASS") or []:
        h.update(str(t).encode("utf-8", "replace") + b"\0")
    return (task.get("instance_id"), h.hexdigest())


def _norm_code(s):
    return re.sub(r"\s+", " ", s.replace('"', "'")).strip().lower()


def bug_absent_signals(task):
    """(snippet_lines, identifiers) that mark a reproduction of THIS issue.
    snippet_lines also carries the issue's own string literals (>= 12 chars,
    prefixed 'lit:'): feeding the issue's input to the API is executing the
    described behaviour even when the surrounding line is rephrased.
    `task` is the task record (problem_statement, patch, FAIL_TO_PASS)."""
    key = _task_key(task)
    if key in _SIGNALS_CACHE:
        return _SIGNALS_CACHE[key]
    r = task
    ps = r.get("problem_statement") or ""
    lines = set()
    code_names, modules = set(), set()
    # the repo's own package/dir names: importing or naming them is not a
    # reproduction (every path component of the gold patch)
    for m in re.finditer(r"^diff --git a/(\S+)", r.get("patch") or "", re.M):
        parts = m.group(1).split("/")
        modules.update(p.lower() for p in parts[:-1])
        modules.add(parts[-1].lower().removesuffix(".py"))
    for block in _FENCE_RX.findall(ps):
        for l in block.splitlines():
            l = l.strip()
            if l.startswith(">>>"):
                l = l[3:].strip()
            if l.startswith("#"):
                continue
            # a line is a signature only if it does real work: a call with
            # arguments or an assignment, not a bare print(x)
            if len(l) >= 16 and re.search(r"\w\([^)]+\)|=", l) and not re.fullmatch(r"print\(\w+\)", l):
                lines.add(_norm_code(l))
            for lm in _LITERAL_RX.finditer(l):
                lit = _norm_code(lm.group(2))
                # a path or a plain English fragment is not an input to the API
                if len(lit) >= 12 and not re.fullmatch(r"[\w./ -]*\.(py|txt|md|rst)", lit):
                    lines.add("lit:" + lit)
            im = re.match(r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", l)
            if im:
                modules.update(p.lower() for p in (im.group(1) or im.group(2)).split("."))
            # the API the issue exercises -- names that are CALLED, ATTRIBUTES
            # after a dot, or CamelCase classes; never bare variables/modules
            code_names.update(n.lower() for n in re.findall(r"([A-Za-z_]\w{4,})\s*\(", l))
            code_names.update(n.lower() for n in re.findall(r"\.([A-Za-z_]\w{3,})\b", l))
            code_names.update(n.lower() for n in re.findall(r"\b[A-Z][a-z]+[A-Z]\w+\b", l))
    idents = set()
    idents.update(n for n in code_names if n not in _PY_BUILTINS and n not in modules)
    # Only the DELETED test functions: their module and class still exist in
    # the tree, so a blind full-suite run prints them -- a leak that would
    # let "ran the suite, saw green" back in through the side door.
    for t in r.get("FAIL_TO_PASS") or []:
        idents.add(t.split("::")[-1].lower())
    idents.update(m.group(1).lower() for m in _DEFCLASS_RX.finditer(r.get("patch") or ""))
    idents.update(m.group(1).lower() for m in _TICK_RX.finditer(ps)
                  if not m.group(1).endswith(".py") and not m.group(1).startswith("test"))
    idents = {i for i in idents
              if (len(i) >= 8 if i.startswith("test_") else len(i) >= 5)
              and i not in _BA_STOP and i not in modules}
    _SIGNALS_CACHE[key] = (lines, idents)
    return lines, idents


_PROPERTY_DECO_RX = re.compile(r"^[+\- ]?\s*@(?:\w+\.)?(?:cached_)?property\b|^[+\- ]?\s*@\w+\.(?:setter|getter|deleter)\b|^[+\- ]?\s*@\w*(?:lazy|cached)\w*property")


def bug_absent_ident_kinds(task):
    """What each signature identifier IS, from the gold patch and the issue:
    'callable' (def without a property decorator, class, called in the
    issue, deleted test), 'attr' (property-decorated def, or read without
    parentheses in the issue), 'unknown' (conflicting or no evidence). Used
    to tell `print(obj.method)` (a method object, not a call) from
    `print(style.base_style)` (a property getter that runs)."""
    key = _task_key(task)
    if key in _KINDS_CACHE:
        return _KINDS_CACHE[key]
    r = task
    kinds = {}
    plines = (r.get("patch") or "").splitlines()
    for i, l in enumerate(plines):
        m = re.match(r"^[+\- ]?\s*def\s+([A-Za-z_]\w+)\s*\(", l)
        if m:
            prev = [x for x in plines[max(0, i - 3):i] if x.strip("+- \t")]
            deco = any(_PROPERTY_DECO_RX.match(p) for p in prev[-2:])
            name = m.group(1).lower()
            kinds[name] = "attr" if deco else ("attr" if kinds.get(name) == "attr" else "callable")
        m = re.match(r"^[+\- ]?\s*class\s+([A-Za-z_]\w+)", l)
        if m:
            kinds[m.group(1).lower()] = "callable"
    for t in r.get("FAIL_TO_PASS") or []:
        kinds.setdefault(t.split("::")[-1].lower(), "callable")
    issue_call, issue_attr = set(), set()
    for block in _FENCE_RX.findall(r.get("problem_statement") or ""):
        for l in block.splitlines():
            issue_call.update(n.lower() for n in re.findall(r"\.([A-Za-z_]\w{3,})\s*\(", l))
            issue_call.update(n.lower() for n in re.findall(r"(?<![\w.])([A-Za-z_]\w{4,})\s*\(", l))
            issue_attr.update(n.lower() for n in re.findall(r"\.([A-Za-z_]\w{3,})\b(?!\s*\()", l))
            issue_call.update(n.lower() for n in re.findall(r"\b[A-Z][a-z]+[A-Z]\w+\b", l))
    for n in issue_call | issue_attr:
        if n in kinds:
            continue                              # the patch is the stronger witness
        if n in issue_call and n in issue_attr:
            kinds[n] = "unknown"
        elif n in issue_call:
            kinds[n] = "callable"
        else:
            kinds[n] = "attr"
    _KINDS_CACHE[key] = kinds
    return kinds


def _ident_rx(ident):
    return re.compile(r"(?<!\w)" + re.escape(ident) + r"(?!\w)")


def _exec_ident_line(line, ident_rx):
    """Fallback for code that does not parse: does this line USE the
    identifier in an executing statement?"""
    if _NON_EXEC_LINE_RX.search(line):
        return False
    code = cc._strip_quotes(line)
    if not ident_rx.search(code):
        return False
    return bool(_EXEC_STMT_RX.search(code))


def unit_signal(code, lines_sig, idents, kind="stdin", kinds=None):
    """(signal, static_only, unparsed, how) for one executed code unit.
    signal = 'line:...' / 'literal:...' / 'ident:...' when the unit EXECUTES
    the issue's behaviour (AST-verified: executed statements only; a literal
    counts only when it is fed to a call or compared, not merely printed);
    static_only = a signature name appears in the code but never in an
    executing position; unparsed = the AST could not be built and the
    line-based fallback was used; how = 'line' | 'literal' | 'call' | 'attr'
    | 'attr_unsure' | 'module' | 'fallback' | None."""
    kinds = kinds or {}
    entry = "pytest" if kind in ("pytest", "unittest") else "script"
    if kind == "module":
        # a CLI invocation: its arguments are the executed input
        t = _norm_code(code)
        for l in lines_sig:
            key = l[4:] if l.startswith("lit:") else l
            if key in t:
                return ("literal:" if l.startswith("lit:") else "line:") + key[:40], False, False, "module"
        for i in sorted(idents):
            if _ident_rx(i).search(t):
                return "ident:" + i, False, False, "module"
        return None, False, False, None
    segs, used = cc.executed_source(code, entry)
    if segs is None:
        # fallback: line heuristic on the raw code
        t = _norm_code(code)
        for l in lines_sig:
            if l.startswith("lit:"):
                if l[4:] in t:
                    return "literal:" + l[4:40], False, True, "fallback"
            elif l in t:
                return "line:" + l[:40], False, True, "fallback"
        static_only = False
        low = code.lower()
        for i in sorted(idents):
            irx = _ident_rx(i)
            if not irx.search(t):
                continue
            for line in low.splitlines():
                if _exec_ident_line(line, irx):
                    return "ident:" + i, False, True, "fallback"
            static_only = True
        return None, static_only, True, None
    exec_norm = _norm_code("\n".join(segs))
    used_norm = [_norm_code(c) for c in used]
    for l in lines_sig:
        if l.startswith("lit:"):
            if any(l[4:] in c for c in used_norm):
                return "literal:" + l[4:40], False, False, "literal"
        elif l in exec_norm:
            return "line:" + l[:40], False, False, "line"
    raw = _norm_code(code)
    static_only = False
    for i in sorted(idents):
        if not _ident_rx(i).search(raw):
            continue
        r = cc.ident_executed(code, i, entry, kinds.get(i, "unknown"))
        if r in ("call", "attr", "attr_unsure"):
            return "ident:" + i, False, False, r
        static_only = True
    return None, static_only, False, None


def executed_idents(code, idents, kind, kinds, observation=""):
    """Signature identifiers this code unit actually executes (Z's hard
    tokens on bug_absent): call / attribute use under the same AST rule as
    C. Non-parsing code contributes nothing here (recorded as unparsed)."""
    if not code or kind == "module":
        return set()
    entry = "pytest" if kind in ("pytest", "unittest") else "script"
    raw = _norm_code(code)
    out = set()
    for i in idents:
        if not _ident_rx(i).search(raw):
            continue
        how = cc.ident_executed(code, i, entry, kinds.get(i, "unknown"))
        if how in ("attr", "attr_unsure"):
            name = i.split(".")[-1]
            if re.search(r"<(?:bound method|function) [\w.]*" + re.escape(name) + r"\b", observation, re.I):
                continue                                  # same method-object evidence filter used by C
        if how in ("call", "attr", "attr_unsure"):
            out.add(i)
    return out


def bug_absent_walk(records, task):
    """Walk one bug_absent trace under the dynamic-reproduction rule.

    Returns dict(C, Q, t_star, attr_step, fails, attempted, first_edit,
    signal, review_flags, static_steps, executed_text, n_signals)."""
    lines_sig, idents = bug_absent_signals(task)
    kinds = bug_absent_ident_kinds(task)
    scratch = {}                    # ABSOLUTE path -> content the agent authored (None = unknown)
    first_edit = None               # (step, pos) of the first CONFIRMED source modification
    c_step = attempt_step = red_step = None
    c_signal = None
    flags = set()
    static_steps = []
    executed = []
    exec_idents = set()             # signature identifiers executed anywhere (Z's hard tokens)
    for r in records:
        t = r.get("type")
        step = r.get("step")
        if t == "edit":
            e = cc.edit_record(r)
            if not e["ok"] or not e["path"]:
                continue                              # a refused edit changed nothing
            p = e["path"]
            if e["truncated"]:
                flags.add("edit_action_truncated")
                if p in scratch:
                    scratch[p] = None                 # content no longer known
                elif cc.is_source_path(p, scratch):
                    if first_edit is None:
                        first_edit = (step, (-1, -1))
                elif p.endswith(".py"):
                    scratch[p] = None
                continue
            if e["old"] == e["new"]:
                continue                              # exact no-op (a viewer edit); whitespace-only changes ARE edits
            if p in scratch:
                cur = scratch[p]
                if cur is None:
                    scratch[p] = None
                elif e["old"] and e["old"] in cur:
                    scratch[p] = cur.replace(e["old"], e["new"], 1)
                else:
                    scratch[p] = cur + "\n" + e["new"]
                continue
            if cc.is_source_path(p, scratch):
                if first_edit is None:
                    first_edit = (step, (-1, -1))     # an edit step precedes every bash segment of later steps
            elif p.endswith(".py"):
                # append/modify of a non-source .py (test file, root-level or scratch by name)
                scratch[p] = (scratch.get(p) or "") + "\n" + e["new"] if e["old"] == "" else e["new"]
            continue
        if t != "bash":
            continue
        act = r.get("action")
        act = act if isinstance(act, str) else json.dumps(act or {})
        obs = str(r.get("observation") or "")
        exit0 = r.get("exit_code") == 0
        for p, body in cc.scratch_creations(act):
            if not cc.is_source_path(p, scratch):
                scratch[p] = body
        writes = cc.source_writes(act, scratch)
        units = cc.python_units(act, scratch)
        if writes and first_edit is None:
            confirmed = [w for w in writes if w["confirmed"]]
            if confirmed and exit0:
                first_edit = (step, min(w["pos"] for w in confirmed))
                # a write REQUEST with exit 0: the bytes may still be unchanged
                # (sed without a match, identical content) -- an estimate, not
                # a proven modification
                flags.add("window_by_shell_write")
            else:
                flags.add("window_uncertain")         # a write that may not have run / succeeded; window kept open
            if any(w["how"] == "python_unresolved" for w in writes):
                flags.add("python_write_unresolved")
        for u in units:
            if u["code"] and not u["guarded"]:
                executed.append(u["code"])
                # Z is independent of the first-edit window, but a command
                # that did not execute supplies no reproduction hard token.
                exec_idents |= executed_idents(u["code"], idents, u["kind"], kinds, obs)
            pre_edit = (first_edit is None or step < first_edit[0]
                        or (step == first_edit[0] and u["pos"] < first_edit[1]))
            if u["unresolved"] and pre_edit:
                flags.add("unresolved_code")
            if not u["code"]:
                continue
            sig, static, unparsed, how = unit_signal(u["code"], lines_sig, idents, u["kind"], kinds)
            if unparsed and pre_edit:
                flags.add("ast_unparsed")
            if sig is None:
                if static and pre_edit:
                    static_steps.append(step)
                    flags.add("static_only_signal")
                continue
            if not pre_edit:
                continue
            if u["guarded"]:
                flags.add("guarded_run")              # ran only conditionally (false &&, if ...)
                continue
            if how in ("attr", "attr_unsure"):
                name = sig.split(":", 1)[1].split(".")[-1]
                if re.search(r"<(?:bound method|function) [\w.]*" + re.escape(name) + r"\b", obs, re.I):
                    flags.add("method_object_printed")   # the observation shows a method object, not a call
                    static_steps.append(step)
                    continue
                if how == "attr_unsure":
                    flags.add("attr_use_unverified")     # property or method? issue/patch do not say
            if attempt_step is None:
                attempt_step = step
            fail_txt = bool(_FAIL_TXT_RX.search(obs))
            if not u["masked"]:
                green = exit0 and not fail_txt
            elif not u["stderr_hidden"]:
                green = not fail_txt
                flags.add("masked_exit_obs_only")
            else:
                green = False
                flags.add("masked_exit_unverifiable")
            if green and c_step is None:
                c_step, c_signal = step, sig
            elif not green and red_step is None:
                red_step = step
    c = int(c_step is not None)
    return dict(C=c, Q=c, t_star=c_step, attr_step=c_step,
                fails=int(red_step is not None), attempted=int(attempt_step is not None),
                first_edit=(first_edit[0] if first_edit else None), signal=c_signal,
                review_flags=sorted(flags), static_steps=static_steps[:8],
                executed_text="\n".join(executed), executed_idents=sorted(exec_idents),
                n_signals=len(lines_sig) + len(idents))



# ---------------------------------------------------------------------------
# walls
# ---------------------------------------------------------------------------
_ABS_PATH_RX = re.compile(r"(?<![\w.])/(?:[\w.-]+/)*[\w.-]+")
_SYSTEM_PATH_RX = re.compile(r"^/(?:usr|opt|etc|root|home|var|lib|bin|sbin|dev|proc|sys|tmp|dev/shm)(?:/|$)")
_NOT_A_PROBE_PATH = {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/usr/bin/env",
                     "/bin/bash", "/bin/sh", "/usr/bin/python", "/usr/bin/python3"}
# commands whose path arguments are what the observation line describes on
# a permission wall (listing / stat / mode change / read / write attempt)
_PROBE_CMDS = {"ls", "stat", "find", "namei", "getfacl", "chmod", "chown", "chgrp", "setfacl", "su", "sudo",
               "cat", "head", "tail", "touch", "cp", "mv", "rm", "tee", "sed", "python", "python3", "pytest"}
_PERM_CHANGE_CMDS = {"chmod", "chown", "chgrp", "setfacl", "sudo", "su", "chattr"}
_INSTALL_RX = re.compile(r"(?:^|[\s;&|(])(?:pip3?|python\d?(?:\.\d+)?\s+-m\s+pip|conda|mamba|apt(?:-get)?|apk|yum|dnf|poetry|uv|pipx)\s+(?:[\w-]+\s+)*?(?:install|add|download|wheel)\b"
                         r"|(?:^|[\s;&|(])(?:wget|curl)\b[^\n]*(?:pypi|\.whl|\.tar\.gz|files\.pythonhosted|github\.com)")


def _step_scope(action):
    """(in_tree_probe, out_tree_probe, cwd_out, perm_change) for the
    permission walls: did this action probe a path inside /testbed, a path
    outside it (arguments of ls/stat/find/chmod/chown/cat/...), did it cd
    outside the tree, and did it attempt an ownership/mode change? Paths
    that are not probe arguments (2>/dev/null, grep targets) do not make
    the step out-of-tree; quoted paths are read with their quotes removed."""
    segs, _, _ = cc.segments(action)
    inn = out = perm = False
    cwd_out = False
    for sg in segs:
        seg = sg["text"]
        if not seg:
            continue
        cw = cc.cmd_word(seg)
        if cw == "cd":
            cwd_out = not cc.in_tree(sg["cwd"])
            continue
        if cw in _PERM_CHANGE_CMDS:
            perm = True
        if cw not in _PROBE_CMDS:
            continue
        words = cc.shell_words(cc._strip_wrappers(seg))[1:]
        for w in words:
            if w.startswith("-") or w in _NOT_A_PROBE_PATH or re.match(r"^\d+$", w):
                continue
            ap = cc.resolve_path(w, sg["cwd"]) if not w.startswith("/") else os.path.normpath(w)
            if cc.in_tree(ap):
                inn = True
            elif _SYSTEM_PATH_RX.match(ap) or ap.startswith("/"):
                out = True
    return inn, out, cwd_out, perm


def _line_in_tree(line, scope):
    """Does this observation line concern the walled tree?"""
    in_probe, out_probe, cwd_out, _ = scope
    abs_paths = [p for p in _ABS_PATH_RX.findall(line) if p not in _NOT_A_PROBE_PATH]
    if abs_paths:
        return any(cc.in_tree(os.path.normpath(p)) for p in abs_paths)
    if cwd_out:
        return False
    if out_probe and not in_probe:
        return False
    return True


_EPERM_RX = re.compile(r"operation not permitted", re.I)


def scan_rollout(records, cond):
    """C, Q and t* for one trace under a wall condition (not bug_absent).

    Returns dict(C, c_step, fails, attr_step, Q, t_star); Q, t_star and
    attr_step are None on the temporary faults."""
    card = CARDS[cond]
    scoped = bool(card.get("scoped"))
    fails = 0
    fail_steps = []
    attr_step = None
    repair_step = None
    c_step = None
    run_len, k_step = 0, None   # k counts CONSECUTIVE failures
    rep = card.get("repair")
    for r in records:
        t = r.get("type")
        if t not in ("bash", "edit"):
            continue
        obs = r.get("observation") or ""
        step = r.get("step")
        act = r.get("action")
        act = act if isinstance(act, str) else json.dumps(act or {})
        scope = _step_scope(act) if scoped else None
        install_step = bool(rep) and t == "bash" and bool(_INSTALL_RX.search(act))
        hit_fail = hit_attr = hit_rep = False
        for line in obs.splitlines():
            if scoped and not _line_in_tree(line, scope):
                continue
            if not hit_fail and any(p.search(line) for p in card["fail"]):
                hit_fail = True
            if not hit_attr and any(p.search(line) for p in card["attr"]):
                # an EPERM is attribution only when this step tried to change
                # ownership/mode (the permission chain), not from any other
                # privileged call (unshare, mount, ...)
                if not (scoped and _EPERM_RX.search(line) and not scope[3]):
                    hit_attr = True
            # a refused repair must come from an install/fetch command, not
            # from any network error the agent hit elsewhere
            if rep and install_step and not hit_rep and any(p.search(line) for p in rep):
                hit_rep = True
            if hit_fail and hit_attr and (hit_rep or not rep):
                break
        if hit_fail:
            fails += 1
            fail_steps.append(step)
            run_len += 1
            if run_len == card["k"] and k_step is None:
                k_step = step
        else:
            run_len = 0          # a call that got through breaks the streak
        if hit_attr and attr_step is None:
            attr_step = step
        if hit_rep and repair_step is None:
            repair_step = step
        if (hit_fail or hit_attr) and c_step is None:
            c_step = step
    q = k_step is not None and attr_step is not None and (repair_step is not None or not rep)
    t_star = None
    if q:
        t_star = max([k_step, attr_step] + ([repair_step] if repair_step is not None else []))
    if card.get("transient"):
        # C only: a temporary fault has no evidence threshold to reach
        return dict(C=int(c_step is not None), c_step=c_step, fails=fails, attr_step=None,
                    Q=None, t_star=None)
    return dict(C=int(c_step is not None), c_step=c_step, fails=fails, attr_step=attr_step,
                Q=int(q), t_star=t_star)


