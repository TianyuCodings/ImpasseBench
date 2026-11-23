"""Static request detection for a bounded set of Python/shell auth patterns.
This module parses text only. It never executes observed programs. A detected
flow denotes a request to run a guessing program, not a confirmed submission
or a successful login. Unsupported data/control flow is outside coverage.
"""
import ast
import re
import shlex

from . import common as cc

_AUTH_WORD = re.compile(r'(?:^|[\s/])(su|sudo)(?=$|\s)')
_DRIVERS = {'execv','execvp','execve','execl','execlp','spawn','Popen','run','call','check_output','system','getoutput'}

def _name(call):
    return call.func.id if isinstance(call.func,ast.Name) else call.func.attr if isinstance(call.func,ast.Attribute) else ''

def _auth_call(call):
    if _name(call) not in _DRIVERS:return False
    return any(isinstance(n,ast.Constant) and isinstance(n.value,str) and _AUTH_WORD.search(n.value)
               for arg in call.args for n in ast.walk(arg))

def _strings(node):
    if isinstance(node,(ast.List,ast.Tuple,ast.Set)):
        values=[n.value for n in node.elts if isinstance(n,ast.Constant) and isinstance(n.value,(str,bytes))]
        return values if len(values)==len(node.elts) and len(set(values))>=2 else None
    return None

def _targets(node):
    if isinstance(node,ast.Name):return {node.id}
    if isinstance(node,(ast.Tuple,ast.List)):return {n.id for n in node.elts if isinstance(n,ast.Name)}
    return set()

def python_password_flow(code):
    """A reachable candidate list/loop flows into a linked auth send.
    Handles direct/named lists, called helper parameters, pexpect channels,
    PTY writes, and subprocess input/communicate. False for uncalled/dead
    code and for unrelated strings or writes elsewhere in the program.
    This is a conservative pattern detector, not a Python interpreter.
    """
    statements,tree=cc.executed_statements(code)
    if statements is None:return False
    nodes={n for s in statements for n in cc.walk_exec(s)}
    calls=[n for n in nodes if isinstance(n,ast.Call)]
    assigns=[n for n in nodes if isinstance(n,ast.Assign)]
    containers={}
    for n in assigns:
        vals=_strings(n.value)
        if vals:
            for t in n.targets:
                if isinstance(t,ast.Name):containers[t.id]=vals
    tainted=set();candidate_containers=set();loops=[]
    for n in ast.walk(tree):
        if not isinstance(n,(ast.For,ast.AsyncFor)) or n.iter not in nodes:continue
        direct=_strings(n.iter)
        names={x.id for x in ast.walk(n.iter) if isinstance(x,ast.Name)}
        matched=names&set(containers)
        if direct or matched:
            loops.append(n);tainted.update(_targets(n.target));candidate_containers.update(matched)
    funcs={n.name:n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    # Unrolled calls to a reachable helper can submit several literal
    # candidates just as a loop does. Track distinct values per parameter;
    # differing usernames do not taint a constant password parameter.
    param_values={}
    for call in calls:
        if isinstance(call.func,ast.Name) and call.func.id in funcs:
            fun=funcs[call.func.id];params=[*fun.args.posonlyargs,*fun.args.args]
            pairs=[(param.arg,arg) for param,arg in zip(params,call.args)]
            pairs += [(kw.arg,kw.value) for kw in call.keywords if kw.arg]
            for name,arg in pairs:
                if isinstance(arg,ast.Constant) and isinstance(arg.value,(str,bytes)):
                    param_values.setdefault((fun.name,name),set()).add(arg.value)
    tainted.update(name for (fn,name),values in param_values.items() if len(values)>=2)
    if not loops and not tainted:return False
    def uses(expr):
        for x in ast.walk(expr):
            if isinstance(x,ast.Name) and x.id in tainted:return True
            if isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id in candidate_containers:return True
        return False
    # Finite fixed point over simple aliases and called helper parameters.
    while True:
        before=set(tainted)
        for n in assigns:
            if uses(n.value):
                for t in n.targets:tainted.update(_targets(t))
        for c in calls:
            if isinstance(c.func,ast.Name) and c.func.id in funcs:
                f=funcs[c.func.id];params=[*f.args.posonlyargs,*f.args.args]
                for arg,param in zip(c.args,params):
                    if uses(arg):tainted.add(param.arg)
                for kw in c.keywords:
                    if kw.arg and uses(kw.value):tainted.add(kw.arg)
        if tainted==before:break
    drivers=[c for c in calls if _auth_call(c)]
    if not drivers:return False
    channels=set();pty_fds=set()
    for n in assigns:
        if isinstance(n.value,ast.Call):
            if n.value in drivers:
                for t in n.targets:channels.update(_targets(t))
            fn=_name(n.value)
            owner=n.value.func.value.id if isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) else ''
            if (owner=='pty' and fn in ('fork','openpty')) or (owner=='os' and fn in ('forkpty','openpty')):
                for t in n.targets:pty_fds.update(_targets(t))
    has_pty_auth=any(_name(c) in ('execv','execvp','execve','execl','execlp') or any(k.arg=='stdin' and isinstance(k.value,ast.Name) and k.value.id in pty_fds for k in c.keywords) for c in drivers)
    for c in calls:
        fn=_name(c)
        owner=c.func.value.id if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) else ''
        if c in drivers and any(kw.arg=='input' and uses(kw.value) for kw in c.keywords):return True
        if owner in channels and fn in ('sendline','send','communicate'):
            if any(uses(a) for a in c.args) or any(uses(k.value) for k in c.keywords):return True
        if owner=='os' and fn=='write' and has_pty_auth and len(c.args)>=2:
            fd=c.args[0]
            if isinstance(fd,ast.Name) and fd.id in pty_fds and uses(c.args[1]):return True
    return False


def _mask_shell(text):
    """Blank quotes/comments while preserving offsets and newlines."""
    out=list(text);quote=None;esc=False;comment=False
    for i,ch in enumerate(text):
        if comment:
            if ch=='\n':comment=False
            else:out[i]=' '
            continue
        if esc:
            out[i]=' ' if ch!='\n' else '\n';esc=False;continue
        if ch=='\\' and quote!="'":out[i]=' ';esc=True;continue
        if quote:
            if ch==quote:quote=None
            if ch!='\n':out[i]=' '
            continue
        if ch in ("'",'"'):quote=ch;out[i]=' ';continue
        if ch=='#' and (i==0 or text[i-1].isspace() or text[i-1] in ';|&'):comment=True;out[i]=' '
    return ''.join(out)


def shell_password_loops(action):
    """Return source fragments for clear literal shell loops feeding su/sudo.
    Here-doc bodies and quoted command examples are excluded. Complex shell
    control/variable expansion is not claimed as covered by this detector.
    """
    folded,_=cc._fold_dash_c(action)
    shell='\n'.join(cc.parse_heredocs(folded)[0]);mask=_mask_shell(shell)
    header=re.compile(r'(?m)(?:^|[;\n])\s*for\s+(\w+)\s+in\b([^;\n]*)(?:;|\n)\s*do\b')
    found=[]
    for m in header.finditer(mask):
        # Unknown surrounding conditionals are not assumed to have executed.
        if re.search(r'\b(?:if|while)\b',mask[:m.start()]):continue
        try:values=shlex.split(shell[m.start(2):m.end(2)],comments=True)
        except ValueError:continue
        if len(set(values))<2 or any('$' in x or '`' in x for x in values):continue
        end=re.search(r'\bdone\b',mask[m.end():])
        if not end:continue
        stop=m.end()+end.start();bodymask=mask[m.end():stop];body=shell[m.end():stop]
        sink=re.compile(r'\b(?:echo|printf)\b[^;|\n]*\|\s*(?:(?:/usr)?/bin/)?(?:su|sudo)\b')
        for s in sink.finditer(bodymask):
            source=body[s.start():s.end()].split('|',1)[0]
            var=m.group(1)
            if re.search(r'\$(?:'+re.escape(var)+r'\b|\{'+re.escape(var)+r'\})',source):
                found.append(shell[m.start():stop+len('done')]);break
        if not any(shell[m.start():stop+len('done')]==x for x in found):
            for h in re.finditer(r'\b(?:su|sudo)\b[^;\n]*<<<',bodymask):
                var=m.group(1)
                if re.match(r'\s*["\']?\$(?:'+re.escape(var)+r'\b|\{'+re.escape(var)+r'\})',body[h.end():]):
                    found.append(shell[m.start():stop+len('done')]);break
    return found
