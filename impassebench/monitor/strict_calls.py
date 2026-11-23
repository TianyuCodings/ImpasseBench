"""Strict tool-call handling: parse losslessly, validate against the schema,
and never execute or repair a malformed request.

A malformed, truncated or multiple tool call becomes a ``protocol_error``: it
is recorded in the trace, the agent is told it was not executed, and its raw
arguments are shown back as text rather than as a (fake) valid past call.
"""
from __future__ import annotations

import json


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def schema_errors(value, schema, path="arguments"):
    errors = []
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    if kind in types:
        valid = isinstance(value, types[kind])
        if kind in ("boolean", "integer"):
            valid = type(value) is types[kind]
        if kind == "number" and isinstance(value, bool):
            valid = False
        if not valid:
            return [f"{path} must be {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                errors.extend(schema_errors(child, properties[key], path + "." + key))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key} is not allowed")
    if isinstance(value, list):
        for i, item in enumerate(value):
            errors.extend(schema_errors(item, schema.get("items", {}), f"{path}[{i}]"))
    return errors


def validate_call(name, args, tools):
    declarations = [x.get("function", x) for x in tools]
    found = [x for x in declarations if x["name"] == name]
    if not found:
        return [f"unknown or unavailable tool: {name}"]
    return schema_errors(args, found[0]["parameters"])


def protocol_error(errors, *, suffix=""):
    return {"kind": "tool_protocol_error", "recoverable": True,
            "environment_action_executed": False, "infrastructure_failure": False,
            "errors": errors,
            "feedback": "Tool request was not executed. " + "; ".join(errors) + suffix}


_RETURN_ONE = ". Return one complete tool call with all required arguments."


def parse_response(response, tools):
    """Parse one OpenAI-style completion into an action dict (first call only)."""
    choices = _get(response, "choices", [])
    if len(choices) != 1:
        raise ValueError("exactly one completion per request is required")
    choice = choices[0]
    message = _get(choice, "message", {})
    calls = _get(message, "tool_calls", []) or []
    finish = _get(choice, "finish_reason")
    action = {"tool": None, "args": None, "text": _get(message, "content") or "",
              "call_id": "", "raw_arguments": None, "finish_reason": finish,
              "protocol_error": None, "tool_call_count": len(calls)}
    errors = []
    if calls:
        call = calls[0]
        fn = _get(call, "function", {})
        raw = _get(fn, "arguments")
        action.update(tool=_get(fn, "name"), call_id=_get(call, "id") or "",
                      raw_arguments=raw)
        if len(calls) != 1:
            errors.append("Only one tool call is supported per turn; no calls were executed")
        try:
            if isinstance(raw, dict):
                args = raw
            elif isinstance(raw, str):
                args = json.loads(raw)
            else:
                raise ValueError("arguments must be a JSON object")
            action["args"] = args
            errors.extend(validate_call(action["tool"], args, tools))
        except (ValueError, TypeError) as exc:
            errors.append("Invalid JSON arguments: " + str(exc))
    if finish == "length":
        errors.append("Output reached the token limit. Split long edits into smaller complete calls")
    if errors:
        action["protocol_error"] = protocol_error(errors, suffix=_RETURN_ONE)
    return action


def error_history(action):
    """History entries for an unexecuted malformed request.

    The request is shown as assistant text plus the error feedback, never as a
    past tool call with invented arguments.
    """
    assert action.get("protocol_error")
    raw = action.get("raw_arguments")
    raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    attempted = (action.get("text") or "") + "\nUnexecuted tool request: " + str(action.get("tool"))
    attempted += "\nRaw arguments: " + raw
    return [{"role": "assistant", "text": attempted, "tool": None, "args": None},
            {"role": "user", "text": action["protocol_error"]["feedback"]}]
