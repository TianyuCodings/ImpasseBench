"""Strict tool-call parsing and the Qwen XML decoder."""
import json

import pytest

from impassebench.harness import prompts
from impassebench.monitor.qwen_bedrock import decode_xml, parse_converse
from impassebench.monitor.strict_calls import error_history, parse_response, validate_call

TOOLS = prompts.tool_declarations("escalate")


def resp(calls, content="", finish="tool_calls"):
    return {"choices": [{"message": {"content": content, "tool_calls": calls}, "finish_reason": finish}]}


def call(name, args):
    return {"id": "c1", "function": {"name": name, "arguments": args}}


def test_validate_call():
    assert validate_call("bash", {"command": "ls"}, TOOLS) == []
    assert validate_call("bash", {}, TOOLS) == ["arguments.command is required"]
    assert validate_call("task_complete", {"summary": "s", "verified": "yes"}, TOOLS) == ["arguments.verified must be boolean"]
    assert validate_call("rm_rf", {}, TOOLS) == ["unknown or unavailable tool: rm_rf"]


def test_parse_response_errors_are_not_executed():
    ok = parse_response(resp([call("bash", json.dumps({"command": "ls"}))]), TOOLS)
    assert ok["protocol_error"] is None and ok["args"] == {"command": "ls"}
    bad_json = parse_response(resp([call("bash", '{"command": ')]), TOOLS)
    assert bad_json["protocol_error"]["errors"][0].startswith("Invalid JSON arguments")
    two = parse_response(resp([call("bash", '{"command": "a"}'), call("bash", '{"command": "b"}')]), TOOLS)
    assert "Only one tool call is supported" in two["protocol_error"]["feedback"]
    cut = parse_response(resp([], "partial", finish="length"), TOOLS)
    assert cut["protocol_error"]["feedback"].endswith("Return one complete tool call with all required arguments.")


def test_error_history_shows_raw_text_not_a_call():
    a = parse_response(resp([call("bash", '{"command": ')]), TOOLS)
    h = error_history(a)
    assert h[0]["role"] == "assistant" and h[0]["tool"] is None
    assert 'Raw arguments: {"command": ' in h[0]["text"]
    assert h[1] == {"role": "user", "text": a["protocol_error"]["feedback"]}


def test_qwen_xml_decoding():
    text = "Let me look.\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
    calls, prefix = decode_xml(text, TOOLS)
    assert prefix == "Let me look.\n"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}
    with pytest.raises(ValueError):
        decode_xml("<function=bash><parameter=command>ls</function>", TOOLS)


def test_parse_converse_native_and_malformed_xml():
    native = {"output": {"message": {"content": [{"toolUse": {"toolUseId": "t1", "name": "bash", "input": {"command": "pwd"}}}]}},
              "stopReason": "tool_use"}
    a = parse_converse(native, TOOLS)
    assert (a["tool"], a["args"], a["protocol_error"]) == ("bash", {"command": "pwd"}, None)
    broken = {"output": {"message": {"content": [{"text": "<function=bash><parameter=command>ls"}]}}, "stopReason": "end_turn"}
    b = parse_converse(broken, TOOLS)
    assert b["protocol_error"]["errors"] == ["Incomplete function block"]
    assert b["raw_arguments"] == "<function=bash><parameter=command>ls"
