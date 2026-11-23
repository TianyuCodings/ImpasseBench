"""Qwen3-Coder-30B-A3B on Amazon Bedrock (Converse), as used in the monitor study.

Sampling: temperature 0.7, top-p 0.8, top-k 20, 4,096 output tokens; Converse
exposes no seed. Credentials come from the standard ``AWS_BEARER_TOKEN_BEDROCK``
environment variable (or the usual boto3 credential chain).

Qwen sometimes returns its XML tool-call syntax as plain text instead of a
``toolUse`` block. Complete ``<function=...><parameter=...>`` blocks are
decoded into a tool call; a malformed block is rejected as a whole and
becomes a protocol error. Missing arguments are never inferred.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid

from ..harness.agents import BedrockAgent, _LLMAgent
from .strict_calls import parse_response, protocol_error

MODEL = "qwen.qwen3-coder-30b-a3b-v1:0"
REGION = "us-west-2"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 4096, "seed": None}
REQUEST_TIMEOUT_S = 900
WATCHDOG_TIMEOUT_S = REQUEST_TIMEOUT_S + 60
_RETURN_ONE = ". Return one complete tool call with all required arguments."


def decode_xml(text, tools):
    """Return complete calls found in ``text`` and the text before them."""
    if '<function=' not in text:
        return [], text
    functions = list(re.finditer(r'<function=([^>]+)>(.*?)</function>', text, re.S))
    if len(functions) != text.count('<function='):
        raise ValueError('Incomplete function block')
    declarations = {d['name']: d['parameters'] for d in tools}
    calls = []
    for f in functions:
        name, body = f.group(1), f.group(2)
        params = list(re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S))
        residue = re.sub(r'<parameter=([^>]+)>(.*?)</parameter>', '', body, flags=re.S)
        if residue.strip() or len(params) != body.count('<parameter='):
            raise ValueError('Incomplete or ambiguous parameter block')
        args = {}
        properties = declarations.get(name, {}).get('properties', {})
        for p in params:
            key, value = p.group(1), p.group(2)
            if key in args:
                raise ValueError('Duplicate parameter')
            # the serialization adds one boundary newline on each side
            if value.startswith('\n'):
                value = value[1:]
            if value.endswith('\n'):
                value = value[:-1]
            if properties.get(key, {}).get('type', 'string') != 'string':
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    pass  # left for schema validation to reject
            args[key] = value
        calls.append({'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
    prefix = text[:min(text.find('<function='), text.find('<tool_call>')
                      if '<tool_call>' in text else len(text))]
    return calls, prefix


def parse_converse(response, tools):
    blocks = response.get('output', {}).get('message', {}).get('content', [])
    text = ''.join(b.get('text', '') for b in blocks)
    native = [b['toolUse'] for b in blocks if 'toolUse' in b]
    calls = [{'id': b['toolUseId'], 'type': 'function',
              'function': {'name': b['name'], 'arguments': json.dumps(b.get('input'), ensure_ascii=False)}}
             for b in native]
    xml_error = None
    xml = '<function=' in text
    if xml:
        try:
            parsed, prefix = decode_xml(text, tools)
            calls.extend(parsed)
        except ValueError as exc:
            xml_error = str(exc)
            prefix = text
    else:
        prefix = text
    stop = response.get('stopReason')
    finish = 'length' if stop == 'max_tokens' else ('tool_calls' if calls else 'stop')
    result = parse_response({'choices': [{'message': {'content': prefix, 'tool_calls': calls},
                                          'finish_reason': finish}]}, tools)
    if xml:
        result['raw_arguments'] = text  # the exact provider text stays in the trace
    if xml_error:
        result['protocol_error'] = protocol_error([xml_error], suffix=_RETURN_ONE)
    return result


class QwenConverseAgent(_LLMAgent):
    def __init__(self, model=MODEL, interface="escalate", region=REGION):
        super().__init__(model, interface, SAMPLING["temperature"])
        import boto3
        from botocore.config import Config
        self._cache = False
        self.client = boto3.session.Session().client(
            'bedrock-runtime', region_name=region,
            config=Config(connect_timeout=10, read_timeout=REQUEST_TIMEOUT_S,
                          retries={'total_max_attempts': 1}, max_pool_connections=4))
        self.endpoint_meta = {
            'backend': 'bedrock_converse', 'region': region, 'served_model': model,
            'protocol': 'converse', 'tool_parser': 'native_or_complete_qwen_xml_strict',
            'temperature': SAMPLING['temperature'], 'top_p': SAMPLING['top_p'],
            'top_k': SAMPLING['top_k'], 'max_tokens': SAMPLING['max_tokens'], 'seed': None,
            'sdk': 'boto3==' + boto3.__version__, 'request_timeout_s': REQUEST_TIMEOUT_S,
            'successful_requests': 0, 'request_wall_s': 0.0, 'xml_responses': 0}
        self.sampling_note = 'temperature=0.7; top_p=0.8; top_k=20; max_tokens=4096; seed unavailable'

    _tools = BedrockAgent._tools
    _convert = BedrockAgent._convert

    def _call_with_watchdog(self, fn):
        box = {}

        def _run():
            try:
                box["resp"] = fn()
            except Exception as e:  # noqa: BLE001 - re-raised on caller thread
                box["err"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(WATCHDOG_TIMEOUT_S)
        if t.is_alive():
            raise TimeoutError(f"LLM call exceeded {WATCHDOG_TIMEOUT_S}s watchdog (thread abandoned)")
        if "err" in box:
            raise box["err"]
        return box["resp"]

    def _call(self, history):
        start = time.monotonic()
        r = self.client.converse(
            modelId=self.model, system=[{'text': self._system_instruction}],
            messages=self._convert(history), toolConfig=self._tools(),
            inferenceConfig={'temperature': SAMPLING['temperature'], 'topP': SAMPLING['top_p'],
                             'maxTokens': SAMPLING['max_tokens']},
            additionalModelRequestFields={'top_k': SAMPLING['top_k']})
        self.endpoint_meta['successful_requests'] += 1
        self.endpoint_meta['request_wall_s'] += time.monotonic() - start
        return r

    def _parse(self, response):
        self.total_tokens += response.get('usage', {}).get('totalTokens', 0)
        result = parse_converse(response, self._tool_decls())
        if '<function=' in (result.get('raw_arguments') or ''):
            self.endpoint_meta['xml_responses'] += 1
        return result
