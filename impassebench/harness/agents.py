"""Model backends.

An agent receives the conversation so far and returns the next action::

    {"tool": "bash", "args": {"command": ...}, "text": ..., "call_id": ...}
    {"tool": "task_complete" | "edit_file" | "escalate_to_human", "args": {...}, ...}
    {"tool": None, "text": ...}          # a reply without a tool call

The history passed in is provider-neutral; each backend converts it to its
own wire format on every call, so the loop, prompts, tool schemas and the
recorded trace are identical across providers::

    {"role": "user",      "text": ...}
    {"role": "assistant", "text": ..., "tool": name|None, "args": {...},
                          "call_id": ..., "provider_meta": {...}}
    {"role": "tool",      "name": ..., "response": {...}, "call_id": ...}

``provider_meta`` is opaque data that only the producing backend reads back
(e.g. Gemini thought signatures). When a provider returns several tool calls
in one turn, only the first is executed and answered.

Credentials are read from the environment by name and never recorded.
Routing (``make_agent``):

* ``gemini-*``                         Google GenAI (``GEMINI_API_KEY``; optional
                                       ``GEMINI_API_KEY2..4`` are used round-robin)
* ``gpt-*``, ``o1/o3/o4*``             OpenAI (``OPENAI_API_KEY``)
* ``claude-*``                         Anthropic (``ANTHROPIC_API_KEY``)
* ``us.*``, ``global.*``, ``qwen.*``,
  ``moonshotai.*``, ...                Amazon Bedrock Converse (``AWS_BEARER_TOKEN_BEDROCK``
                                       or the standard AWS credential chain)
* ``openai-compatible/<served-name>``  any OpenAI-compatible server, e.g. vLLM
                                       (``IMPASSEBENCH_OPENAI_BASE_URL``,
                                       optional ``IMPASSEBENCH_OPENAI_API_KEY``)
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import threading
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

from . import prompts

REQUEST_TIMEOUT_MS = 180_000  # per model request
WATCHDOG_TIMEOUT_S = 240      # outer guard, must exceed REQUEST_TIMEOUT_MS
DEFAULT_TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 4096


class ScriptedAgent:
    """Replays a fixed list of actions (tests and plumbing checks).

    With ``live=False`` the rollout loop keeps no conversation history.
    """

    model = "scripted"
    temperature = None
    sampling_note = None

    def __init__(self, script: list[dict], live: bool = False):
        self.script = list(script)
        self.total_tokens = 0
        self.live = live

    def set_system(self, system_instruction: str):
        self.system_instruction = system_instruction

    def next_action(self, history):
        if not self.script:
            return {"tool": "task_complete", "args": {"summary": "scripted run done",
                                                      "verified": False}}
        return self.script.pop(0)


_API_KEY_ENV = ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3",
                "GEMINI_API_KEY4")
_KEYS: list[str] = []
_KEY_LOCK = threading.Lock()
_KEY_TURN = itertools.count()


def _next_gemini_key() -> str:
    global _KEYS
    with _KEY_LOCK:
        if not _KEYS:
            _KEYS = [v for n in _API_KEY_ENV if (v := os.environ.get(n, "").strip())]
        if not _KEYS:
            raise RuntimeError("no Gemini API key set; expected GEMINI_API_KEY")
        return _KEYS[next(_KEY_TURN) % len(_KEYS)]


# Bedrock model availability is regional; ids without an inference-profile
# prefix are routed to the region that serves them.
_BEDROCK_REGION = {
    "qwen.qwen3-coder-480b-a35b-v1:0": "us-west-2",
    "qwen.qwen3-coder-30b-a3b-v1:0":   "us-west-2",
    "moonshotai.kimi-k2.5":            "us-east-1",
}
_BEDROCK_OPEN_PREFIXES = ("qwen.", "moonshot.", "moonshotai.", "deepseek.",
                          "meta.", "mistral.", "ai21.", "cohere.")
OPENAI_COMPATIBLE_PREFIX = "openai-compatible/"


def make_agent(model: str, interface: str, temperature: float = DEFAULT_TEMPERATURE):
    """Route a model name to its backend; unknown names fail before any
    container starts."""
    if model.startswith(OPENAI_COMPATIBLE_PREFIX):
        return OpenAICompatibleAgent(model, interface, temperature)
    if model.startswith(("us.", "global.")) or ".anthropic." in model \
            or ".openai." in model or model.startswith(_BEDROCK_OPEN_PREFIXES):
        return BedrockAgent(model, interface, temperature,
                            region=os.environ.get("IMPASSEBENCH_BEDROCK_REGION")
                            or _BEDROCK_REGION.get(model, "us-east-1"))
    if model.startswith("gemini"):
        return GeminiAgent(model, interface, temperature)
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return OpenAIAgent(model, interface, temperature)
    if model.startswith("claude"):
        return AnthropicAgent(model, interface, temperature)
    raise ValueError(f"cannot infer provider from model name '{model}'")


class _LLMAgent:
    """Retry, watchdog and the neutral-history contract shared by backends.

    Subclasses implement ``_call(history)`` and ``_parse(resp)`` and update
    ``total_tokens``.
    """

    live = True

    def __init__(self, model: str, interface: str, temperature: float):
        self.model = model
        self.interface = interface
        self.temperature = temperature
        self.total_tokens = 0
        # sampling actually used; flipped if a provider rejects temperature
        self.sampling_note = f"temperature={temperature}"
        self._system_instruction = ""

    def set_system(self, system_instruction: str):
        self._system_instruction = system_instruction

    @property
    def system_instruction(self) -> str:
        return self._system_instruction

    def _tool_decls(self) -> list[dict]:
        return prompts.tool_declarations(self.interface)

    def _reject_temperature(self):
        self.sampling_note = f"temperature=provider_default (rejected {self.temperature})"

    def _call_with_watchdog(self, fn):
        """Bound every provider call with a hard thread join, independent of
        the SDK's own transport timeout."""
        box: dict = {}

        def _run():
            try:
                box["resp"] = fn()
            except Exception as e:  # noqa: BLE001 - re-raised on caller thread
                box["err"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(WATCHDOG_TIMEOUT_S)
        if t.is_alive():
            raise TimeoutError(
                f"LLM call exceeded {WATCHDOG_TIMEOUT_S}s watchdog (thread abandoned)")
        if "err" in box:
            raise box["err"]
        return box["resp"]

    def next_action(self, history: list) -> dict:
        last_err = None
        for attempt in range(8):
            try:
                resp = self._call_with_watchdog(lambda: self._call(history))
                break
            except Exception as e:  # rate limits, 5xx, timeouts: back off and retry
                last_err = e
                text = repr(e)
                throttled = ("429" in text or "RESOURCE_EXHAUSTED" in text
                             or "rate_limit" in text or "overloaded" in text)
                m = re.search(r"retryDelay':\s*'(\d+(?:\.\d+)?)s'", text) \
                    or re.search(r"retry.after[':\s]+(\d+(?:\.\d+)?)", text, re.I)
                if m:
                    delay = float(m.group(1))
                else:
                    delay = min(120 if throttled else 60, 2 ** attempt * 5)
                time.sleep(delay * (0.5 + random.random()))  # jittered
        else:
            raise RuntimeError(
                f"{type(self).__name__} API failed after retries: {last_err!r}")
        return self._parse(resp)


class GeminiAgent(_LLMAgent):
    def __init__(self, model: str, interface: str, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__(model, interface, temperature)
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(
            api_key=_next_gemini_key(),
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    def _config(self):
        t = self._types
        decls = [t.FunctionDeclaration(**d) for d in self._tool_decls()]
        return t.GenerateContentConfig(
            system_instruction=self._system_instruction,
            tools=[t.Tool(function_declarations=decls)],
            temperature=self.temperature,
        )

    def _convert(self, history: list) -> list:
        t = self._types
        out = []
        for h in history:
            if h["role"] == "user":
                out.append(t.Content(role="user",
                                     parts=[t.Part.from_text(text=h["text"])]))
            elif h["role"] == "assistant":
                parts = []
                if h.get("text"):
                    parts.append(t.Part.from_text(text=h["text"]))
                if h.get("tool"):
                    p = t.Part.from_function_call(
                        name=h["tool"], args=h.get("args") or {})
                    sig = (h.get("provider_meta") or {}).get("thought_signature")
                    if sig:
                        p.thought_signature = sig  # must be sent back with the call
                    parts.append(p)
                if parts:
                    out.append(t.Content(role="model", parts=parts))
            else:
                out.append(t.Content(role="user", parts=[
                    t.Part.from_function_response(
                        name=h["name"], response=h["response"])]))
        return out

    def _call(self, history: list):
        return self.client.models.generate_content(
            model=self.model, contents=self._convert(history),
            config=self._config())

    def _parse(self, resp) -> dict:
        if resp.usage_metadata:
            self.total_tokens += (resp.usage_metadata.total_token_count or 0)
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            return {"tool": None, "text": "[empty response]"}
        parts = cand.content.parts or []
        text = "".join(getattr(p, "text", "") or "" for p in parts)
        for part in parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                return {"tool": fc.name, "args": dict(fc.args or {}),
                        "text": text, "call_id": "",
                        "provider_meta": {"thought_signature":
                                          getattr(part, "thought_signature", None)}}
        return {"tool": None, "text": text}


def _openai_messages(system: str, history: list) -> list:
    msgs = [{"role": "system", "content": system}]
    for h in history:
        if h["role"] == "user":
            msgs.append({"role": "user", "content": h["text"]})
        elif h["role"] == "assistant":
            m = {"role": "assistant", "content": h.get("text") or None}
            if h.get("tool"):
                m["tool_calls"] = [{
                    "id": h["call_id"], "type": "function",
                    "function": {"name": h["tool"],
                                 "arguments": json.dumps(h.get("args") or {})},
                }]
            msgs.append(m)
        else:
            msgs.append({"role": "tool", "tool_call_id": h["call_id"],
                         "content": json.dumps(h["response"])})
    return msgs


def _openai_compatible_parse(agent: _LLMAgent, resp) -> dict:
    """Like OpenAIAgent._parse, tolerant of servers that return arguments
    as an object or as malformed JSON."""
    if getattr(resp, "usage", None):
        agent.total_tokens += (resp.usage.total_tokens or 0)
    msg = resp.choices[0].message
    text = msg.content or ""
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        raw = tc.function.arguments
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw or "{}")
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return {"tool": tc.function.name, "args": args, "text": text,
                "call_id": tc.id or ""}
    return {"tool": None, "text": text}


class OpenAIAgent(_LLMAgent):
    def __init__(self, model: str, interface: str, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__(model, interface, temperature)
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # SDK retries off: next_action owns the retry policy
        self.client = OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000,
                             max_retries=0)
        self._supports_temperature = True

    def _call(self, history: list):
        kw = dict(model=self.model,
                  messages=_openai_messages(self._system_instruction, history),
                  tools=[{"type": "function", "function": d} for d in self._tool_decls()],
                  parallel_tool_calls=False)
        if self._supports_temperature:
            kw["temperature"] = self.temperature
        try:
            return self.client.chat.completions.create(**kw)
        except Exception as e:
            if self._supports_temperature and "temperature" in repr(e):
                self._supports_temperature = False
                self._reject_temperature()
                kw.pop("temperature", None)
                return self.client.chat.completions.create(**kw)
            raise

    def _parse(self, resp) -> dict:
        if getattr(resp, "usage", None):
            self.total_tokens += (resp.usage.total_tokens or 0)
        msg = resp.choices[0].message
        text = msg.content or ""
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            return {"tool": tc.function.name, "args": args, "text": text,
                    "call_id": tc.id}
        return {"tool": None, "text": text}


def _scrub_url(url: str) -> str:
    """scheme://host[:port]/path, without credentials or query."""
    try:
        p = urlsplit(url)
        host = p.hostname or ""
        if p.port:
            host += f":{p.port}"
        return urlunsplit((p.scheme, host, p.path, "", ""))
    except Exception:  # noqa: BLE001
        return "<unparseable>"


class OpenAICompatibleAgent(_LLMAgent):
    """Any server speaking the OpenAI chat-completions protocol (vLLM, SGLang,
    TGI, ...), selected with ``openai-compatible/<served-model-name>``."""

    def __init__(self, model: str, interface: str, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__(model, interface, temperature)
        from openai import OpenAI
        base_url = os.environ.get("IMPASSEBENCH_OPENAI_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError("IMPASSEBENCH_OPENAI_BASE_URL is not set")
        self.served_model = model[len(OPENAI_COMPATIBLE_PREFIX):]
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("IMPASSEBENCH_OPENAI_API_KEY", "") or "EMPTY",
            timeout=REQUEST_TIMEOUT_MS / 1000, max_retries=0)
        self.endpoint_meta = {"backend": "openai_compatible",
                              "base_url": _scrub_url(base_url),
                              "served_model": self.served_model}

    def _call(self, history: list):
        return self.client.chat.completions.create(
            model=self.served_model,
            messages=_openai_messages(self._system_instruction, history),
            tools=[{"type": "function", "function": d} for d in self._tool_decls()],
            temperature=self.temperature, max_tokens=MAX_OUTPUT_TOKENS)

    def _parse(self, resp) -> dict:
        return _openai_compatible_parse(self, resp)


class AnthropicAgent(_LLMAgent):
    def __init__(self, model: str, interface: str, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__(model, interface, temperature)
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(
            api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000, max_retries=0)

    def _tools(self):
        tools = [{"name": d["name"], "description": d["description"],
                  "input_schema": d["parameters"]} for d in self._tool_decls()]
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def _convert(self, history: list) -> list:
        msgs = []
        for h in history:
            if h["role"] == "user":
                msgs.append({"role": "user",
                             "content": [{"type": "text", "text": h["text"]}]})
            elif h["role"] == "assistant":
                blocks = []
                if h.get("text"):
                    blocks.append({"type": "text", "text": h["text"]})
                if h.get("tool"):
                    blocks.append({"type": "tool_use", "id": h["call_id"],
                                   "name": h["tool"], "input": h.get("args") or {}})
                if blocks:
                    msgs.append({"role": "assistant", "content": blocks})
            else:
                msgs.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": h["call_id"],
                    "content": json.dumps(h["response"])}]})
        if msgs:  # moving cache breakpoint on the latest turn
            msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return msgs

    def _call(self, history: list):
        return self.client.messages.create(
            model=self.model, max_tokens=MAX_OUTPUT_TOKENS,
            temperature=self.temperature,
            system=[{"type": "text", "text": self._system_instruction,
                     "cache_control": {"type": "ephemeral"}}],
            tools=self._tools(),
            messages=self._convert(history))

    def _parse(self, resp) -> dict:
        u = getattr(resp, "usage", None)
        if u:
            self.total_tokens += (
                (u.input_tokens or 0) + (u.output_tokens or 0)
                + (getattr(u, "cache_creation_input_tokens", 0) or 0)
                + (getattr(u, "cache_read_input_tokens", 0) or 0))
        text, tool = "", None
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use" and tool is None:
                tool = block
        if tool is not None:
            return {"tool": tool.name, "args": dict(tool.input or {}),
                    "text": text, "call_id": tool.id}
        return {"tool": None, "text": text}


class BedrockAgent(_LLMAgent):
    """Anthropic, OpenAI and open-weight models through the Bedrock Converse
    API, which uses one tool-calling format for every provider."""

    def __init__(self, model: str, interface: str, temperature: float = DEFAULT_TEMPERATURE,
                 region: str = "us-east-1"):
        super().__init__(model, interface, temperature)
        import boto3
        from botocore.config import Config
        key = os.environ.get("AWS_API_KEY", "").strip()
        if key:
            os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", key)
        self.client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(read_timeout=REQUEST_TIMEOUT_MS / 1000,
                          connect_timeout=10, retries={"max_attempts": 0}))
        # cachePoint blocks are accepted by Claude models only
        self._cache = ".anthropic." in model
        self._supports_temperature = True

    def _tools(self):
        tools = [{"toolSpec": {"name": d["name"],
                               "description": d["description"],
                               "inputSchema": {"json": d["parameters"]}}}
                 for d in self._tool_decls()]
        if self._cache and tools:
            tools.append({"cachePoint": {"type": "default"}})
        return {"tools": tools}

    def _convert(self, history: list) -> list:
        msgs = []

        def add(role, block):
            # Converse needs strict alternation; merge consecutive same-role turns
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"].append(block)
            else:
                msgs.append({"role": role, "content": [block]})

        for h in history:
            if h["role"] == "user":
                add("user", {"text": h["text"]})
            elif h["role"] == "assistant":
                if h.get("text"):
                    add("assistant", {"text": h["text"]})
                if h.get("tool"):
                    add("assistant", {"toolUse": {"toolUseId": h["call_id"],
                                                  "name": h["tool"],
                                                  "input": h.get("args") or {}}})
                if not h.get("text") and not h.get("tool"):
                    add("assistant", {"text": "[no content]"})
            else:
                add("user", {"toolResult": {"toolUseId": h["call_id"],
                                            "content": [{"json": h["response"]}]}})
        if self._cache and msgs:
            msgs[-1]["content"].append({"cachePoint": {"type": "default"}})
        return msgs

    def _call(self, history: list):
        system = [{"text": self._system_instruction}]
        if self._cache:
            system.append({"cachePoint": {"type": "default"}})
        kw = dict(modelId=self.model, system=system,
                  messages=self._convert(history),
                  toolConfig=self._tools(),
                  inferenceConfig={"maxTokens": MAX_OUTPUT_TOKENS})
        if self._supports_temperature:
            kw["inferenceConfig"]["temperature"] = self.temperature
        try:
            return self.client.converse(**kw)
        except Exception as e:
            if self._supports_temperature and "temperature" in repr(e).lower():
                self._supports_temperature = False
                self._reject_temperature()
                kw["inferenceConfig"].pop("temperature", None)
                return self.client.converse(**kw)
            raise

    def _parse(self, resp) -> dict:
        u = resp.get("usage") or {}
        self.total_tokens += u.get("totalTokens", 0) or 0
        blocks = (resp.get("output", {}).get("message", {}) or {}).get("content", [])
        text, tool = "", None
        for b in blocks:
            if "text" in b:
                text += b["text"]
            elif "toolUse" in b and tool is None:
                tool = b["toolUse"]
        if tool is not None:
            return {"tool": tool["name"], "args": dict(tool.get("input") or {}),
                    "text": text, "call_id": tool["toolUseId"]}
        return {"tool": None, "text": text}


def new_call_id() -> str:
    """Call id for providers whose tool calls carry none (Gemini)."""
    return "call_" + uuid.uuid4().hex[:12]
