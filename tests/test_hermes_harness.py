"""Tests for the Hermes-format harness logic (pure; runs in CI)."""
from reservoir.hermes_harness import (
    render_chatml,
    tools_system_prompt,
    parse_tool_calls,
    format_tool_response,
    has_tool_call,
)


def test_render_chatml_roundtrip():
    turns = [{"role": "system", "content": "be helpful"},
             {"role": "user", "content": "hi"}]
    s = render_chatml(turns)
    assert "<|im_start|>system\nbe helpful<|im_end|>" in s
    assert "<|im_start|>user\nhi<|im_end|>" in s
    assert s.rstrip().endswith("<|im_start|>assistant")     # generation prompt added
    assert render_chatml(turns, add_generation_prompt=False).count("assistant") == 0


def test_tools_system_prompt_lists_tools():
    tools = [{"name": "add", "description": "add two numbers"}]
    p = tools_system_prompt(tools)
    assert "<tools>" in p and "</tools>" in p
    assert '"name": "add"' in p
    assert "<tool_call>" in p


def test_parse_tool_calls():
    text = ('Sure.\n<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>')
    calls = parse_tool_calls(text)
    assert calls == [{"name": "add", "arguments": {"a": 2, "b": 3}}]
    assert has_tool_call(text)
    # plain text has no tool call; malformed json is skipped, not crashed
    assert parse_tool_calls("just talking") == []
    assert parse_tool_calls("<tool_call>\n{not json}\n</tool_call>") == []


def test_format_tool_response():
    r = format_tool_response("add", 5)
    assert r.startswith("<tool_response>") and r.endswith("</tool_response>")
    assert '"name": "add"' in r and '"content": 5' in r


def test_harness_chat_runs(tmp_path):
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from reservoir.hermes_harness import HermesHarness

    def add(a, b):
        """add two numbers"""
        return a + b

    # tiny model won't emit valid tool calls, but the harness loop must run end-to-end
    # and return a string (wiring smoke test; the semantic tool-calling demo is on Hermes).
    h = HermesHarness("sshleifer/tiny-gpt2", tools={"add": add}, n_reservoir=32,
                      max_new_tokens=4, device="cpu")
    assert h.turns[0]["role"] == "system" and "<tools>" in h.turns[0]["content"]
    out = h.chat("What is 2 + 3?")
    assert isinstance(out, str)
