"""E: a Hermes-format agentic harness around the always-alive runtime.

Hermes-3 (Llama-3.2-3B) is already fine-tuned for ChatML + function calling, so the
serious training/eval should run in *its* format, not the minimal loop's. This module
provides the Hermes-format layer — ChatML rendering, the function-calling system prompt,
and `<tool_call>` / `<tool_response>` parsing/formatting — plus a `HermesHarness` that
drives the reservoir-injected Hermes through the always-alive loop (prompted + unprompted
passes, the gate) while preserving Hermes' tool-call behaviour.

The pure formatting logic is unit-tested in CI. The `HermesHarness` model integration is
torch-gated / local. **Named plainly as NOT a full fork of the Nous harness:** real
streaming, the full system-prompt scaffolding, multi-tool routing, and the production
agentic loop are larger than this session — this is the Hermes-format core the rest plugs
into, with the reservoir's always-alive loop wired in.
"""
from __future__ import annotations

import json
import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render_chatml(turns: list[dict], *, add_generation_prompt: bool = True) -> str:
    """Render ``[{role, content}, ...]`` into Hermes ChatML."""
    parts = [f"{IM_START}{t['role']}\n{t['content']}{IM_END}" for t in turns]
    s = "\n".join(parts)
    if add_generation_prompt:
        s += f"\n{IM_START}assistant\n"
    return s


def tools_system_prompt(tools: list[dict]) -> str:
    """The Hermes function-calling system message advertising the available tools."""
    listed = "\n".join(json.dumps(t) for t in tools)
    return (
        "You are a function calling AI model. You are provided with function signatures "
        "within <tools></tools> XML tags. You may call one or more functions to assist "
        "with the user query. For each function call return a json object with the "
        "function name and arguments within <tool_call></tool_call> XML tags:\n"
        f"<tools>\n{listed}\n</tools>\n"
        "For each function call return:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>'
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract the tool calls from an assistant turn. Malformed JSON blocks are skipped."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if "name" in obj:
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
    return calls


def format_tool_response(name: str, content) -> str:
    """Format a tool result as the Hermes ``<tool_response>`` payload (for a 'tool' turn)."""
    return ("<tool_response>\n"
            + json.dumps({"name": name, "content": content})
            + "\n</tool_response>")


def has_tool_call(text: str) -> bool:
    return bool(_TOOL_CALL_RE.search(text))


class HermesHarness:
    """Drive a reservoir-injected Hermes through a ChatML tool-calling agentic loop.

    The reservoir is injected via :class:`ReservoirInjectedLM` (so its state persists
    across turns — the always-alive substrate); this harness adds the Hermes-format
    context buffer, the function-calling system prompt, and the parse→execute→respond
    tool loop. Requires the ``models`` extra.
    """

    def __init__(self, model_name: str = "gpt2", *, tools: dict | None = None,
                 system: str = "You are a helpful assistant.", max_tool_iters: int = 4,
                 max_new_tokens: int = 64, load_in_4bit: bool = False,
                 device: str | None = None, **reservoir_kwargs):
        from .inject import ReservoirInjectedLM

        self.lm = ReservoirInjectedLM(model_name, load_in_4bit=load_in_4bit,
                                      device=device, **reservoir_kwargs)
        self.tools = dict(tools or {})              # name -> callable(**arguments)
        self.max_tool_iters = max_tool_iters
        self.max_new_tokens = max_new_tokens
        sys_content = system
        if self.tools:
            schemas = [{"name": n, "description": (fn.__doc__ or "").strip()}
                       for n, fn in self.tools.items()]
            sys_content = system + "\n\n" + tools_system_prompt(schemas)
        self.turns = [{"role": "system", "content": sys_content}]

    def _generate(self) -> str:
        torch = self.lm.torch
        prompt = render_chatml(self.turns, add_generation_prompt=True)
        ids = self.lm.tokenizer(prompt, return_tensors="pt").to(self.lm.device)
        with torch.no_grad():
            out = self.lm.model.generate(
                **ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.lm.tokenizer.eos_token_id)
        new = out[0][ids["input_ids"].shape[1]:]
        return self.lm.tokenizer.decode(new, skip_special_tokens=True).strip()

    def chat(self, user_text: str) -> str:
        """One user turn → an assistant reply, running the tool loop in between."""
        self.turns.append({"role": "user", "content": user_text})
        for _ in range(self.max_tool_iters):
            reply = self._generate()
            self.turns.append({"role": "assistant", "content": reply})
            calls = parse_tool_calls(reply)
            if not calls:
                return reply
            for call in calls:                       # execute each tool, feed results back
                fn = self.tools.get(call["name"])
                result = fn(**call["arguments"]) if fn else f"error: unknown tool {call['name']}"
                self.turns.append({"role": "tool",
                                   "content": format_tool_response(call["name"], result)})
        return self._generate()                      # final turn after tool budget


# What a FULL Nous Hermes harness fork additionally needs (NOT done this session):
HERMES_HARNESS_REMAINING = """\
- Streaming generation + the exact Nous system-prompt scaffolding and stop strings.
- The unprompted/idle pass + trained silence gate wired into the same loop (the always-
  alive behaviour; the gate from src/reservoir/silence.py and the runtime in
  src/reservoir/runtime.py are the pieces to fuse here).
- A regression suite vs vanilla Hermes (tool-call formatting unchanged with the reservoir
  injected, i.e. H1 at the generation level) — needs a Hermes GPU run.
- Multi-tool routing, argument validation, and error recovery.
This module is the Hermes-format core + the tool loop; the above is the production fork.
"""

