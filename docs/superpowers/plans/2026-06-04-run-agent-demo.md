# `run_agent.bat` Recall-Demo Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repo-local `run_agent.bat` that loads the most recent top Hugging Face reservoir-agent model, prints a guided cross-pass recall demo on the downloaded weights, then drops into a genuinely-stateful REPL.

**Architecture:** Reuse the existing `reservoir.installer` pipeline (registry → download → load → console). Factor the cross-pass eval out of `crosspass.run_cross_pass_kv` into a reusable `eval_recall`; add a `recall_demo_session` printer and a `generate_stateful` greedy decoder (which fixes the existing REPL's dangling write-path); wire `menu.main` to default to no-menu demo+REPL; add the `.bat`.

**Tech Stack:** Python 3.13, pytest, numpy (pure tests), torch/peft/huggingface_hub (runtime-gated), Windows batch.

---

## File Structure

- `src/reservoir/crosspass.py` (modify) — extract `_kv_forward_pair`, add `eval_recall` + `recall_accuracy`.
- `src/reservoir/installer/console.py` (modify) — add `recall_demo_session`, `generate_stateful`; rewrite `ReservoirConsole.step`; new `run(repo_id, *, demo, repl)`.
- `src/reservoir/installer/menu.py` (modify) — flags `--menu`/`--demo-only`/`--no-demo`, default auto-pick, utf-8 stdout.
- `run_agent.bat` (create) — repo-root launcher (local debug tool, not distributable).
- `tests/test_crosspass.py` (modify) — `eval_recall`/`recall_accuracy` tests with a fake lm.
- `tests/test_console.py` (modify) — `recall_demo_session` printer test; `generate_stateful` torch-gated test.
- `tests/test_installer_menu.py` (modify) — `main` flag-routing tests.

---

## Task 1: `eval_recall` + `recall_accuracy` in crosspass.py

**Files:**
- Modify: `src/reservoir/crosspass.py`
- Test: `tests/test_crosspass.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crosspass.py`:

```python
import numpy as np
from reservoir.crosspass import eval_recall, recall_accuracy


class _FakeEnc(dict):
    def to(self, device):
        return self


class _FakeTok:
    eos_token_id = 999

    def __call__(self, text, return_tensors=None):
        return _FakeEnc(input_ids=text, attention_mask=None)


class _FakeReservoirLM:
    """Simulates a stateful kv reservoir: remembers the word from pass 1 and predicts it
    in pass 2 — unless the state was wiped between the passes (the baseline)."""

    def __init__(self, vocab):
        self.tokenizer = _FakeTok()
        self.device = "cpu"
        self.vocab = vocab            # stripped word -> tok_id
        self._mem = None

    def reset_state(self):
        self._mem = None

    def forward_logits(self, input_ids, attention_mask):
        text = input_ids
        prefix = "The secret word is "
        if text.startswith(prefix):
            self._mem = text[len(prefix):].rstrip(".")
        V = max(self.vocab.values()) + 2
        logits = np.full((1, 1, V), -1.0)
        idx = self.vocab.get(self._mem, V - 1)   # no memory -> sentinel id (a miss)
        logits[0, -1, idx] = 1.0
        return logits


def test_eval_recall_stateful_hits_baseline_misses():
    vocab = {"red": 1, "blue": 2, "green": 3}
    keys = [("red", 1), ("blue", 2), ("green", 3)]
    recs = eval_recall(_FakeReservoirLM(vocab), keys)
    assert all(r["stateful_ok"] for r in recs)
    assert not any(r["baseline_ok"] for r in recs)
    assert [r["word"] for r in recs] == ["red", "blue", "green"]
    assert recall_accuracy(recs, "stateful") == 1.0
    assert recall_accuracy(recs, "baseline") == 0.0


def test_recall_accuracy_empty_is_zero():
    assert recall_accuracy([], "stateful") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crosspass.py::test_eval_recall_stateful_hits_baseline_misses -v`
Expected: FAIL with `ImportError: cannot import name 'eval_recall'`.

- [ ] **Step 3: Write minimal implementation**

In `src/reservoir/crosspass.py`, add these module-level functions after `_single_token_keys` (and before `run_cross_pass`):

```python
def _kv_forward_pair(lm, key_word, *, reset_between):
    """One cross-pass recall trial on a kv-prefix reservoir model: encode the key in
    pass 1, optionally wipe the reservoir state, then read the recall logits in pass 2.

    Returns the next-token logits at the recall point (``logits[0, -1]``)."""
    tok = lm.tokenizer
    p1 = tok(f"The secret word is {key_word}.", return_tensors="pt").to(lm.device)
    p2 = tok("The secret word was", return_tensors="pt").to(lm.device)
    lm.reset_state()
    lm.forward_logits(p1["input_ids"], p1["attention_mask"])     # pass 1: read the key
    if reset_between:
        lm.reset_state()                                         # baseline: wipe carried state
    logits = lm.forward_logits(p2["input_ids"], p2["attention_mask"])
    return logits[0, -1]


def eval_recall(lm, keys):
    """Evaluate cross-pass recall on an already-loaded kv-prefix model (no training).

    For each ``(word, tok_id)`` in ``keys``, run the recall task twice: ``stateful`` (the
    reservoir state carries across the two passes) and ``baseline`` (state wiped between
    them). Returns one record per key with both predictions and whether each recalled."""
    records = []
    for word, tok_id in keys:
        s = int(_kv_forward_pair(lm, word, reset_between=False).argmax().item())
        b = int(_kv_forward_pair(lm, word, reset_between=True).argmax().item())
        records.append({"word": word, "tok_id": tok_id,
                        "stateful_pred": s, "stateful_ok": s == tok_id,
                        "baseline_pred": b, "baseline_ok": b == tok_id})
    return records


def recall_accuracy(records, which):
    """Fraction of ``records`` where ``which`` ('stateful' or 'baseline') recalled the key."""
    if not records:
        return 0.0
    return sum(r[f"{which}_ok"] for r in records) / len(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_crosspass.py -v -k "eval_recall or recall_accuracy"`
Expected: both tests PASS.

- [ ] **Step 5: Refactor `run_cross_pass_kv` to use `_kv_forward_pair` (DRY)**

In `src/reservoir/crosspass.py`, inside `run_cross_pass_kv`, delete the nested `forward_pair` definition:

```python
    def forward_pair(key_word, *, reset_between):
        p1 = tok(f"The secret word is {key_word}.", return_tensors="pt").to(lm.device)
        p2 = tok("The secret word was", return_tensors="pt").to(lm.device)
        lm.reset_state()
        lm.forward_logits(p1["input_ids"], p1["attention_mask"])   # pass 1: read key
        if reset_between:
            lm.reset_state()
        logits = lm.forward_logits(p2["input_ids"], p2["attention_mask"])
        return logits[0, -1]
```

Then replace its two call sites — the training-loop line and the eval line — changing
`forward_pair(word, reset_between=not stateful)` to
`_kv_forward_pair(lm, word, reset_between=not stateful)` in both places.

- [ ] **Step 6: Run the full crosspass + persist + train_seed suites to confirm no regression**

Run: `python -m pytest tests/test_crosspass.py tests/test_persist.py tests/test_train_seed.py -v`
Expected: all PASS (the refactor is behaviour-preserving; the kv tests exercise `run_cross_pass_kv`).

- [ ] **Step 7: Commit**

```bash
git add src/reservoir/crosspass.py tests/test_crosspass.py
git commit -m "feat: factor eval_recall out of run_cross_pass_kv (reusable cross-pass eval)"
```

---

## Task 2: `recall_demo_session` printer in console.py

**Files:**
- Modify: `src/reservoir/installer/console.py`
- Test: `tests/test_console.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_console.py`:

```python
import numpy as np

from reservoir.installer.console import recall_demo_session


class _FakeEnc(dict):
    def to(self, device):
        return self


class _FakeTok:
    eos_token_id = 999

    def __call__(self, text, return_tensors=None):
        return _FakeEnc(input_ids=text, attention_mask=None)


class _FakeReservoirLM:
    """Stateful kv reservoir stub: recalls pass-1 word in pass 2 unless state is wiped."""

    def __init__(self, vocab):
        self.tokenizer = _FakeTok()
        self.device = "cpu"
        self.vocab = vocab
        self._mem = None

    def reset_state(self):
        self._mem = None

    def forward_logits(self, input_ids, attention_mask):
        prefix = "The secret word is "
        if input_ids.startswith(prefix):
            self._mem = input_ids[len(prefix):].rstrip(".")
        V = max(self.vocab.values()) + 2
        logits = np.full((1, 1, V), -1.0)
        logits[0, -1, self.vocab.get(self._mem, V - 1)] = 1.0
        return logits


def test_recall_demo_session_prints_table_and_summary():
    keys = [("red", 1), ("blue", 2)]
    lines = []
    recs = recall_demo_session(_FakeReservoirLM({"red": 1, "blue": 2}), keys,
                               print_fn=lines.append)
    text = "\n".join(lines)
    assert "red" in text and "blue" in text
    assert "recall accuracy" in text
    assert "✓" in text          # at least one stateful hit marker
    assert "✗" in text          # at least one baseline miss marker
    assert len(recs) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_console.py::test_recall_demo_session_prints_table_and_summary -v`
Expected: FAIL with `ImportError: cannot import name 'recall_demo_session'`.

- [ ] **Step 3: Write minimal implementation**

In `src/reservoir/installer/console.py`, add this function (after `download_and_resolve`, before `class ReservoirConsole`):

```python
def recall_demo_session(lm, keys, *, print_fn=print):
    """Print the guided cross-pass recall demonstration for a loaded reservoir agent.

    Runs the (training-free) recall eval and prints, per key, the stateful prediction
    (reservoir carried across passes) vs the baseline (state wiped) with a hit/miss mark,
    then a recall-accuracy summary. Returns the eval records."""
    from reservoir.crosspass import eval_recall, recall_accuracy

    records = eval_recall(lm, keys)
    id2word = {tok_id: word for word, tok_id in keys}
    print_fn("")
    print_fn("Cross-pass recall demo — does the reservoir carry a wiped secret word "
             "across forward passes?")
    print_fn("  pass 1: 'The secret word is X.'  ->  context wiped  ->  "
             "pass 2: 'The secret word was ___'")
    print_fn("")
    for r in records:
        sp = id2word.get(r["stateful_pred"], f"<{r['stateful_pred']}>")
        bp = id2word.get(r["baseline_pred"], f"<{r['baseline_pred']}>")
        sm = "✓" if r["stateful_ok"] else "✗"
        bm = "✓" if r["baseline_ok"] else "✗"
        print_fn(f"  secret word -> {r['word']:<8}  reservoir (stateful): {sp:<8} {sm}"
                 f"   stateless baseline: {bp:<8} {bm}")
    n = len(records)
    print_fn("")
    print_fn(f"  recall accuracy   stateful: {recall_accuracy(records, 'stateful'):.0%}"
             f"   baseline: {recall_accuracy(records, 'baseline'):.0%}"
             f"   chance: {1.0 / max(n, 1):.0%}")
    print_fn("")
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_console.py::test_recall_demo_session_prints_table_and_summary -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reservoir/installer/console.py tests/test_console.py
git commit -m "feat: recall_demo_session — guided cross-pass recall printout"
```

---

## Task 3: `generate_stateful` + stateful REPL step

**Files:**
- Modify: `src/reservoir/installer/console.py`
- Test: `tests/test_console.py`

This fixes the existing bug: `ReservoirConsole.step` used `lm.model.generate`, which fires
the read hook but never applies the trained reservoir→prefix write-path (that lives only
in `forward_logits`). Greedy-decoding over `forward_logits` makes the REPL genuinely stateful.

- [ ] **Step 1: Write the failing test (torch-gated)**

Append to `tests/test_console.py`:

```python
def test_generate_stateful_greedy_decodes_and_stops_at_eos():
    torch = pytest.importorskip("torch")
    from reservoir.installer.console import generate_stateful

    V = 5

    class _StubTok:
        eos_token_id = 4

    class _StubLM:
        def __init__(self):
            self.torch = torch
            self.device = "cpu"
            self.tokenizer = _StubTok()
            self.calls = 0

        def forward_logits(self, ids, attention_mask):
            scripted = [2, 3, 4]                      # 4 == eos -> stop
            nxt = scripted[min(self.calls, len(scripted) - 1)]
            self.calls += 1
            logits = torch.full((1, ids.shape[1], V), -1.0)
            logits[0, -1, nxt] = 1.0
            return logits

    ids = torch.tensor([[0, 1]])
    mask = torch.ones((1, 2), dtype=torch.long)
    out = generate_stateful(_StubLM(), ids, mask, max_new_tokens=10)
    assert out == [2, 3, 4]
```

(`pytest` and `import pytest` are already in `tests/test_console.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_console.py::test_generate_stateful_greedy_decodes_and_stops_at_eos -v`
Expected: FAIL with `ImportError: cannot import name 'generate_stateful'` (or SKIP if torch absent — then run on a machine with torch).

- [ ] **Step 3: Write minimal implementation**

In `src/reservoir/installer/console.py`, add `generate_stateful` (after `recall_demo_session`):

```python
def generate_stateful(lm, input_ids, attention_mask, max_new_tokens):
    """Greedy-decode from a kv-prefix reservoir model using ``forward_logits`` (which
    applies the trained reservoir->prefix write-path and ticks the state every pass).
    Returns the list of generated token ids; stops early at eos."""
    torch = lm.torch
    ids, mask = input_ids, attention_mask
    out = []
    for _ in range(max_new_tokens):
        logits = lm.forward_logits(ids, mask)
        next_id = int(logits[0, -1].argmax().item())
        out.append(next_id)
        if next_id == lm.tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=lm.device)], dim=1)
        mask = torch.cat(
            [mask, torch.ones((1, 1), dtype=mask.dtype, device=lm.device)], dim=1)
    return out
```

- [ ] **Step 4: Rewrite `ReservoirConsole.step` to use it**

In `src/reservoir/installer/console.py`, replace the body of `ReservoirConsole.step`:

```python
    def step(self, text: str) -> str:
        """One turn: greedy-decode the model's continuation via the reservoir write-path.
        Reservoir state is NOT reset — it carries into the next call."""
        tok = self.lm.tokenizer
        enc = tok(text, return_tensors="pt").to(self.lm.device)
        ids = generate_stateful(self.lm, enc["input_ids"], enc["attention_mask"],
                                self.max_new_tokens)
        return tok.decode(ids, skip_special_tokens=True).strip()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_console.py -v`
Expected: PASS (or the torch-gated test SKIPs where torch is absent; the pure tests pass).

- [ ] **Step 6: Commit**

```bash
git add src/reservoir/installer/console.py tests/test_console.py
git commit -m "fix: REPL generates via forward_logits write-path (genuinely stateful)"
```

---

## Task 4: `console.run(repo_id, *, demo, repl)`

**Files:**
- Modify: `src/reservoir/installer/console.py`

No pure unit test (this path downloads + loads a real model; it is runtime-gated and
verified by the manual smoke run in Task 6). Keep the change small and exact.

- [ ] **Step 1: Replace `console.run`**

In `src/reservoir/installer/console.py`, replace the existing `run` function:

```python
def run(repo_id, *, demo=True, repl=True, max_new_tokens=40):
    """Download + load the model, optionally run the recall demo, then optionally the
    stateful REPL. Network + torch gated."""
    from reservoir.persist import load_reservoir_model, load_model_config
    from reservoir.crosspass import _single_token_keys

    load_dir = download_and_resolve(repo_id)
    lm = load_reservoir_model(load_dir)
    if demo:
        meta = load_model_config(load_dir).get("meta") or {}
        n_keys = int(meta.get("n_keys", 6))
        keys = _single_token_keys(lm.tokenizer, n_keys)
        recall_demo_session(lm, keys)
    if repl:
        ReservoirConsole(lm, max_new_tokens=max_new_tokens).repl()
```

- [ ] **Step 2: Sanity-check the module imports cleanly**

Run: `python -c "import reservoir.installer.console as c; print(c.run, c.recall_demo_session, c.generate_stateful)"`
Expected: prints three function reprs, no error.

- [ ] **Step 3: Commit**

```bash
git add src/reservoir/installer/console.py
git commit -m "feat: console.run grows demo/repl switches + n_keys from model meta"
```

---

## Task 5: `menu.main` flag routing + utf-8 stdout

**Files:**
- Modify: `src/reservoir/installer/menu.py`
- Test: `tests/test_installer_menu.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_installer_menu.py`:

```python
import reservoir.installer.console as console
from reservoir.installer.menu import main

DEFAULT_REPO = "EmmaLeonhart/reservoir-agent-gpt2-crosspass"


def _spy(monkeypatch):
    calls = {}
    def fake_run(repo_id, **kw):
        calls["repo_id"] = repo_id
        calls.update(kw)
    monkeypatch.setattr(console, "run", fake_run)
    return calls


def test_default_flow_auto_picks_and_runs_demo_and_repl(monkeypatch):
    calls = _spy(monkeypatch)
    assert main(["--no-hf"]) == 0
    assert calls["repo_id"] == DEFAULT_REPO
    assert calls["demo"] is True and calls["repl"] is True


def test_demo_only_disables_repl(monkeypatch):
    calls = _spy(monkeypatch)
    main(["--no-hf", "--demo-only"])
    assert calls["demo"] is True and calls["repl"] is False


def test_no_demo_disables_demo(monkeypatch):
    calls = _spy(monkeypatch)
    main(["--no-hf", "--no-demo"])
    assert calls["demo"] is False and calls["repl"] is True


def test_menu_flag_uses_chooser(monkeypatch):
    calls = _spy(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")   # empty -> default
    main(["--no-hf", "--menu"])
    assert calls["repo_id"] == DEFAULT_REPO


def test_repo_id_skips_selection(monkeypatch):
    calls = _spy(monkeypatch)
    main(["--no-hf", "--repo-id", "Foo/bar"])
    assert calls["repo_id"] == "Foo/bar"


def test_list_prints_and_does_not_run(monkeypatch):
    calls = _spy(monkeypatch)
    assert main(["--no-hf", "--list"]) == 0
    assert "repo_id" not in calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_installer_menu.py -v -k "default_flow or demo_only or no_demo or menu_flag or repo_id_skips or list_prints"`
Expected: FAIL (the new flags don't exist / default flow still prompts via `input`).

- [ ] **Step 3: Replace `menu.main`**

In `src/reservoir/installer/menu.py`, replace the `main` function:

```python
def main(argv=None):
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")   # the demo prints ✓/✗ marks
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Install and run a reservoir-agent model.")
    ap.add_argument("--repo-id", default=None, help="skip selection; load this repo directly")
    ap.add_argument("--no-hf", action="store_true", help="use only the bundled model list")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    ap.add_argument("--menu", action="store_true",
                    help="show the model chooser instead of auto-picking the default")
    ap.add_argument("--demo-only", action="store_true",
                    help="run the recall demo and exit (no REPL)")
    ap.add_argument("--no-demo", action="store_true",
                    help="skip the recall demo, go straight to the REPL")
    args = ap.parse_args(argv)

    from . import console

    models = list_models(use_hf=not args.no_hf)
    if args.list:
        print(_format_menu(models))
        return 0

    repo_id = args.repo_id
    if not repo_id:
        if args.menu:
            print(_format_menu(models))
            repo_id = choose_repo(models, input("model> "))
        else:
            d = default_model(models)
            if d is None:
                print("no reservoir-agent models available to run", file=sys.stderr)
                return 1
            repo_id = d["repo_id"]

    print(f"loading {repo_id} …")
    console.run(repo_id, demo=not args.no_demo, repl=not args.demo_only)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_installer_menu.py -v`
Expected: all PASS (old `choose_repo` tests stay green; six new routing tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/reservoir/installer/menu.py tests/test_installer_menu.py
git commit -m "feat: installer menu defaults to auto-pick + demo+REPL; --menu/--demo-only/--no-demo"
```

---

## Task 6: `run_agent.bat` + manual smoke verification

**Files:**
- Create: `run_agent.bat`

- [ ] **Step 1: Create the launcher**

Create `run_agent.bat` at the repo root:

```bat
@echo off
REM run_agent.bat — local debug/demo launcher (NOT the distributable; that's the .exe).
REM Loads the most recent top reservoir-agent model, runs the recall demo, then a REPL.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

set "PY=python"
if exist "C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe" set "PY=C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe"
if defined RESERVOIR_PYTHON set "PY=%RESERVOIR_PYTHON%"

echo Using Python: %PY%
"%PY%" -m reservoir.installer %*
endlocal
```

- [ ] **Step 2: Verify the entry point resolves without running a full session**

Run: `python -m reservoir.installer --no-hf --list`
Expected: prints the bundled model menu (the `reservoir-agent-gpt2-crosspass` row) and exits 0.

- [ ] **Step 3: Full manual smoke run (downloads the model; needs torch + network)**

Run: `run_agent.bat`
Expected: prints `Using Python: ...`, `loading EmmaLeonhart/reservoir-agent-gpt2-crosspass …`, downloads, then prints the cross-pass recall demo table (stateful column should mostly show ✓, baseline mostly ✗, with a recall-accuracy summary), then shows the `you> ` REPL prompt. Type a line, confirm an `agent> ` reply, press Enter on an empty line to exit.

If the demo's stateful accuracy is not clearly above baseline/chance, STOP and report — that means the loaded artifact or the eval prompts don't match the trained distribution (do not paper over it).

- [ ] **Step 4: Run the full pure test suite**

Run: `python -m pytest -q`
Expected: green (torch/network-gated tests SKIP where deps are absent).

- [ ] **Step 5: Commit + devlog entry**

Add a dated entry to `devlog.md` summarizing: `run_agent.bat` local launcher; `eval_recall` factored out of `run_cross_pass_kv`; `recall_demo_session`; the REPL write-path fix (`generate_stateful`); menu default = auto-pick + demo + REPL. Note the result of the manual smoke run (stateful vs baseline recall accuracy observed).

```bash
git add run_agent.bat devlog.md
git commit -m "feat: run_agent.bat — local recall-demo + stateful REPL launcher"
```

- [ ] **Step 6: Push**

```bash
git push
```

---

## Self-Review

**Spec coverage:**
- `run_agent.bat` local launcher + Python resolution → Task 6 ✓
- Most-recent top model via `default_model()` → Task 5 (unchanged registry) ✓
- `eval_recall` factored from `run_cross_pass_kv` → Task 1 ✓
- Guided demo (✓/✗ table + summary) → Task 2 ✓
- REPL write-path fix (`generate_stateful`) → Task 3 ✓
- `console.run` demo/repl switches + n_keys from meta → Task 4 ✓
- Menu flags (`--menu`/`--demo-only`/`--no-demo`, default auto-pick) → Task 5 ✓
- Tests for pure parts; torch/network gated → Tasks 1,2,3,5 ✓
- Error handling: empty registry → Task 5 (`d is None` → exit 1); missing `n_keys` → Task 4 (default 6); no Python → Task 6 (`.bat` falls back to `python`) ✓

**Placeholder scan:** none — every code step shows full code.

**Type/name consistency:** `eval_recall(lm, keys)` / `recall_accuracy(records, which)` / `recall_demo_session(lm, keys, *, print_fn)` / `generate_stateful(lm, input_ids, attention_mask, max_new_tokens)` / `console.run(repo_id, *, demo, repl, max_new_tokens)` are used consistently across tasks and tests. Record keys (`stateful_pred`/`stateful_ok`/`baseline_pred`/`baseline_ok`/`word`/`tok_id`) match between Task 1 producer and Task 2 consumer.
