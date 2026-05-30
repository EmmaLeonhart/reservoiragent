"""H1 model-surgery regression tests.

These require the optional `models` extra (torch + transformers) and a small model
download, so they skip where those are unavailable (e.g. the light CI job) and run
locally. They assert the core feasibility property: the injected model degrades
gracefully to the base model when the readout is zero, yet a nonzero readout really
does perturb the output (the injection point is live).
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from reservoir.inject import ReservoirInjectedLM, base_logits  # noqa: E402

MODEL = "sshleifer/tiny-gpt2"  # tiny, fast; H1 holds for any base model
PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def injected():
    return ReservoirInjectedLM(MODEL, n_reservoir=64, seed=0, device="cpu")


def test_h1_zero_readout_matches_base_model(injected):
    base = base_logits(MODEL, PROMPT, device="cpu").detach().numpy()
    injected.reset_reservoir()
    got = injected.logits(PROMPT).detach().numpy()
    # W_out = 0 by default -> injection is a no-op -> identical logits.
    assert np.allclose(base, got, atol=1e-5)


def test_nonzero_readout_changes_output(injected):
    base = injected.logits(PROMPT).detach().numpy()  # W_out still 0 here
    rng = np.random.default_rng(0)
    injected.set_readout(rng.standard_normal((injected.d_model, injected.n_reservoir)))
    injected.reset_reservoir()
    perturbed = injected.logits(PROMPT).detach().numpy()
    assert not np.allclose(base, perturbed, atol=1e-4)
    injected.set_readout(np.zeros((injected.d_model, injected.n_reservoir)))


def test_h1_holds_on_llama_architecture():
    # the generalized injection must also work on a Llama-family model (Hermes is
    # Llama-based): decoder blocks at model.model.layers, width at config.hidden_size.
    llm = ReservoirInjectedLM("hf-internal-testing/tiny-random-LlamaForCausalLM",
                              n_reservoir=32, seed=0, device="cpu")
    base = base_logits("hf-internal-testing/tiny-random-LlamaForCausalLM", PROMPT,
                       device="cpu").detach().numpy()
    llm.reset_reservoir()
    got = llm.logits(PROMPT).detach().numpy()
    assert np.allclose(base, got, atol=1e-5)      # H1 on Llama arch too


def test_reservoir_state_persists_across_passes(injected):
    # The reservoir carries state between independent forward passes (a time axis):
    # two passes leave the state different from one pass.
    injected.reset_reservoir()
    injected.logits(PROMPT)
    after_one = injected.reservoir_state()
    injected.logits(PROMPT)
    after_two = injected.reservoir_state()
    assert not np.allclose(after_one, after_two)
