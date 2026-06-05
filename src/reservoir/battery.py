"""The stateful-task battery — episode generators for the composite agent loss.

Eight task families, each a parameterised :class:`~reservoir.episode.Episode` generator
that *requires* the reservoir to carry state across a context wipe. The composite training
loss samples episodes from a weighted mix of these (see ``train_battery``). Generators are
pure given a numpy ``Generator`` (deterministic for a fixed seed) and unit-tested without
torch.

Note: this is ``battery.py`` for the *episode* battery. The older N-seed model batch lives
in ``batch.py`` — different thing, similar word.
"""
from __future__ import annotations

import numpy as np

from .episode import Episode, Step, SILENCE

# Short single-token words (verified single-token for GPT-2 with a leading space, so emit
# targets are one token). Kept small: a large vocab makes the content-memory tasks need far
# more samples to learn the mapping (the first 24-word run left recall at 0).
WORDS = ["red", "blue", "green", "gold", "black", "white",
         "cat", "dog", "star", "moon", "key", "fire"]


# Common function/pronoun words to exclude — they appear everywhere in prompts, so they
# make terrible "secret words" (non-distinctive recall targets).
_STOPWORDS = set("""the and for you are was with that this have from they will would there
their what when your who which been were them then than some more will into over also been
just like only your our out who has had his her its can may one two who why how all any new
get got our use now off per via yet etc""".split())


def large_word_pool(tokenizer, n: int = 1200, min_len: int = 3) -> list:
    """Build a large pool of real, lowercase, **single-token** content words (one token with
    a leading space) — so emit targets stay one token while the vocabulary is large. Scaling
    the word pool past a handful of words is how the content tasks stop being a toy.

    Prefers a real frequency-ranked English list (``wordfreq``) over scraping BPE fragments
    like "ing"/"ion"; falls back to a vocab scan if ``wordfreq`` is unavailable."""
    words, seen = [], set()

    def ok(w):
        return (w.isalpha() and w.lower() == w and len(w) >= min_len
                and w not in _STOPWORDS and w not in seen
                and len(tokenizer(" " + w, add_special_tokens=False)["input_ids"]) == 1)

    try:
        from wordfreq import top_n_list
        for w in top_n_list("en", 30000):
            if ok(w):
                seen.add(w)
                words.append(w)
            if len(words) >= n:
                return words
    except Exception:
        pass

    vocab = getattr(tokenizer, "vocab_size", 50000)   # fallback: scan the tokenizer vocab
    for tid in range(vocab):
        try:
            w = tokenizer.decode([tid]).strip()
        except Exception:
            continue
        if ok(w):
            seen.add(w)
            words.append(w)
        if len(words) >= n:
            break
    return words


def set_word_pool(words: list) -> None:
    """Replace the module word pool used by the generators (for large-scale training)."""
    global WORDS
    WORDS = list(words)


def _w(rng):
    return str(rng.choice(WORDS))


def gen_recall(rng) -> Episode:
    """Store a word, wipe the context, recall it (the proven cross-pass case)."""
    w = _w(rng)
    return Episode([
        Step(inject=f"The secret word is {w}."),
        Step(wipe=True, inject="The secret word was", target=f" {w}"),
    ], "recall")


def gen_accumulate(rng) -> Episode:
    """Add several numbers, context wiped between each, then report the total from state."""
    nums = [int(rng.integers(1, 6)) for _ in range(int(rng.integers(2, 4)))]
    steps = [Step(inject=f"Add {n}.", wipe=True) for n in nums]   # free ticks: sum into state
    steps.append(Step(wipe=True, inject="The total is", target=f" {sum(nums)}"))
    return Episode(steps, "accumulate")


def gen_sequence(rng) -> Episode:
    """Remember several items given over separate passes; recall them in order after a wipe."""
    k = int(rng.integers(2, 4))
    items = [str(x) for x in rng.choice(WORDS, size=k, replace=False)]
    steps = [Step(inject=f"Remember {w}.", wipe=True) for w in items]
    steps.append(Step(wipe=True, inject="In order, the words were",
                      target=" " + " ".join(items)))
    return Episode(steps, "sequence")


def gen_deferred(rng) -> Episode:
    """Note a fact, then many idle ticks, then recall it — fading memory under load."""
    w = _w(rng)
    delay = int(rng.integers(3, 8))
    steps = [Step(inject=f"Note: the keyword is {w}.")]
    steps += [Step(wipe=True) for _ in range(delay)]              # idle free ticks
    steps.append(Step(wipe=True, inject="The keyword was", target=f" {w}"))
    return Episode(steps, "deferred")


def gen_timed(rng) -> Episode:
    """Emit a word only after N passes — stay silent until then (a clock in state)."""
    w = _w(rng)
    n = int(rng.integers(2, 5))
    steps = [Step(inject=f"Say {w} after {n} steps.")]
    steps += [Step(wipe=True, target=SILENCE) for _ in range(n - 1)]   # silent, counting
    steps.append(Step(wipe=True, target=f" {w}"))                # now emit, unprompted
    return Episode(steps, "timed")


def gen_interrupt(rng) -> Episode:
    """Start A, get interrupted by B, then resume A from state."""
    a, b = (str(x) for x in rng.choice(WORDS, size=2, replace=False))
    return Episode([
        Step(inject=f"Start task {a}."),
        Step(wipe=True, inject=f"Wait, first do {b}.", target=f" {b}"),
        Step(wipe=True, inject="Now resume. The original task was", target=f" {a}"),
    ], "interrupt")


def gen_selfinit(rng) -> Episode:
    """Hold a pending item, stay silent a while, then raise it unprompted."""
    w = _w(rng)
    k = int(rng.integers(2, 5))
    steps = [Step(inject=f"Remind me about {w} soon.")]
    steps += [Step(wipe=True, target=SILENCE) for _ in range(k - 1)]
    steps.append(Step(wipe=True, target=f" {w}"))                # proactively raise it
    return Episode(steps, "selfinit")


def gen_silence(rng) -> Episode:
    """Nothing pending: every idle tick should stay silent."""
    k = int(rng.integers(2, 5))
    steps = [Step(inject="Just rest for now.")]
    steps += [Step(wipe=True, target=SILENCE) for _ in range(k)]
    return Episode(steps, "silence")


GENERATORS = {
    "recall": gen_recall, "accumulate": gen_accumulate, "sequence": gen_sequence,
    "deferred": gen_deferred, "timed": gen_timed, "interrupt": gen_interrupt,
    "selfinit": gen_selfinit, "silence": gen_silence,
}

DEFAULT_WEIGHTS = {k: 1.0 for k in GENERATORS}


def sample_episode(rng, weights: dict | None = None) -> Episode:
    """Sample one episode from the weighted mix of task generators."""
    weights = weights or DEFAULT_WEIGHTS
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=float)
    p /= p.sum()
    name = names[int(rng.choice(len(names), p=p))]
    return GENERATORS[name](rng)


def make_eval_set(rng, *, n_per_task: int = 8, weights: dict | None = None) -> list[Episode]:
    """A fixed evaluation set: ``n_per_task`` episodes for each enabled task."""
    weights = weights or DEFAULT_WEIGHTS
    eps = []
    for name in weights:
        for _ in range(n_per_task):
            eps.append(GENERATORS[name](rng))
    return eps
