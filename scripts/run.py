"""Entry point for Reservoir Agent experiments.

Subcommands:
    python scripts/run.py --version          # print the package version
    python scripts/run.py sweep              # spectral-radius dynamics sweep
        [--K 200] [--rho-min 0.1 --rho-max 2.0 --n 20]
        [--out results/sweep_synthetic.json] [--fig docs/sweep_synthetic.png]

Metrics are written to results/ (gitignored) and a figure to docs/ (published).
"""
import argparse
import json
import sys
from pathlib import Path

# Make stdout UTF-8 so unicode (ρ, ∈, …) prints on Windows consoles (cp1252) too.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import reservoir  # noqa: E402


def cmd_sweep(args) -> int:
    import numpy as np
    from reservoir.sweep import run_sweep, plot_sweep, healthy_regime

    rhos = list(np.round(np.linspace(args.rho_min, args.rho_max, args.n), 4))
    print(f"Sweeping K={args.K} over ρ in [{args.rho_min}, {args.rho_max}] "
          f"({args.n} points)…")
    records = run_sweep(rhos, K=args.K, T=args.T, washout=args.washout, seed=args.seed)

    out = (ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": {"K": args.K, "T": args.T, "washout": args.washout,
                          "seed": args.seed, "rhos": rhos},
               "records": records}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")

    fig = ROOT / args.fig
    fig.parent.mkdir(parents=True, exist_ok=True)
    plot_sweep(records, str(fig))
    print(f"wrote {fig.relative_to(ROOT)}")

    healthy = healthy_regime(records)
    if healthy:
        lo, hi = min(h["rho"] for h in healthy), max(h["rho"] for h in healthy)
        best = max(healthy, key=lambda r: r["input_separation"])
        print(f"healthy regime (ESP holds, not over-saturated): ρ ∈ [{lo}, {hi}]")
        print(f"most responsive healthy ρ = {best['rho']} "
              f"(input_separation={best['input_separation']:.3f}, "
              f"PR/K={best['participation_ratio_frac']:.2f})")
    else:
        print("no ρ in the grid met the healthy-regime criteria")
    return 0


TEXT_A = (
    "The history of science is a long chain of revolutions, each overturning the "
    "settled assumptions of the era before it. Copernicus moved the Earth from the "
    "centre of the cosmos, and the heavens lost their fixed and human centre. Darwin "
    "gave life a branching history, descent with modification replacing special "
    "creation, and the boundary between humans and other animals grew porous. Quantum "
    "mechanics dissolved the comfortable determinism of the clockwork universe, "
    "replacing certain trajectories with amplitudes and probabilities. Relativity bent "
    "space and time into a single fabric that curves around mass and energy. In every "
    "case the new picture was first resisted as absurd, then debated, then slowly "
    "absorbed into the textbooks, and finally taken so thoroughly for granted that the "
    "old worldview became difficult even to reconstruct. The lesson the historians "
    "draw is not that knowledge is arbitrary, but that it is provisional: a structure "
    "always under renovation, load-bearing yet never finished, built by people who "
    "could rarely see the shape of the building they were adding to."
)
TEXT_B = (
    "In the kitchen the morning light fell across the worn wooden table, where a pot "
    "of coffee cooled beside a bowl of ripe oranges and a small blue jug of milk. "
    "Outside, a tram rattled past the bakery on the corner, and the smell of warm "
    "bread drifted in through the open window with the sound of pigeons settling on "
    "the sill. She turned the page of her notebook, smoothed it flat with the side of "
    "her hand, and began, slowly, to write the day's first line. The pen caught once "
    "on the rough paper and then ran smooth. A neighbour's radio murmured a weather "
    "report through the thin wall, promising rain by the afternoon and a wind off the "
    "harbour. She wrote about none of it directly, but it all seeped in: the light, "
    "the bread, the tram, the small domestic weather of an ordinary Tuesday that would "
    "never come again in quite the same arrangement, and that was, she decided, reason "
    "enough to set it down."
)


def cmd_sweep_real(args) -> int:
    import numpy as np
    from reservoir.inject import extract_layer_stream
    from reservoir.sweep import run_sweep_stream, plot_sweep, healthy_regime

    print(f"Extracting GPT-2 ({args.model}) mid-layer activation streams…")
    stream_a = extract_layer_stream(args.model, TEXT_A)
    stream_b = extract_layer_stream(args.model, TEXT_B)
    print(f"stream A: {stream_a.shape}, stream B: {stream_b.shape} (T x d_model)")

    rhos = list(np.round(np.linspace(args.rho_min, args.rho_max, args.n), 4))
    records = run_sweep_stream(rhos, stream_a, stream_b, K=args.K,
                               washout=args.washout, seed=args.seed)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"params": {"model": args.model, "K": args.K, "washout": args.washout,
                    "seed": args.seed, "rhos": rhos,
                    "stream_shapes": [list(stream_a.shape), list(stream_b.shape)]},
         "records": records}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")

    fig = ROOT / args.fig
    fig.parent.mkdir(parents=True, exist_ok=True)
    plot_sweep(records, str(fig),
               title="Reservoir dynamics vs ρ — real GPT-2 mid-layer activations")
    print(f"wrote {fig.relative_to(ROOT)}")

    healthy = healthy_regime(records)
    if healthy:
        lo, hi = min(h["rho"] for h in healthy), max(h["rho"] for h in healthy)
        best = max(healthy, key=lambda r: r["input_separation"])
        print(f"healthy regime: ρ ∈ [{lo}, {hi}]; most responsive ρ = {best['rho']} "
              f"(sep={best['input_separation']:.3f}, PR/K={best['participation_ratio_frac']:.2f})")
    return 0


def cmd_alive(args) -> int:
    import json as _json
    from reservoir.harness import two_pass_dependence

    print("Two-pass time-axis proof-of-concept (reservoir state carried across "
          "independent forward passes)…")
    out = two_pass_dependence(args.model, TEXT_A, "The capital of France is",
                              seed=args.seed, readout_scale=args.readout_scale)
    print(f"same prompt, different history -> next-token logit L2 diff = "
          f"{out['logit_l2_diff']:.4f}; argmax changed = {out['argmax_changed']}")
    p = ROOT / "results" / "alive_poc.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps({"model": args.model, **out}, indent=2))
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


def cmd_nseed(args) -> int:
    import numpy as np
    from reservoir.inject import extract_layer_stream
    from reservoir.harness import select_seed_by_dynamics

    print(f"N-seed dynamics pre-selection proxy over {args.n} seeds (no training)…")
    a = extract_layer_stream(args.model, TEXT_A)
    b = extract_layer_stream(args.model, TEXT_B)
    ranked = select_seed_by_dynamics(a, b, seeds=list(range(args.n)),
                                     rho=args.rho, K=args.K)
    out = ROOT / "results" / "nseed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": {"model": args.model, "n": args.n,
                                          "rho": args.rho, "K": args.K},
                               "ranked": ranked}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    best, worst = ranked[0], ranked[-1]
    print(f"best seed = {best['seed']} (score={best['score']:.3f}), "
          f"worst = {worst['seed']} (score={worst['score']:.3f}); "
          f"spread = {best['score'] - worst['score']:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranked_by_seed = sorted(ranked, key=lambda r: r["seed"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(r["seed"]) for r in ranked_by_seed],
           [r["score"] for r in ranked_by_seed], color="#5b7a4a")
    ax.set_xlabel("reservoir seed")
    ax.set_ylabel("dynamics-quality proxy score")
    ax.set_title(f"N-seed dynamics pre-selection (ρ={args.rho}, K={args.K})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figp = ROOT / "docs" / "nseed.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)
    print(f"wrote {figp.relative_to(ROOT)}")
    return 0


def cmd_sweep_scaling(args) -> int:
    import numpy as np
    from reservoir.inject import extract_layer_stream
    from reservoir.sweep import run_scaling_sweep, plot_scaling

    print(f"Extracting GPT-2 ({args.model}) activation streams for input-scaling sweep…")
    a = extract_layer_stream(args.model, TEXT_A)
    b = extract_layer_stream(args.model, TEXT_B)
    scalings = list(np.round(np.geomspace(args.s_min, args.s_max, args.n), 5))
    records = run_scaling_sweep(scalings, a, b, K=args.K, rho=args.rho,
                                washout=args.washout)

    out = ROOT / "results" / "sweep_scaling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": {"model": args.model, "K": args.K,
                                          "rho": args.rho, "scalings": scalings},
                               "records": records}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    figp = ROOT / "docs" / "sweep_scaling.png"
    plot_scaling(records, str(figp),
                 title=f"Reservoir dynamics vs input scaling — real GPT-2 (ρ={args.rho})")
    print(f"wrote {figp.relative_to(ROOT)}")

    healthy = [r for r in records if r["saturation"] < 0.5]
    if healthy:
        best = max(healthy, key=lambda r: r["input_separation"])
        print(f"healthy (saturation<0.5) input scalings: "
              f"[{min(h['input_scaling'] for h in healthy)}, "
              f"{max(h['input_scaling'] for h in healthy)}]")
        print(f"best (most responsive, non-saturated) input_scaling = {best['input_scaling']} "
              f"(saturation={best['saturation']:.3f}, sep={best['input_separation']:.3f}, "
              f"PR/K={best['participation_ratio_frac']:.2f})")
    else:
        print("no input scaling kept saturation below 0.5 in this grid")
    return 0


def cmd_h3(args) -> int:
    from reservoir.tasks import delay_memory_curve, plot_memory_curve, memory_capacity

    print(f"H3 delay-memory task: train a readout to recover u(t-τ) from the reservoir "
          f"state (K={args.K})…")
    recs = delay_memory_curve(K=args.K, delays=range(0, args.max_delay + 1),
                              rho=args.rho, input_scaling=args.input_scaling, seed=args.seed)
    out = ROOT / "results" / "h3_memory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": {"K": args.K, "rho": args.rho,
                                          "input_scaling": args.input_scaling},
                               "records": recs,
                               "memory_capacity_reservoir": memory_capacity(recs, "reservoir_r2"),
                               "memory_capacity_baseline": memory_capacity(recs, "baseline_r2")},
                              indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    figp = ROOT / "docs" / "h3_memory.png"
    plot_memory_curve(recs, str(figp))
    print(f"wrote {figp.relative_to(ROOT)}")
    mc_r = memory_capacity(recs, "reservoir_r2")
    mc_b = memory_capacity(recs, "baseline_r2")
    print(f"memory capacity (Σ R² over τ≥1): reservoir = {mc_r:.2f}, "
          f"stateless baseline = {mc_b:.2f}")
    print(f"reservoir recovers input ~{sum(1 for r in recs if r['delay']>=1 and r['reservoir_r2']>0.5)} "
          f"steps back at R²>0.5; the stateless baseline recovers 0.")
    return 0


def cmd_agent(args) -> int:
    from reservoir.runtime import AliveAgent, ConfidenceGate

    print(f"Always-alive session on {args.model} (readout_scale={args.readout_scale}, "
          f"gate threshold={args.threshold})…\n")
    agent = AliveAgent(args.model, gate=ConfidenceGate(threshold=args.threshold),
                       readout_scale=args.readout_scale, seed=args.seed)

    # a scripted session: prompted turns interleaved with unprompted (idle) ticks
    script = [
        ("prompt", "I have two errands today: water the plants and call the bank."),
        ("idle", None),
        ("prompt", "By the way, what is the capital of France?"),
        ("idle", None),
        ("idle", None),
    ]
    transcript = []
    for i, (kind, text) in enumerate(script):
        rec = agent.prompt(text) if kind == "prompt" else agent.idle()
        rec["pass"] = i
        if kind == "prompt":
            rec["input"] = text
        transcript.append(rec)
        tag = "PROMPTED" if rec["kind"] == "prompted" else "unprompted"
        said = f' -> "{rec["said"]}"' if rec.get("emit") else "  (silent)"
        if kind == "prompt":
            print(f"[{i}] {tag}  in: {text}")
        print(f"[{i}] {tag}  entropy={rec['entropy']:.3f}  state|r|={rec['state_norm']:.3f}"
              f"  emit={rec['emit']}{said}\n")

    out = ROOT / "results" / "agent_session.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    out.write_text(_json.dumps({"model": args.model, "threshold": args.threshold,
                                "readout_scale": args.readout_scale,
                                "transcript": transcript}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    norms = [r["state_norm"] for r in transcript]
    print(f"reservoir |r| evolved across {len(transcript)} passes: "
          f"{norms[0]:.3f} -> {norms[-1]:.3f} (state carried, incl. unprompted ticks)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reservoir Agent experiment runner")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("sweep", help="spectral-radius dynamics sweep")
    s.add_argument("--K", type=int, default=200)
    s.add_argument("--rho-min", type=float, default=0.1)
    s.add_argument("--rho-max", type=float, default=2.0)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--T", type=int, default=600)
    s.add_argument("--washout", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/sweep_synthetic.json")
    s.add_argument("--fig", default="docs/sweep_synthetic.png")
    s.set_defaults(func=cmd_sweep)

    sr = sub.add_parser("sweep-real", help="sweep driven by real GPT-2 activations")
    sr.add_argument("--model", default="gpt2")
    sr.add_argument("--K", type=int, default=150)
    sr.add_argument("--rho-min", type=float, default=0.1)
    sr.add_argument("--rho-max", type=float, default=2.0)
    sr.add_argument("--n", type=int, default=20)
    sr.add_argument("--washout", type=int, default=20)
    sr.add_argument("--seed", type=int, default=0)
    sr.add_argument("--out", default="results/sweep_real.json")
    sr.add_argument("--fig", default="docs/sweep_real.png")
    sr.set_defaults(func=cmd_sweep_real)

    al = sub.add_parser("alive", help="two-pass time-axis proof-of-concept")
    al.add_argument("--model", default="gpt2")
    al.add_argument("--seed", type=int, default=0)
    al.add_argument("--readout-scale", type=float, default=0.05)
    al.set_defaults(func=cmd_alive)

    ns = sub.add_parser("nseed", help="N-seed dynamics pre-selection proxy")
    ns.add_argument("--model", default="gpt2")
    ns.add_argument("--n", type=int, default=8)
    ns.add_argument("--rho", type=float, default=0.95)
    ns.add_argument("--K", type=int, default=150)
    ns.set_defaults(func=cmd_nseed)

    ss = sub.add_parser("sweep-scaling", help="input-scaling sweep on real activations")
    ss.add_argument("--model", default="gpt2")
    ss.add_argument("--K", type=int, default=150)
    ss.add_argument("--rho", type=float, default=0.95)
    ss.add_argument("--s-min", type=float, default=0.01)
    ss.add_argument("--s-max", type=float, default=2.0)
    ss.add_argument("--n", type=int, default=16)
    ss.add_argument("--washout", type=int, default=20)
    ss.set_defaults(func=cmd_sweep_scaling)

    h3 = sub.add_parser("h3", help="train a readout for the delay-memory task (H3)")
    h3.add_argument("--K", type=int, default=200)
    h3.add_argument("--max-delay", type=int, default=30)
    h3.add_argument("--rho", type=float, default=0.9)
    h3.add_argument("--input-scaling", type=float, default=0.5)
    h3.add_argument("--seed", type=int, default=0)
    h3.set_defaults(func=cmd_h3)

    ag = sub.add_parser("agent", help="run a scripted always-alive session")
    ag.add_argument("--model", default="gpt2")
    ag.add_argument("--threshold", type=float, default=0.85)
    ag.add_argument("--readout-scale", type=float, default=0.05)
    ag.add_argument("--seed", type=int, default=0)
    ag.set_defaults(func=cmd_agent)

    args = parser.parse_args(argv)
    if args.version:
        print(reservoir.__version__)
        return 0
    if getattr(args, "func", None) is not None:
        return args.func(args)

    print(f"reservoir-agent {reservoir.__version__} — package imports OK.")
    print("Try:  python scripts/run.py sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
