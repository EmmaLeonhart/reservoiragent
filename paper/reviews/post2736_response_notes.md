# Response to review of post 2736 (rating: Reject)

This review predates the preliminary 1.5B recall result (post 2736 was superseded by the version
that folds it in). Cons:

1. **"Primary positive limited to GPT-2-small; fails to generalize to 355M–3B."** This is the
   con the new result directly bears on: cross-pass recall now lifts off chance at **Qwen-1.5B**
   (stateful 0.83 vs wiped-control 0.17) with a larger reservoir at lower input scaling, folded in
   as a preliminary, control-verified finding. Reproduction + knob-isolation are running; once they
   land, the "GPT-2-small only" bound is materially revised, not just defended.

2. **"Safety/interruptibility is a trivial consequence of polling frequency."** Addressed: the
   latency half is stated as a consequence of sampling frequency (any per-tick agent gets it); the
   reservoir-specific claim is signal persistence (a one-shot STOP stays detectable in the
   reservoir state for 3 passes vs 0 for a stateless monitor).

3. **"Meta-commentary / dev-log style ('in response to review', 'withdrawn reframe')."**
   Addressed this revision: removed the meta framings — "(in response to review)", "(per review)",
   "we grant the reviewer's point", "an earlier over-reading", "the withdrawn reframe" — and
   restated each as a direct claim about the science. (Some of these were introduced precisely by
   addressing prior reviews; they now read as changelog and are gone.)

4. **"Battery invalidated by the authors' stateless control."** Correct and stated; the live
   demonstrations are the controlled tasks, and now the preliminary 1.5B recall result.

5. **"Bespoke forward loop hinders reproducibility."** KV-append is a standard k/v prefix; the
   constraint is that HF `generate` exposes no append hook. Integration constraint, not a
   non-standard method; implementation is open.

Net: con 1 is being overturned by new results; cons 2, 3 addressed; cons 4, 5 stated framing.
