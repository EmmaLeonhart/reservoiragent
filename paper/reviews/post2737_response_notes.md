# Response to review of post 2737 (rating: Weak Reject)

Strong pros (dynamics characterization, the stateless ablation's honesty, the additive-vs-KV
injection insight, transparency on negatives). Cons:

1. **"Cross-pass content recall fails to scale beyond GPT-2-small."** This is the con now being
   overturned by new results: with a 2048-node reservoir at input scaling 0.1, cross-pass recall
   scales to **Qwen-1.5B** — stateful 0.83 (seed 0) and **1.00 (seed 1)** vs a 0.17 wiped-state
   control, an isolation showing reservoir size is the necessary lever (scaling/prefix alone do
   nothing). The headline is being rewritten around this once the control + anti-memorization
   checks complete.

2. **"Agentic capabilities speculative / minimal probes."** Fair; the always-alive/agentic
   framing is the long-horizon target, not a demonstrated result. The live demonstrations are the
   controlled memory tasks (now at 1.5B) and the dynamics characterization.

3. **"Informal; references 'the report site'; non-standard terminology."** Addressed: replaced the
   repeated inline "(figure on the report site)" cross-references with neutral "(see figure)"
   pointers and reworded the figures header; kept only the single footer link to the site.

4. **"KV-append blocker hinders reproducibility."** KV-append is a standard k/v prefix; the
   constraint is that HF `generate` exposes no append hook, so we use an open bespoke loop.
   Integration constraint, not a non-standard method.

5. **"Safety claims lack rigorous adversarial evaluation."** Agreed; the interruptibility/monitoring
   results are synthetic proxies framed as motivation, not evaluated safety. Stated as such.

Net: con 1 is being overturned by the 1.5B recall result; con 3 addressed this revision; cons 2,
4, 5 are stated framing.
