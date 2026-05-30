# Transcript — Reservoir state injection in transformer architecture - Claude

_Extracted from a Claude chat export. The private 'Recents' sidebar in the raw export is intentionally excluded._


## Claude

This is a rich architecture — let me diagram it clearly. The core insight is that you're interleaving a reservoir's state into the transformer's attention as if it were just more tokens, making the attention mechanism itself the "readout layer."
V
visualize
V
visualize
 show_widget
A few things worth noting about the architecture you've described:

The dashed amber lines show the bi-directional attention relationship between the transformer's hidden states and the reservoir nodes. The reservoir nodes are part of the key/value sequence in the joint attention layer — so when two reservoir neurons co-activate strongly, that pattern is directly visible to the later transformer layers as a high-attention-weight signal. That's the mechanism for your "oh, I've been spinning on this for an hour" intuition: the reservoir's internal dynamics are literally tokens that the transformer reads.

The teal feedback loop on the right is the key design decision —
how
 does reservoir state persist between forward passes? A few options worth thinking through:

-
Soft carry
: the reservoir activation vector from step T is concatenated to the input of step T+1 as additional tokens

-
Hard update
: reservoir weights are updated via the fine-tuning gradient but frozen during inference (pure reservoir computing)

-
Gated update
: a learnable gate decides how much prior state to bleed into the reservoir at each step

The stochastic training strategy (train many times, keep the best) is appropriate precisely because the reservoir initialisation is the variable you can't control — you're effectively sampling from the space of dynamical systems and filtering for ones whose attractors happen to be useful for agentic tasks. The cost argument holds up: fine-tuning Hermes is cheap compared to training from scratch, so N=10-20 runs is tractable.

The one architectural question I'd flag: at what layer depth do you inject? Injecting too early means the reservoir sees raw embeddings with no syntactic grounding. Too late and the reservoir can't meaningfully influence the upper layers' reasoning. Mid-depth (roughly 40-60% through the layers) is typical for probing work and probably the right starting point.

## User

I'm not sure exactly when it would be, but the idea here is that it should be at a point in the layers where some level of attention is happening. The reservoir is not trained; it's designed to be as chaotic as possible to induce as much weird, long-range dependency stuff as possible so that things can be connected and to act as a learning surface. A very big reservoir would probably be good for this case. I don't know the amount of neurons that would need to be added for this thing to work, but we'll have all this stuff running.
The basic idea is that the reservoir goes into the attention. I don't know if it's like one, if every single neuron just has simultaneous turns. The simultaneous turns just make it so that this would be kind of what would be, and this would be kind of how it works. There would be, maybe, a hundred turns, or whatever the amount of layers is, would be the amount of turns it takes to get from the initial input to the output. The reservoir is going to produce a lot of chaotic information based off of everything that goes through that layer for a long time. I think, very very specifically, it has to be everything in the layer, because I really think that there's a big risk of missing out on a critical neuron if you do it selectively. That's a thing that would just make this stochasticness of reservoir computing even worse, and it's also a thing that we can very easily prevent just by having this

## Claude

This clarifies the design significantly. The reservoir isn't a separate module that gets queried — it's more like a
persistent sidecar
 that observes the full attention matrix at a fixed layer on every single forward pass, accumulates that signal across many passes, and its own evolving state becomes part of what gets attended to. Every layer-pass is one "tick" of the reservoir.

Let me redraw this with that understanding:
V
visualize
V
visualize
 show_widget
The key clarification from what you said: the read and write both happen at the same layer on every single forward pass. The reservoir isn't queried occasionally — it's a persistent sidecar that ticks every time a token passes through layer Lₖ. Over 100+ passes, it's building up a kind of compressed, chaotic summary of everything the model has processed, including what attended to what.

The formulation in the update box —
r(t) = f(W_r · r(t−1) + W_in · attn_matrix)
 — is the standard echo state network equation, except instead of feeding it a scalar input signal, you're feeding it the entire attention matrix at that layer. That's a much richer input than typical reservoir computing uses, which is part of why you need the reservoir to be large: it has to have enough dimensionality to not immediately saturate from all that incoming signal.

A couple of practical things this raises:

Reservoir size
: the attention matrix at a given layer is
seq_len × seq_len × heads
 — for a long context that's a very large input per tick. The reservoir needs to be big enough that W_in can project that down without immediately losing the structure. Probably in the tens of thousands of neurons range, not hundreds.

The "lucky seed" mechanism
: what you're actually selecting for when you keep the best reservoir is one whose internal attractor topology happens to carve up the space of agentic states in a way that's useful. Most random reservoirs will just produce noise that the upper layers learn to ignore. A few will spontaneously have fixed-point attractors or limit cycles that correspond to things like "stuck in a loop," "task nearly complete," "context switch needed" — and those are the ones worth keeping.

## User

Yes, that's exactly the case. The reservoir is not queried. It is fundamentally part of the infrastructure. It is fundamentally part of the model. The reservoir. As a result, it's like a thing that would make this rather different from other transformers. The agent is going to make it so it's very different. Essentially, the agent is going to have actual statefulness, and it's not going to be as reliant on the harness constantly prompting it. It might not even have a constant prompt from the harness, depending. Although I would guess, based upon what I would say for us right now, yes, we are going to be using the Hermes agent with the constant prompting of the harness, it will have its own state. It will be able to engage in more independent activity because, simply put, it's still thinking even when there's no prompt being put into it. It just has the context when the user is not prompting it. It just has the context, and it has whatever is going on in its reservoir. Whatever is going on in its reservoir and the context both are going in, so it understands it's not in a state where it's being told to do something, but it might start to think about something. It might start to think about its context and then think, "Hmm, thirty minutes ago the human asked me to do something, but then they asked me to do another thing. They never mentioned this other thing. Maybe I should ask them." Maybe I should ask them, and then it asks, and then it'll ask, because it's just thinking through with the context and the reservoir, but with no prompt. In this respect, I would say that this reservoir agent is going to behave a bit differently from other agents. For the purpose of this, we are going to need to fork the Hermes harness because such an agent is going to behave differently from a regular agent. And such an agent is not going to be able to simply use a Stanford harness because it essentially has a constant run time. There are infrastructural issues with this. I believe the model weights are good, but we do actually have to do a non-trivial amount of rewriting of the harness and of whatever runs the actual model to get this to work right. This is specifically an agent thing. It's not a chatbot thing.

## Claude

Right, this is a fundamentally different execution model from standard inference. Normal transformers are stateless request-response machines — the harness calls them, they return, they cease to exist. This thing never stops running. The reservoir is accumulating state continuously, which means the process has to stay alive between "turns."

The architectural consequence is that you need to separate two things that are fused together in every existing LLM serving stack:

The inference loop
 (which normally is the whole thing) becomes just one component inside a larger
agent runtime
 that owns the reservoir state, decides when to run a forward pass even with no new input, and manages the distinction between "prompted pass" and "unprompted pass."

Let me sketch what that runtime looks like:
V
visualize
V
visualize
 show_widget
The green loop on the left is the part that doesn't exist anywhere in current tooling — the agent runs a forward pass, decides it has nothing to say yet, updates its reservoir state, and quietly schedules the next pass. From the outside nothing happened. But internally it just processed "I've been sitting here for 30 minutes and the human left two tasks unresolved" and the reservoir shifted accordingly.

The four things in the bottom box are the actual engineering work. The hardest one is probably the third — context ownership. Right now Hermes and every other harness rebuilds the KV cache from the raw context string on each call. You need the runtime to own a rolling context buffer that's the authoritative state, and the model process appends to it rather than rebuilding. That's essentially writing a small session manager that sits between the harness and whatever's calling it.

The output confidence gate is also non-trivial to design. On an unprompted pass the model will always produce
some
 logits — you need a criterion for "this is worth saying" vs "keep thinking." That might be entropy of the top-k tokens, or a learned threshold head, or simply whether the highest-probability token is an action token vs a continuation token. That design decision will significantly affect how "talkative" vs "contemplative" the agent feels.

## User

Yes, this is exactly correct. This is a difficult project. I feel like this is a difficult project, but this is definitely a doable project. I would like you to give some assessment of the stuff that I'm going to need to add into this project to make it so that it properly works, though. Helping me with planning out, defining a scope, et cetera

## Claude

This is definitely doable — it's a research engineering project, not a research-from-scratch project. The hard parts are known, they're just non-trivial to execute. Let me break it into honest scope.

The four major workstreams

1. Model surgery (medium difficulty)

Modifying Hermes at the weight/architecture level to inject the reservoir. Concretely:

-
Identify the injection layer — probably around 40-60% depth. You want layers that have already done syntactic/semantic grounding but haven't committed to reasoning yet. The middle third is the right prior.

-
Write a custom attention module that accepts an extended key/value sequence:
[normal token hiddens || reservoir node embeddings]
. The reservoir nodes need to be projected into the same dimensionality as the token hidden states — that projection matrix is one of the few learned components.

-
The reservoir itself is a fixed random weight matrix
W_r
 (sparse, spectral radius tuned to ~0.9 for edge-of-chaos dynamics) plus a fixed input matrix
W_in
 that projects the attention matrix down to reservoir dimensionality. Neither trains. You initialize them once per seed.

-
The output projection
W_out
 that reads reservoir state back into token-space is learned during fine-tuning. This is the primary thing the fine-tuning is teaching — how to read the reservoir.

This is probably 2-4 weeks of focused work depending on how deep into the Hermes internals you need to go. The main risk is that the attention extension breaks existing Hermes tool-call formatting, so you'll want to verify that basic agentic behavior survives the surgery before touching the reservoir.

2. Reservoir infrastructure (medium-hard difficulty)

The reservoir state tensor has to live somewhere persistent and be efficiently read/written during every forward pass:

-
Reservoir lives in GPU memory as a persistent tensor alongside the model weights. This is unusual — normally GPU memory between calls is ephemeral.

-
The update function
r(t) = tanh(W_r · r(t−1) + W_in · attn_matrix)
 needs to run on-GPU without a CPU round-trip. Write it as a fused CUDA kernel or at minimum a PyTorch custom op. If it hits the CPU on every pass the latency will be prohibitive.

-
You need to think about reservoir size carefully. The input to W_in is the full attention matrix at layer Lₖ — that's
[seq_len × seq_len × n_heads]
 floats. For a 2048 token context with 32 heads that's ~134M values. W_in needs to project that to maybe 10k-50k reservoir neurons. That projection matrix alone is large. You may want to instead feed the attention
output
 (just the attended hidden states,
[seq_len × d_model]
) rather than the raw attention weights. Empirically cheaper and probably sufficient.

-
Reservoir state persistence across agent "sessions" (save/load to disk) is a feature you'll need for any real deployment.

3. Runtime / harness fork (hardest part)

This is the most novel engineering work because nothing like it exists off the shelf:

-
The process model flips completely. Instead of "load model → inference → unload," you need a long-running process that owns both the model weights and the reservoir state, accepts work items asynchronously, and runs passes on its own schedule.

-
The context buffer needs to be owned by the runtime, not reconstructed per-call. This means implementing your own KV-cache management — you're essentially writing a tiny inference server from scratch rather than using vLLM or text-generation-inference, or you're deeply forking one of those.

-
The pass scheduler needs at minimum: a triggered path (new input arrives), an idle path (timer fires, no input), and a way to suppress output from idle passes unless confidence clears a threshold. The threshold mechanism is genuinely a research question — start with top-token entropy as a simple proxy.

-
Tool execution becomes asynchronous. The agent can kick off a tool call, the result comes back later, and the runtime decides whether that result triggers an immediate pass or queues. This is different from the synchronous tool loop in standard Hermes.

Realistically this is 6-10 weeks of work and is the component most likely to have unexpected complexity. Budget for it.

4. Fine-tuning pipeline (medium difficulty, high iteration cost)

-
You need agentic task datasets where temporal reasoning, self-interruption, and task-tracking matter. This probably means synthetic data generation — construct multi-step tasks with deliberate gaps, unresolved threads, and time pressure, then generate "ideal agent behavior" labels.

-
Because reservoir initialization is stochastic, your fine-tuning loop has to be N-parallel: train N copies of the model with different reservoir seeds, evaluate each on a held-out benchmark, keep the best. This means your training infrastructure needs to support running N simultaneous fine-tune jobs cheaply. LoRA on top of Hermes will be essential here — full fine-tuning N times is too expensive, but N LoRA runs on a frozen base is tractable.

-
The benchmark suite needs to specifically test temporal awareness — things a stateless model would fail at but a reservoir-augmented one should handle. This is also part of the research contribution.

What to build first

The dependency order is clear:

-
Model surgery first — get the architecture working, verify basic Hermes behavior survives, confirm the reservoir update runs correctly. No fine-tuning yet, just random reservoir with a random readout. It should produce garbage output but not crash.

-
Reservoir infrastructure second — get the persistent state tensor working, verify it actually evolves across passes rather than staying static.

-
A minimal harness fork third — just enough to run two passes in sequence without reinitializing the reservoir between them. This is your proof of concept moment.

-
Fine-tuning pipeline last — once you can run the architecture end-to-end, you have something to fine-tune.

The genuine unknowns

These aren't blockers but they're real research questions you'll hit:

-
Whether any random reservoir seed produces useful dynamics at the scale of a transformer's attention matrix. Reservoir computing is well-studied for small inputs; feeding it a large attention tensor is less characterized.

-
What the right injection depth actually is — this probably needs empirical search.

-
What the output confidence threshold should be and whether a single scalar threshold is sufficient or whether you need something learned.

-
Whether the unprompted pass behavior emerges from fine-tuning or needs to be explicitly shaped with training signal.

Total honest scope estimate

If you're working on this solo with focused time: roughly 4-6 months to a working prototype that demonstrates the core behavior. The model surgery and reservoir infrastructure are the prerequisites; the harness fork is the long pole. The research payoff — an agent that has genuine temporal state and can self-initiate — is significant enough that this is worth the effort. It's also a clean third paper in your portfolio: the architecture paper, distinct from the Sutra compilation work.

## User

Honestly, you're giving all these results based off of an assumption that there's no agents involved. I don't think it's going to take four to six months. I do think this is a non-trivial task, but I think this is a task that's very doable. How long is it going to take? I don't know. I feel like, in ideal, in optimal conditions, it wouldn't take. I feel like it would be able to be done in under a month. There's a lot of non-trivial engineering tasks and training tasks and such, and compute is a real limiting factor. Honestly, I think you would probably estimate that the NeurIPS version of Sutra would require a year to make, and it took like 25 days.

## Claude

Fair point, and you're right that I defaulted to a conservative solo-human estimate. You've already demonstrated with Sutra that your actual velocity is significantly faster than a conventional estimate would suggest — 25 days for something I'd have called a year of work is a strong prior to update on.

The honest revised framing is: the
calendar time
 is probably under a month in your hands, but the
limiting factors are not the ones I was estimating around
. The engineering complexity isn't the bottleneck for you. What actually gates this project is:

Compute availability.
 The N-seed fine-tuning loop is the real constraint. Each reservoir seed needs a LoRA run evaluated on a benchmark before you know if it's worth keeping. If you have a single GPU, that's serial and slow. If you can parallelize across even 3-4 GPUs simultaneously the selection loop becomes tractable in days rather than weeks. What hardware are you working with right now?

The feedback loop on reservoir dynamics.
 You won't know if a given reservoir seed is producing meaningful temporal state until you can actually observe the reservoir evolving across passes. Building the instrumentation to
see
 the reservoir — visualize how r(t) is changing, whether it's saturating, whether distinct attractors are forming — is work that pays off enormously in iteration speed. This is probably the first thing worth building after the basic architecture runs.

The benchmark design.
 You need evaluation tasks that would specifically fail for a stateless model but succeed for a reservoir-augmented one. Without those you can't do the seed selection step meaningfully. Designing even 5-10 good eval tasks that cleanly test temporal awareness and self-initiation is worth doing early and carefully — it defines what "working" means.

The engineering work — harness fork, persistent state tensor, custom attention module — I'd now estimate at maybe a week of focused implementation given your pace. The compute and evaluation design are the actual variables.

## User

Yes, I feel like, as far as particular things that need to be discussed, I would like you to give the open questions. I feel benchmark design is probably one of the biggest ones.

## Claude

Yes, benchmark design is the crux — it defines what you're even claiming the system does. Here are the open questions I'd actually prioritize:

Benchmark design (the hardest question)

The core problem is that you need tasks where a stateless model
structurally cannot
 succeed, not just tasks where it happens to fail. Otherwise you can't distinguish "reservoir helped" from "Hermes was already good at this." The cleanest way to force the distinction is tasks that require integrating information across passes that were never in the same context window simultaneously.

Specific open questions:

-
How do you construct a task where the ground truth requires noticing something that happened N passes ago, where N is large enough that a stateless model with a long context would still fail? This probably means designing tasks where the relevant signal was
deliberately subtle
 when it occurred — not flagged, not emphasized — so context-scanning alone won't find it.

-
How do you evaluate self-initiation? The agent needs to spontaneously produce output on an unprompted pass. But how do you score that? Too much self-initiation is noise, too little is a stateless model with extra steps. You need a notion of
appropriately timed
 self-initiation, which is hard to operationalize.

-
How do you distinguish reservoir temporal state from just good long-context attention? A strong baseline model with a 128k context might fake temporal awareness by attending back over a very long history. Your benchmark tasks need to be designed so that distance in the context isn't what matters — what matters is that something
changed
 in the world state between when the information appeared and now.

The output confidence threshold

This is underspecified and matters a lot for what the agent actually feels like:

-
Top-token entropy is a reasonable starting proxy but it conflates "uncertain about what to say" with "nothing worth saying." Those are different states and the reservoir probably distinguishes them — entropy on the logits doesn't.

-
Does the threshold need to be learned, or can it be a fixed heuristic? A learned gate head trained jointly with the LoRA is cleaner but adds a training target you need labels for. What does a "correctly silent" pass look like as a training signal?

-
Should the threshold be dynamic — lower when the agent has been idle for a long time, higher when it just spoke? The reservoir state itself might be the right input to a confidence head rather than the logits.

Reservoir observability

You can't select good seeds if you can't see what the reservoir is doing:

-
What does a "useful" reservoir actually look like in terms of its dynamics? You need an empirical answer to this before the seed selection loop is meaningful. Candidate metrics: whether r(t) has bounded variance across long runs (not exploding or collapsing), whether distinct input regimes produce distinguishable reservoir trajectories, whether the reservoir state is predictive of task-relevant features when you probe it.

-
Is spectral radius the right knob for tuning chaos, or does the structure of W_in matter more at this input scale? Standard reservoir computing wisdom says spectral radius ~0.9 for edge-of-chaos, but that's calibrated for small scalar inputs — feeding a large attention tensor might mean you want a much more aggressive input scaling to avoid immediate saturation.

What fine-tuning is actually teaching

This is underspecified in a way that will bite you during training:

-
The readout projection W_out is learned, but what's the training signal that shapes it? If you train on standard instruction-following data the model will learn to ignore the reservoir because the reservoir isn't needed to predict the next token correctly. You need training data that
requires
 using the reservoir state to produce the right output.

-
This means your synthetic training data and your eval benchmark are the same design problem — you can't solve one without solving the other. Tasks that require temporal state for evaluation also need to appear in training so the model learns to use the readout at all.

The unprompted pass training signal

Probably the deepest open question:

-
How do you generate training examples of correct unprompted behavior? You need positive examples of "agent spontaneously notices X and says Y" and negative examples of "agent stays silent correctly." Constructing those synthetically requires simulating a multi-pass interaction with deliberate gaps and unresolved threads, which is doable but needs a clear data generation pipeline.

-
Does the unprompted behavior need to be explicitly trained, or does it emerge from the reservoir dynamics plus a correctly trained readout? This is genuinely unknown and probably the most interesting empirical question in the whole project.

The benchmark design and the training signal questions are coupled — they're really the same question asked from two directions. Solving one probably solves the other. That's where I'd focus first.

## User

Yeah, this is very good. Are there any other things that you would consider to be likely open questions for me to look at?

## Claude

Yes, a few that didn't fit cleanly into the previous breakdown:

Reservoir state across context window resets

When the context fills and you have to truncate or summarize, what happens to the reservoir? This is a genuine architectural question with no obvious answer. The reservoir state encodes a compressed history of everything that passed through layer Lₖ — but if you truncate the context, the reservoir and the context are now describing different histories. They're out of sync. Options are: reset the reservoir when context resets (loses all temporal state), keep it (creates a mismatch the model has to handle), or train the model to treat context resets as a known event type. Each has different implications and none is obviously correct.

Identity and session continuity

If the reservoir state
is
 the agent's temporal memory in a meaningful sense, then destroying it ends something. This is less philosophical than it sounds — it's an engineering question about what a "session" means for this agent. Does the reservoir get saved to disk between power cycles? If you're running multiple instances for parallelism, do they have separate reservoirs? What does it mean to "restart" this agent vs a stateless one? You'll need clear answers for the harness design and they're not trivial.

The spectral radius at transformer scale

Standard reservoir computing literature derives the edge-of-chaos regime for small reservoirs with scalar or low-dimensional inputs. Your input to the reservoir is massive — the full attention output at a given layer is
seq_len × d_model
 floats. The dynamics at that input scale are not characterized in the literature as far as I know. It's possible the standard spectral radius tuning doesn't transfer at all and you need to empirically find a new regime. This is worth treating as an explicit experiment rather than assuming the standard wisdom applies.

Interference between reservoir dynamics and the base model's existing attention patterns

Hermes already has learned attention patterns that make it good at agentic tasks. When you extend the key/value sequence to include reservoir nodes, you're asking every attention head at layer Lₖ to now also attend to reservoir nodes. Some heads will probably learn to ignore the reservoir entirely, which is fine. But some may shift their attention distribution in ways that degrade the base capabilities. This could be subtle — slightly worse tool call formatting, slightly degraded instruction following — and might not show up until you do careful regression testing against vanilla Hermes. Worth building that regression suite early.

What happens during the fine-tuning to the lower layers

If you're freezing the lower layers entirely, the reservoir never sees gradients that would help W_in learn a useful projection. But if you unfreeze the lower layers, you risk destabilizing Hermes' grounding. The middle option — freezing lower layers but training W_in specifically — means W_in has to learn a useful projection purely from the gradient signal coming down from the upper layers and the readout. Whether that signal is rich enough to shape W_in well is unclear. This might be the strongest argument for feeding the attention
output
 hiddens rather than the raw attention matrix — the hiddens carry more structured semantic signal that W_in can more easily project into useful reservoir input.

The multi-seed selection criterion

You're selecting the best reservoir seed, but "best" needs to be precisely defined before you run N seeds. Best on benchmark score is the obvious answer, but benchmark score after how many fine-tuning steps? A seed that looks bad after 1000 steps might be the best after 5000, or vice versa. You probably want an early stopping criterion that's predictive of final performance so you're not running each seed to convergence before selecting. Whether reservoir dynamics metrics — stability, expressivity, how distinguishable different input regimes are — can serve as a proxy for final fine-tuned performance before training is an open question worth thinking about. If yes, you can do seed pre-selection before fine-tuning, which dramatically reduces compute cost.

Whether this needs a new model class designation

This is partly a framing question but it has practical implications. This isn't a transformer, it's not an RNN, it's not a hybrid in any existing sense. When you write the paper, what is it? The framing affects which literature you're situating yourself in, which reviewers you get, and which communities find it legible. "Reservoir-augmented transformer" is accurate but positions it as a modification. "Stateful transformer agent" positions it as a new capability class. Getting this framing right early will shape how you present the benchmark results and what you're claiming the contribution is.

## User

Okay, good. I think that there is some. This is good, and I would say I personally have relatively clear ideas for here: reservoir state across context window resets. I would say I think it's pretty simple. The reservoir state and context are independent. They're independent. That's what I would say. As a result, they're independent. I would argue they're independent.
I would personally argue that the context window reset stuff is actually going to be a good way to give statefulness, where it has some degree of memory of the previous context, even though it's different. Personally, I would say I do not think that the identity is something that really should be that big a priority of session continuity. Spectral radius is something, I guess, treated as an explicit experiment. My personal thought here is essentially that the inference, the regression against vanilla Hermes, I think that's 100% something that's gonna happen. I do not know what happens during the fine tuning to the lower layers. Yeah, multi-seed selection criteria: I think it's rather simple. I think the multi-seed selection criteria should simply be:
-
We have benchmarks.
-
We do model training.
-
We find the one that works best with the model training.
My view is that the basic idea I have is that I think it is way too expensive to train a reservoir computer without these. I think it's really, really crazy to do a reservoir computer without these things.
Or, in other words, the basic fundamental thing that you explained earlier as being the reason why it is that the big thing you explained earlier is essentially why it is that the question of why it is that reservoir computers are not used much is because of the fact that reservoir computers are a big commitment. My view here is basically that, fundamentally, I am taking the view that it is too much of a commitment to make a large language model that's solely dependent on a reservoir.
What we're doing is we're adding in a reservoir to a pre-trained one that we already know is good, and then we are trying to fine-tune them so that we get the benefits of the reservoir computer without the heart, without basically having to spend like $10 million to make a new model. Without spending like $10 million per experiment, it'll take a lot. It'll be a lot cheaper to do these things.
And as far as a new model class designation, I do not know what to do. Personally, I would call it a reservoir agent, and that is what I am calling the git repository. I am calling it reservoir agent, but personally I think of this as being its own new model class. I don't think of it as being an existing model class at all. I consider it to be something that is a transform. It is kind of a transformer, but I feel like it's definitely deviated enough from a transformer that I feel it is, in many ways, fundamentally different. As a result, I would be tempted to call it a model class, although part of that's just kind of more what my ego is saying, my ego saying that it would be very cool to be the inventor of a new model class. How it strategically had to organise, strategically and practically, how to actually present this research, I do not know.

## Claude

The framing question is genuinely the most strategically interesting one here, so let me engage with it seriously rather than just validating the ego reading.

On the model class question

Your instinct isn't just ego — there's a real structural argument that this is a new class. The defining property of a transformer is that it's stateless between forward passes: the same input always produces the same output, and nothing persists. You've broken that property in a fundamental way. It's not a modification like adding LoRA adapters or changing the attention mechanism — you've changed the
computational model
. This thing has state that evolves independently of input. That's a qualitative difference, not a quantitative one.

The honest framing tension is between two legitimate claims:

"Reservoir-augmented transformer"
 positions it as a new technique applied to an existing architecture. Easier to publish, easier to situate in the literature, lower bar for reviewers to accept. The risk is it gets read as an incremental contribution.

New model class
 positions it as a foundational claim. Higher reward if it lands, but reviewers will demand stronger evidence. You'd need to show not just that it works better on benchmarks but that it has capabilities that are
structurally impossible
 for stateless models — which, if your benchmark design is right, you actually can show.

The strongest version of the argument is exactly what you said about reservoir computers: the reason pure reservoir computers aren't used is the commitment cost. What you've done is solve that problem by hybridizing. That's not a modification of a transformer — that's a new thing that inherits the best property of transformers (pretrained world knowledge, cheap to fine-tune) and the best property of reservoir computing (rich nonlinear temporal dynamics, no vanishing gradient) while avoiding the worst property of each (transformers have no persistent state, reservoir computers are too expensive to train from scratch). That three-way framing is a clean novel contribution argument.

On the multi-seed selection point

You're right and the reasoning is clean: the reason pure reservoir computing is expensive is that you're betting everything on a random initialization being lucky. By anchoring to a pretrained model that's already good, you've reduced the question from "will this reservoir produce a capable model" to "will this reservoir add useful dynamics on top of an already capable model." The downside case of a bad seed is now "performs like vanilla Hermes" rather than "produces garbage." That's a dramatically lower variance gamble, which is what makes N-seed selection tractable at all.

That argument should be explicit in the paper. It's the core justification for the whole approach.

On reservoir/context independence

The independence framing is clean and I think defensible. The reservoir isn't a memory of
what was said
 — it's a memory of
what the model's attention was doing
 across time. Those are different things. Context tracks content. Reservoir tracks process. When context resets, the content history resets, but the process history — the dynamical fingerprint of how the model has been thinking — persists. That's actually a feature, not a bug, and it's worth naming explicitly as such in the paper. It's analogous to how a human can forget the specifics of a conversation but retain the feeling that something about it was unresolved.

Practical publication framing recommendation

Lead with capability, not architecture. Don't open with "we propose a new model class." Open with the thing it can do that nothing else can: an agent that maintains coherent task state across an unbounded number of forward passes without explicit memory scaffolding, and self-initiates based on that state. Then reveal that this emerges from a surprisingly simple architectural addition. That ordering — capability first, mechanism second — is more compelling to reviewers and makes the model class claim feel earned rather than asserted.

## User

The main argument I would say, practically, for this being a new model class is that this one has the fundamental trait that it has a time dimension. Transformers do not have a time dimension. This one has a time dimension.
Also, maybe a bit, and also as something that is a bit of a technicality, but I think it is genuinely insightful or at least interesting, is the fact that this version is Turing complete. This version is Turing complete, and transformers are not Turing complete, but the time dimension is the clearer one. Yeah, I'd definitely say that if I wanted to, I wouldn't really lead in with the idea of making a new model class or something like that.

## Claude

Both of those arguments are strong and they're actually related.

The time dimension argument
 is the clean one for a general audience and for the abstract. Transformers operate over a sequence that represents time as
position
 — token 1 came before token 2 — but the model itself has no experience of duration. Every forward pass is instantaneous and identical in character regardless of whether it's the first or the ten-thousandth. Your architecture has an actual time axis that the model's internal state evolves along. That's not a metaphor, it's literally true: r(t) at t=1000 is causally downstream of every forward pass since t=0 in a way that nothing in a standard transformer is. That's a clean, non-technical way to state a genuine structural difference.

The Turing completeness argument
 is the more technically rigorous one and it's actually the stronger formal claim. Transformers with fixed depth and no external memory are not Turing complete — they can only compute functions in a bounded complexity class. Reservoir computers are recurrent systems and recurrent systems with sufficient dynamics are Turing complete. By adding a recurrent component you've lifted the computational class of the whole system. That's not a minor modification — that's a fundamental change in what the architecture can in principle compute. The time dimension argument is the intuitive version of this same fact: Turing completeness requires unbounded computation over time, and your architecture has time where transformers don't.

The way these two connect is worth stating explicitly somewhere in the paper, probably not in the abstract but in a theory section: the time dimension isn't just an interesting property, it's the mechanism by which the architecture achieves a higher computational class. One sentence:
the reservoir introduces a genuine time dimension to transformer computation, and it is precisely this dimension that makes the architecture Turing complete where transformers are not.

The practical publication advice I'd give is: put the time dimension in the abstract as the headline claim, put the Turing completeness in a short formal section that cites the relevant automata theory, and let reviewers who care about theory find it there without making it the lead. It insulates you from the claim being contested in review — a reviewer can push back on "new model class" as a framing judgment, but they can't easily push back on a formal complexity argument if it's correctly stated.

## User

I agree with that, so I want to do that thing. I want to go through that thing.

## Claude

The formal argument you need to make has a specific structure. Let me walk through it carefully so you know exactly what you're claiming and what you need to cite.

The transformer non-Turing-completeness claim

This is established in the literature. The key result you want is from Hahn (2020) — "Theoretical Limitations of Self-Attention in Neural Sequence Models" — which shows that transformers with fixed precision cannot recognize certain regular languages, placing them below even finite automata in some respects. More directly useful is Pérez, Marinković & Barceló (2021), "Attention is Turing Complete," which is a nuanced result: they show transformers
can
 be Turing complete but only under assumptions of arbitrary precision arithmetic and unbounded sequence length. Under realistic finite-precision and finite-context conditions, they are not.

The careful version of your claim is therefore:
transformers under realistic operating conditions are not Turing complete.
 That's the version that's defensible. You don't want to assert the unconditional claim because the Pérez et al. result will be thrown at you in review.

The reservoir Turing completeness claim

This comes from the RNN side of the literature. Siegelmann & Sontag (1992, 1995) is the foundational result — "On the Computational Power of Neural Nets" — showing that recurrent neural networks with rational weights are Turing complete. This is the result you're inheriting. Your reservoir is a recurrent system: r(t) depends on r(t−1), which depends on r(t−2), and so on without bound. It satisfies the conditions of Siegelmann & Sontag directly.

The one thing you need to be careful about is that Siegelmann & Sontag's result assumes unbounded runtime and unbounded precision. You'll want to briefly acknowledge this and note that the same caveats apply to your system as to any physical instantiation of a Turing-complete model — which is fine, because the same is true of any computer.

The connective argument

The formal section would go roughly:

-
Standard transformers under finite precision and finite context are bounded-computation models — cite Hahn, cite Pérez et al. with the caveat about their assumptions.

-
The reservoir component is a recurrent system satisfying the conditions of Siegelmann & Sontag, making the composite architecture Turing complete in the same sense any RNN is.

-
The mechanism by which this is achieved is the introduction of a genuine time dimension — r(t) is a state variable that evolves without bound across forward passes, which is precisely the property transformers lack and which is necessary for Turing completeness.

-
Therefore the reservoir agent is not merely a modified transformer but occupies a strictly higher computational class.

That four-step argument is tight, cites established results, and doesn't overreach. The conclusion — strictly higher computational class — is the formal grounding for the new model class designation.

What this section needs to not do

Don't claim the reservoir agent is practically more powerful than transformers on arbitrary tasks because of Turing completeness. Turing completeness is a theoretical ceiling, not a practical performance claim. A reviewer will correctly note that transformers are empirically very capable despite their theoretical limitations. The claim is purely structural:
this is a different kind of thing
, and here is the formal proof. The empirical performance claims live in your benchmark results, separately.

Keep the formal section short — half a page to a page. State the claims, cite the results, give the four-step argument. Don't over-expand it. Its job is to be there for the reviewers who want it, not to be the centerpiece of the paper.

## User

Yeah, this is good. What I would say about the Turing completeness is that it doesn't necessarily indicate anything about its actual real-world performance. All that the Turing completeness of this indicates is that it has a capacity for statefulness that allows for much more complex actions to be taken. This is something that is potentially very useful for it. The usefulness of it being able to do this stuff is definitely going to be dependent very heavily on what you actually train it to do and how you train it to do the thing. It has a capacity for statefulness that is meaningfully different from a Transformer. In a bit of a more philosophical sense, you might argue that something like this could be a lot closer to something like a living organism. Although I would very much hesitate to use the term Artificial General Intelligence or AGI, because AGI implies a certain level of intelligence. This is not necessarily a level of intelligence; it's just a quality of intelligence that, when raised to a certain level, might have characteristics that are more human-like or organism-like.

## Claude

Yes, that's a precise and important distinction. The claim isn't "this is smarter" — it's "this is a different
kind
 of thing." Capability level and capability
type
 are orthogonal axes, and conflating them is where a lot of AGI discourse goes wrong.

The organism analogy is actually scientifically grounded rather than just poetic. What distinguishes a living organism from a very sophisticated calculator is not primarily intelligence — it's that the organism has
endogenous state
 that evolves according to its own dynamics, and that state influences its behavior independently of external input. A bacterium isn't intelligent by any reasonable definition but it's alive in this sense: it has internal state that persists, that drives behavior, that isn't purely a response function over inputs. Your reservoir agent has this property in a way that a standard transformer genuinely does not. The transformer is, in this precise sense, more like a very sophisticated lookup table — extraordinarily capable, but with no inner life between queries.

The careful way to use this in the paper — if you use it at all — is exactly the framing you just used:
capacity
 language, not
achievement
 language. "Introduces a capacity for organism-like endogenous state" is a defensible structural claim. "Creates an organism-like agent" is an overclaim that will distract reviewers. The former says something true about the architecture. The latter makes an empirical claim about behavior that your benchmarks would need to directly support.

On the AGI framing — your instinct to avoid it is correct and the reasoning you gave is exactly right. AGI discourse is about level. What you're describing is about
kind
. There's actually a useful terminological distinction from the philosophy of mind literature here:
substrate independence of state
 versus
general intelligence
. Your architecture is interesting because of the former, not because it claims the latter. Keeping those separate protects the paper from being read as an AGI claim, which would invite a completely different and less productive set of reviewers and responses.

The most defensible and interesting version of the organism claim, if you want to include it, is probably a single sentence in the discussion section: something like —
while we make no claims about general intelligence, the presence of endogenous state that evolves independently of input is a property shared with living organisms and absent from standard transformer architectures, and may be a precondition for certain classes of genuinely agentic behavior.
 That's philosophically honest, scientifically grounded, and interesting without overclaiming.

## User

Yeah, can you please write out a good summary and such of everything that we're doing and a good plan for all the stuff we're going to be doing for this research engineering paper, research engineering project, and stuff like this, as a clear, clear-cut thing to resolve this conversation?

## Claude

Viewed a file, created a file, read a file
Viewed a file, created a file, read a file
That's the full document. It covers everything we worked through: the architecture, the runtime, the training pipeline, the benchmarks, the open questions with their status, the formal claims and how to structure them in the paper, and the build order.

A few things worth flagging about the document:

The build order is the most important section practically — step 3 (spectral radius experiment before any training) is easy to skip when you're moving fast but will save significant compute if a bad regime would have wasted several LoRA runs. And step 5 (benchmarks and training data together before fine-tuning) is the other one not to skip, because without it you have no way to know if the fine-tuning worked.

The formal claims section is written to be relatively close to paper-ready language — the Turing completeness four-step argument in particular is structured so you can lift it more or less directly into a theory section.
Reservoir agent plan
Document
 ·
MD 
Download

Reservoir Agent — Research & Engineering Plan

What this is

A new model class: a reservoir-augmented transformer agent with a genuine time dimension. The core idea is to take a pretrained agentic transformer (Hermes) and inject a fixed randomly-initialised reservoir into its attention mechanism at a chosen mid-depth layer. The reservoir accumulates state across every forward pass, giving the model persistent internal dynamics that evolve independently of external input. This is architecturally distinct from all existing transformer variants in one precise sense:
the model has state between forward passes
. It is not stateless. It has a time axis.

Two formal properties follow from this:

-
Genuine time dimension.
 r(t) at pass 1000 is causally downstream of every forward pass since t=0. This is not positional encoding or context length — it is actual temporal state evolution.

-
Turing completeness.
 The reservoir is a recurrent system satisfying Siegelmann & Sontag (1992/1995). Standard transformers under finite precision and finite context are not Turing complete (Hahn 2020; Pérez et al. 2021 with caveats). The composite architecture occupies a strictly higher computational class.

Neither claim is about capability level. They are claims about capability
kind
. The architecture has a capacity for endogenous state evolution that standard transformers structurally lack — a property shared with living organisms, though no claims about general intelligence are made or implied.

Why this approach

Pure reservoir computing is expensive: you are betting entirely on a random initialisation being lucky, with no fallback. Training a large reservoir model from scratch costs on the order of training any large model — the stochasticity of the reservoir just adds variance on top of that cost.

This approach solves that problem by anchoring to a pretrained model that is already known to be good. The downside case of a bad reservoir seed is now "performs like vanilla Hermes" — not "produces garbage." This makes N-seed selection tractable: run N LoRA fine-tuning jobs with different reservoir initialisations, evaluate each on a benchmark suite, keep the best. The cost per experiment is a LoRA run, not a full training run.

The three-way inheritance:

-
From transformers: pretrained world knowledge, cheap to fine-tune, strong agentic baseline

-
From reservoir computing: rich nonlinear temporal dynamics, no vanishing gradient, long-range dependency

-
Avoided: transformer statelessness, reservoir training cost

Architecture

Core design

A Hermes model with a single injection layer Lₖ at approximately 40–60% depth. At this layer, every forward pass does two things:

Read:
 the full attention output (hidden states
[seq_len × d_model]
) at Lₖ is projected via fixed random matrix W_in into the reservoir as input.

Write:
 the current reservoir state r(t) is projected via learned matrix W_out back into token-space and appended to the key/value sequence of Lₖ's attention, so upper layers can attend to it.

The reservoir update equation:

```
r(t) = tanh(W_r · r(t−1) + W_in · attn_output(t))
```

Where:

-
W_r is fixed random sparse matrix, spectral radius tuned near 0.9 (edge-of-chaos prior; actual value determined empirically)

-
W_in is fixed random projection matrix

-
W_out is
learned
 during fine-tuning — this is the primary thing fine-tuning teaches

-
r(t) lives in GPU memory as a persistent tensor, mutated in-place each pass

What is frozen vs trained

-
Lower layers (L₁ … Lₖ₋₁): frozen Hermes weights

-
W_r, W_in: fixed random, never trained

-
W_out (reservoir readout projection): learned

-
Upper layers (Lₖ₊₁ … Lₙ): LoRA fine-tuned

-
Injection layer Lₖ attention: minimally modified to accept extended key/value sequence

Key architectural property

The reservoir and context are
independent
. The context tracks content — what was said. The reservoir tracks process — what the model's attention has been doing. When context resets on a long session, the content history resets but the reservoir state persists. This gives the agent a form of memory of prior cognitive state even after context truncation, which is a feature rather than a bug.

Runtime

This architecture requires a fundamentally different execution model from standard inference. Standard transformers are stateless request-response machines. This one is not.

What changes

Standard Hermes
Reservoir Agent
Stateless request handler
Stateful always-alive process
Model loaded per request
Model + reservoir pinned in GPU memory
Context rebuilt each call
Context buffer owned by runtime, appended to
No between-pass logic
Scheduler, idle timer, confidence gate
Synchronous tool loop
Asynchronous tool execution

Runtime components

Pass scheduler:
 decides when to run a forward pass. Two paths:

-
Prompted pass:
 new input arrives, append to context, run pass

-
Unprompted pass:
 idle timer fires with no new input, run pass over context + reservoir state only

Persistent context buffer:
 owned by the runtime, never rebuilt from scratch. The model process appends to it.

Reservoir state store:
 r(t) tensor in GPU memory, mutated in-place each pass, periodically checkpointed to disk.

Output confidence gate:
 after each pass, decide whether to emit output or stay silent. Starting proxy: entropy of top-k logits. Whether this needs to be a learned head is an open question. Silent passes update reservoir state and schedule the next pass — from the outside nothing happened, but internally the agent processed its current situation.

Harness fork

The standard Hermes harness cannot be used. It needs to be forked and rewritten around the always-alive process model. The existing Hermes tool-call formatting and function-calling behaviour should be preserved where possible — regression testing against vanilla Hermes is an explicit early milestone.

Training pipeline

Approach

N-seed parallel LoRA fine-tuning:

-
Initialise N reservoir instances with different random seeds

-
Fine-tune each with LoRA on the same training data

-
Evaluate each on the benchmark suite

-
Keep the best-performing seed, discard the rest

N is probably 10–20. Each run is cheap relative to full training. The selection criterion is benchmark performance after a fixed number of training steps — not convergence, because convergence per seed is too expensive to wait for.

Training data requirements

Standard instruction-following data is insufficient. The model needs training examples that
require
 using the reservoir state to produce the correct output — otherwise it will learn to ignore the reservoir entirely since it isn't needed for standard next-token prediction.

Training data must include:

-
Multi-pass tasks with deliberate temporal gaps

-
Unresolved threads that require noticing something from N passes ago

-
Examples of correct self-initiation (agent speaks on unprompted pass)

-
Examples of correct silence (agent stays silent on unprompted pass)

The training data design and benchmark design are the same problem from two directions and should be developed together.

Benchmarks

Core requirement

Tasks where a stateless model
structurally cannot
 succeed — not just tasks where it happens to fail. The distinction matters for the paper's claims. If a strong long-context baseline can fake temporal awareness by attending back over history, the benchmark is not testing what you think it is.

Design principles

-
Relevant signal must be subtle when it first appears — not flagged or emphasised — so context-scanning alone cannot find it

-
The thing that matters must be a
change in world state
 between when information appeared and now, not just distance in context

-
Self-initiation tasks need an operationalised notion of appropriately timed output — not too much, not too little

-
Tasks should have clear pass/fail criteria that don't require human judgment to score

Candidate task types

Unresolved thread detection:
 agent is given two tasks in separate passes, completes one, second is never mentioned again. Correct behaviour: agent eventually notices and asks. Stateless model with no unprompted passes cannot do this structurally.

Elapsed time estimation:
 agent is asked to estimate how long it has been working on something, without being told. Reservoir state encodes a compressed history of pass count and attention patterns. Stateless model has no signal for this.

State change detection:
 a fact established early in a long session is later implicitly contradicted. Agent should notice the contradiction. Reservoir fingerprint of the earlier passes differs from the current one.

Appropriate silence:
 agent should not speak on every unprompted pass, only when it has something worth saying. Evaluate precision and recall of self-initiation decisions.

Open questions

Resolved or have clear answers

-
Reservoir/context independence:
 independent by design, feature not bug

-
Multi-seed selection:
 benchmark performance after fixed training steps

-
Session continuity / identity:
 not a priority for initial implementation

-
Regression against vanilla Hermes:
 explicit early milestone, build the suite first

Active open questions

Spectral radius at transformer scale.
 Standard W_r tuning is calibrated for small scalar inputs. Feeding a large attention tensor may require different scaling. Treat as an explicit early experiment — probe reservoir dynamics (variance, saturation, trajectory distinguishability) across spectral radius values before committing to a training run.

What fine-tuning teaches the lower layers.
 If lower layers are frozen, W_in learns only from gradient signal flowing down from upper layers and the readout. Whether that signal is rich enough to shape W_in well is unknown. May be an argument for feeding attention output hiddens rather than raw attention weights — hiddens carry more structured semantic signal.

Output confidence threshold design.
 Top-token entropy as a starting proxy. May need to be a learned gate head with r(t) as input rather than just logits. Needs labels for "correctly silent" passes to train — which requires the training data pipeline to be in place first.

Whether unprompted behaviour needs explicit training signal or emerges.
 Probably the most interesting empirical question in the project. Run ablations: one model trained with explicit self-initiation examples, one without. See if the behaviour emerges from reservoir dynamics alone.

Seed pre-selection proxy.
 Can reservoir dynamics metrics — stability, expressivity, trajectory distinguishability across input regimes — predict final fine-tuned performance before training? If yes, pre-select seeds before any LoRA runs, dramatically reducing compute cost. Worth one experiment.

Attention head interference.
 Some heads at Lₖ may shift their attention distribution toward reservoir nodes in ways that degrade base capabilities. Subtle degradation in tool-call formatting or instruction following. Regression suite catches this.

Formal claims for the paper

Time dimension claim (abstract / introduction)

Transformers represent time as token position — an index, not a dimension the model evolves along. The reservoir agent has a genuine time axis: r(t) evolves continuously across forward passes, causally accumulating the history of the model's attention dynamics. This is not a metaphor.

Turing completeness claim (theory section, ~half page)

-
Standard transformers under finite precision and finite context are not Turing complete — cite Hahn (2020), Pérez et al. (2021) with caveats on their assumptions.

-
The reservoir is a recurrent system satisfying Siegelmann & Sontag (1992/1995) — Turing complete in the same sense any RNN is, with the same caveats about unbounded runtime and precision.

-
The mechanism is the time dimension: Turing completeness requires unbounded computation over time, and this architecture has time where transformers do not.

-
Therefore the reservoir agent occupies a strictly higher computational class than a standard transformer.

Important:
 this section makes no claim about practical performance. Turing completeness is a theoretical ceiling. The empirical claims live in the benchmark results and are separate.

Organism analogy (discussion section, optional, single paragraph)

The reservoir introduces endogenous state that evolves independently of external input — a property shared with living organisms and absent from standard transformers. No claims about general intelligence are made. The claim is structural: this architecture has a capacity for organism-like state evolution, and this capacity may be a precondition for certain classes of genuinely agentic behaviour that are inaccessible to stateless models regardless of their capability level.

Build order

-
Model surgery
 — inject reservoir at Lₖ, verify basic Hermes behaviour survives, run reservoir with random W_out to confirm it produces output (garbage is fine at this stage). Build regression suite against vanilla Hermes.

-
Reservoir infrastructure
 — persistent state tensor in GPU memory, efficient update kernel, basic observability tooling (visualise r(t) evolution, check for saturation/explosion).

-
Spectral radius experiment
 — probe dynamics across W_r initialisations before any training. Establish what a "healthy" reservoir looks like empirically.

-
Minimal harness fork
 — two passes in sequence without reinitialising the reservoir. This is the proof-of-concept moment.

-
Benchmark and training data design
 — these are the same problem, develop together. Need at least 5 structurally clean eval tasks before running any fine-tuning.

-
N-seed fine-tuning pipeline
 — LoRA runs, parallel if hardware allows, evaluated on benchmark suite.

-
Output confidence gate
 — implement entropy proxy first, evaluate whether a learned head is necessary.

-
Paper
 — capability first (what it can do that nothing else can), mechanism second (the reservoir), formal claims third (time dimension, Turing completeness), organism framing last (discussion only).

Repository

reservoir-agent

A fork of the Hermes harness rewritten around an always-alive process model. The Hermes tool-call interface is preserved. The execution model is not.
