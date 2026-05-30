# Transcript — Transformer diagram misconceptions - Claude

_Extracted from a Claude chat export. The private 'Recents' sidebar in the raw export is intentionally excluded._


## User

This is the diagram that everybody shows you for what a Transformer looks like, and in my opinion it's like the worst diagram possible for showing what a Transformer is like. Comparing it to this diagram of an RNN that actually shows what's going on in it. Also, the Transformer is straight-up wrong in the sense that the diagram (the one everybody shows) doesn't show what's actually going on and it's also outdated.

## Claude

You're pointing at something real. The "Attention Is All You Need" figure has become almost a religious icon at this point — everyone reproduces it, few question it.

The RNN diagram is doing real work: you can see the actual nodes, the weight connections, the recurrence arrows looping back, the layer structure. Someone unfamiliar with neural networks could look at it and build a mental model of
what is happening computationally
.

The Transformer diagram by contrast is essentially a
block diagram of named operations
 — it tells you what components exist and in what order, but gives you almost no intuition about:

-
What the actual data structure is (sequences of vectors — where are they?)

-
What attention
mechanically does
 (weighted averaging over value vectors, selected by query-key dot products)

-
That the "multi-head" part means running several of these in parallel and concatenating

-
That most of the parameters actually live in the FFN, not the attention

-
The residual stream as the primary information-carrying structure (which is arguably
the
 key insight for mechanistic interpretability)

The "outdated" point is also correct. Modern decoder-only models (GPT lineage, basically everything deployed) have no encoder, no cross-attention block, no encoder-decoder split — yet this diagram is what gets shown for "what a Transformer is." Someone learning from it would be confused when they actually look at LLaMA or GPT-2's architecture.

There's also the Pre-LN vs Post-LN issue — the diagram shows Post-LN (norm after residual addition), but virtually all modern models switched to Pre-LN for training stability, which changes the topology.

What would a
good
 diagram show in your view? Something more like the residual stream picture that Anthropic and Neel Nanda's work tends to use?

## User

Can you link or embed one of those images so I can see it?

## Claude

I can't fetch or embed images from the web directly in chat. But I can do the next best thing and draw you one.
Here's the residual stream view — the mental model that's actually useful for thinking about how a modern decoder-only transformer works:
V
visualize
V
visualize
 show_widget
The key things this view makes visible that the original diagram hides:

Each token has its own
residual stream
 — a vector that persists through all layers. Attention and MLP don't
transform
 it, they each compute a result and
add
 it in. The stream is the primary information-carrying structure, and all the components are just writing to it.

The
attention lines
 between tokens show the actual cross-token communication — something completely absent from the original. Each layer's attention pattern is different, which you can see with the varying line weights.

Only the
last token's
 residual stream gets unembedded into a next-token prediction (in a decoder-only autoregressive model). The dashed lines show the other tokens' streams just... stop. The original diagram implies some symmetric encode-decode structure that doesn't exist in GPT-family models.

The
+ symbols
 represent the residual addition — this is what makes the "residual stream" framing coherent and why it's the right mental model for interpretability work.

## User

I think that one makes a lot more sense, but personally I would just at least give some level of indication of the actual structure of the neurons, which I feel like that one also is missing.  this one looks good

## Claude

Yeah, that diagram is much better — it's showing the actual
causal story
 of how "Paris" gets predicted: FFN query neurons recognize the factual pattern, attention pulls in context from earlier tokens, FFN value neurons write the answer into the stream. You can see the mechanistic interpretability framing directly in the picture.

What it's still missing, as you're pointing at, is any sense of what the residual stream
is
 as a high-dimensional vector — the neurons aren't shown as a structured space, just as abstract colored blobs. You lose the intuition that:

-
The residual stream is a single vector of ~4096 dimensions (or whatever model width)

-
Attention heads and MLP neurons are reading from and writing to
subspaces
 of that vector via low-rank projections

-
The "addition into the stream" means specific directions in that vector space are being shifted

A diagram that combined both — the token column / residual stream structure from mine, the causal circuit story from that one,
plus
 some indication that the stream is a high-dimensional vector with features occupying directions — would be significantly closer to what's actually happening.

Something like showing the residual vector as a small bar of dimensions with certain ranges lighting up when a feature is active, rather than just a rectangle. That's hard to do statically but pretty natural as an interactive — click a neuron, see which dimensions of the stream it reads from and writes to.

Want me to try building that? It'd essentially be a hybrid of the two with an interactive "probe a neuron" element.
