# The marker grammar is enforced while decoding

`llm::markup::OpenBlocks` is a pushdown automaton over the reserved marker block.
`infer::stream_generation` masks every illegal marker's logit to `-inf` *before*
sampling, so a malformed reply is not merely unlikely — it is unrepresentable.

The model cannot do this on its own and is not close. The markup suite scores
`closer_matches_opener` at **52.2%** against a 50% chance baseline and
`nesting_order` at 63.3%, and served replies showed exactly that shape: `<|par|>`
closed by `<|/item|>`, blocks left hanging, and a whole `<|/user|><|assistant|>`
turn boundary invented mid-reply. The rules are the four the suite tests plus two
containment ones — `<|item|>` only directly inside `<|list|>`, and only `<|item|>`
directly inside it — with `MAX_DEPTH` 6 bounding a model that opens and never closes.

- **Ordinary text is legal in every state, unconditionally.** That is what
  guarantees masking cannot empty the candidate set, and it is asserted rather
  than assumed (`ordinary_tokens_are_always_legal`).
- **The end-of-turn markers are gated on an empty stack.** This is also what makes
  the model *close* its blocks: it cannot reach `<|/assistant|>` any other way.
- **The table fails closed.** A marker in `SPECIAL_TOKENS` with no rule in `PAIRS`
  or `STANDALONE` is forbidden, so adding one cannot silently make it emittable.
- **The token cap is the one stop the grammar cannot refuse**, so the loop reserves
  `depth() + 1` tokens against `max_tokens` and drains the open blocks when it
  stops. The `+ 1` is because the token about to be sampled may itself open a
  block; without it a reply that opens on its last token overruns the cap by one.
- **`generate_response_streaming` passes `None`** — a bare prompt completion is raw
  continuation, not an assistant turn, and has no scaffold to keep balanced.

**It must stay out of `eval`.** The markup suite exists to measure whether the model
has *learned* the grammar; generating its structure stage under the constraint would
measure the constraint and score every model alike. `eval::generate::continue_tokens`
therefore does not use it, and the suite's score is still the number to watch — this
changes what the model *emits*, not what it knows. Expect the served output to look
dramatically better while the benchmark does not move at all; that is correct, and
the gap between them is the honest measure of how much work the constraint is doing.
