# Architecture diagram

`components/training/architecture-flow.tsx` owns the shared explanatory diagram.
Callers explicitly choose a view: training in both fresh and inherited training
composers, generation in model details. There is no view switch. Model training
family affects supervision labels, never the model page's generation behavior.

## Meaning

Both views show embeddings, a representative transformer block, final
normalization and an output projection tied to embedding weights. The flat stack
shows up to four sheets with the exact configured layer count; it does not allocate
one element per layer, attention head or context token. The two sublayers show
pre-normalization, causal attention with RoPE, a GELU feed-forward network and
residual additions. Operation miniatures illustrate embedding expansion, causal
attention, feed-forward transformation, normalization and output scores.

Generation continues through sampling. The selected token travels back into the
prompt strip and remains there for the next pass. After six visible tokens the
strip scrolls, preserving token identity and position across the loop boundary.
The ellipsis denotes an excerpt, not truncation of the model's actual context.
Caching is omitted: this is a conceptual sequence illustration, not a claim about
runtime KV-cache execution or live attention values.

Training combines output with separate targets (assistant-token targets for SFT),
then computes gradients and optimizer updates. Dashed branches distribute updates
to embeddings, transformer layers, normalization and output before a new batch
replaces the input strip. KTO uses feedback/preference labels and RL uses
rewards/policy-objective labels; these are schematic summaries of supervision,
not full diagrams of their training algorithms.

## Rendering and lifecycle

`lib/architecture-flow.ts` owns phase calculation, bounded token windows, operation
ports and route interpolation. The same gently rounded route draws each
connection and locates its traveling mark. Travel and processing use eased timing. Each operation reaches
its final state before its phase ends and retains that result until the next input
resets it; emphasis fades independently on the same clock. The longer generation
return and weight-update stages have time to finish their full paths. Responsive
layouts use the diagram container's width rather than the viewport width, so a sidebar cannot squeeze the horizontal chart into an
unreadable column. Existing theme roles supply surfaces, strokes and one accent.
Invalid draft dimensions remain unavailable instead of producing real counts.

`use-architecture-playback.ts` owns one clock across responsive layouts, using
the browser’s animation-frame cadence to avoid uneven throttling on 60 Hz
displays. Pause freezes the current phase and resumes from it. Offscreen diagrams and hidden tabs suspend their clock without catching
up on return. Unmount cancels the frame and releases visibility observers.
A resize observer mounts only the layout that fits the figure. Unchanged operation
blocks retain their render while the active miniature animates. Reduced motion
renders a complete static schematic and removes the play control.
SVG titles and descriptions convey the flow without announcing every animation
frame to screen readers. The drawing is labelled as an illustration.

## Verification

`lib/architecture-flow.test.ts` covers generation continuity across append and
scroll boundaries, training batch ordering, completed-operation retention, easing
boundaries, route bends, responsive view semantics and invalid dimensions. Browser checks cover
pause/resume, reduced motion, visibility, both page contexts and narrow layouts.
Run the frontend verification commands in the [module map](README.md#verification).
