# Live throbber

`components/ui/throbber.tsx` selects the classic trellis or optional live art
using the account's **Use live throbber** Appearance setting. It defaults to
off; `/live-throbber on|off` controls the same preference. The existing settings
JSON stores it without a backend schema change. The wrapper owns the mark;
the clock, token count, thinking label and tool label stay with their
existing owners. Live art uses one fixed **32 × 32 CSS pixel** footprint across
the app, set by `--size-live-throbber` in `app/globals.css`. Its fluid detail
needs more room than the classic trellis, which stays at 28 × 25 pixels.
Loading and fallback centre the original trellis in the live slot, so the row
does not shift when the canvas appears or fails.

## Rendering boundary

The live mark is a clean 32 pixel circle containing procedural Random Art.
`frontend/gpu/` is a browser renderer, not a backend crate: Rust **wgpu 28**
compiles to WASM, uses browser WebGPU and evaluates WGSL on the GPU. It has no
service, database, model or network dependency beyond fetching its static WASM.
The bundle is loaded when live art is requested. One shared low-power device
serves the canvases, each of which owns its surface, pipeline and uniform buffer.

The design draws on the seeded expression grammar in
[Hash Visualization](https://users.ece.cmu.edu/~adrian/projects/validation/validation.pdf),
section 3. This is an animation adaptation, not an implementation of its
security requirements. A full depth-four binary tree has 31 nodes. Leaves are
bounded coordinate functions; internal nodes use sine, cosine, multiplication
or mixing. The turn's identity seeds the topology. Ordered tokens seed its
continuous coefficients. Four existing active theme colours supply the palette.
The shader uses constant node references instead of a dynamically indexed
per-pixel array, so intermediate values can stay in registers. Transparent
corners exit before the expression, and inactive ripple/scan math is skipped.

## Token contract

The frontend accepts the optional unsigned 32-bit integer `token_id` on each
`{ type: "token", token: string, token_id?: number }` chat frame. Token zero is
valid. Negative, fractional, non-finite, string or out-of-range IDs fall back to
the text delta. `lib/chat-socket.ts` validates this field and passes it as the
optional second argument of `onToken`; it never alters the decoded text.

`useChat` resets an `ArtStream` when a new turn starts and feeds every streamed
token, including reasoning and successive tool rounds. Cancellation or completion
does not erase the image. No backend/LLM changes ship here: until a producer
actually supplies `token_id`, live chat uses **text fallback**. This cannot
recover tokenizer IDs and is not presented as doing so. Prompt text participates
in the initial seed, but prompt IDs are unavailable in this contract.

`ArtStream` folds each ID's four bytes into an order-sensitive 32-bit rolling
hash. Text uses a separate domain and UTF-16 units; boundaries are included.
Only the latest snapshot and subscribers are retained. No token history, queued
images or expanded text are stored by the renderer. The seed is decorative,
noncryptographic and never sent or persisted by this feature.

## Motion and lifecycle

The renderer samples the latest accumulated seed at most **1.25 times a second**
(one target every 800 ms), then interpolates coefficients **linearly over
800 ms**. Equal time steps advance a segment equally, avoiding an exponential
slowdown followed by a kick at each new target. Retargeting starts from the
current interpolated values, so positions remain continuous. The expression is
nonlinear, so this is constant coefficient velocity, not a claim that every
pixel changes at the same rate.
Every token affects the hash even when intermediate targets are coalesced.
Keeping the grammar fixed during a turn lets coefficients morph continuously.
Individual tokens do not restart a transition or build a backlog. Bursts therefore
stay bounded. At 240 tokens/s, roughly one target represents 192 arrivals; a
1 token/s stream can use each arrival. Sampling is independent of inference
speed and always takes the latest seed: skipped targets never catch up later.
A reset or a new turn bypasses the previous turn’s sampling deadline. Identical
ordered inputs produce identical seeds and target
expressions; animation timing and GPU rounding can produce different pixels.

Active drawing targets **60 fps**, paced directly by animation frames with a
deadline that tolerates timestamp rounding and differing display refresh rates.
Backing resolution remains capped at twice CSS size and 192 pixels per side.
Each canvas allows at most **two unfinished GPU submissions**, overlapping work
with the browser's asynchronous completion notification; slow devices
skip stale frames instead of accumulating a queue. The default footprint is much
smaller. Activity modes ease between distinct internal transformations over 600 ms;
no fake token IDs are invented during a wait. The circle itself stays fixed.

| Mode | Ambient motion without output tokens |
| --- | --- |
| Loading | Breathing zoom; its rate slows logarithmically with wait time, with a positive floor. |
| Thinking | A slow wandering field and slight rotation. |
| Generating | Minimal drift underneath token-driven coefficient changes. |
| Tool | A small breath and gentle rock. |
| Training | Flowing coordinate ripples and a slow turn. |
| Tokenizing | A smooth scanning distortion. |
| Benchmarking | Paired opposing scans. |
| Idle | The transform settles to neutral and stops. |

These are activity signatures, not measurements of job progress. Idle settles
and stops its loop. Hidden documents and canvases with no visible intersection
area suspend drawing and cancel their frame timers.
Unmount releases the surface/buffer, subscriptions, timers and observers.

Reduced motion removes ambient movement and interpolation. Drawing is limited
to twice a second; new visual targets still obey the 1.25/s sampling cap.
Unsupported WebGPU, insecure
contexts, failed initialization/shader validation, or a device/render failure
show the existing trellis. Startup has an eight-second deadline and cancelled
initializations release their resources when they resolve. The trellis stays
visible until the first successful GPU frame
and is unmounted while live art is displayed. A device failure is latched for
the page lifetime; reloading retries initialization.

## Showcase

`/throbber` includes an enlarged 192 pixel canvas, the actual status-row mark,
classic/live comparison, all eight modes, pause/resume, replay, a 256-token burst
and ID/text source controls. Slow/steady/fast fixtures send 1/24/240 tokens/s,
so inference speed can be compared against the fixed visual cadence. IDs are
explicitly synthetic fixtures. Replay resets
the same seed; pauses stop token input while the active mode keeps its ambient
motion. The original multi-state gallery lives in a separate **Classic showcase**
tab, so its SVG animations and counter timer are unmounted while reviewing live art. Those panels
still exercise the shared `Throbber` component and follow the saved preference.

## Build

Generated JS, declarations and `public/live-throbber/renderer_bg.wasm` are checked
in so normal Next.js and Docker builds need no Rust installation. `bun run build`
checks their SHA-256 manifest before bundling. To modify Rust or WGSL:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.125 --locked
cd frontend
bun run build:throbber
bun run check:throbber
bun test lib/live-throbber/stream.test.ts lib/live-throbber/sampler.test.ts
bun test lib/live-throbber/frame-clock.test.ts lib/live-throbber/morph.test.ts
bun run typecheck
bun run lint
```

The packager must match the exact wasm-bindgen version in `Cargo.lock`. Its
generated loader is rewritten to the public WASM path because Next resolves
`new URL(..., import.meta.url)` even when an explicit init path is provided.
Only generated glue is excluded from lint; application TypeScript remains
strictly typed. Commit the generated assets and manifest with source changes.

## Verification (2026-10-12)

The release WASM build has zero warnings; asset validation, frontend typecheck,
lint, nine stream/sampler/motion tests (including a 100,000-token burst and
1–1,000 tokens/s sampling) and the webpack
production build pass. A production browser run on Intel Gen12LP verified
192/50/50 backing sizes at DPR 2, at most 300 submissions across three canvases over
five seconds, no submissions after idle settles or while offscreen, and two
updates per second per canvas under reduced motion. Slow/fast fixture rates were
verified; all four non-output modes kept animating with the input paused. Device destruction
selected the classic fallback when each canvas was visible. Mocked account requests verified both setting values reach
the existing settings PUT. Dark/light and narrow viewport screenshots are in
[FEAT-1](https://discord.com/channels/1538659898383011961/1559193308117344298).
These frame counts are a bounded desktop observation, not a mobile benchmark.

## Performance verification (2026-10-13)

The WASM release rebuild has zero warnings. Asset validation, 15 stream,
sampler, frame-clock and linear-morph tests, typecheck, lint and the webpack
production build pass. Clock tests cover 59.94–240 Hz and missed frames.

On Intel Gen12LP, a batched GPU timestamp test at 64 × 64 measured median
shader time of 0.0195 ms before and 0.0036 ms after optimization (about 5× faster).
Readbacks matched the original shader exactly across eight poses and two seeds.
These isolate shader cost, not presentation or inference contention.
Live uniform captures showed constant coefficient slopes within each segment.
Foreground browser samples delivered 43–56 fps while animation callbacks were
available at 45–56 fps; **60 fps is a target, not a verified lock under load**.
Per-canvas pending submissions stayed at two or fewer. Idle and offscreen
drawing stopped, and visibility restored the 32 pixel mark without resizing.
Reduced-motion drawing remained at two frames per second.

## Review plan and risks

1. Confirm off/default uses the original animation and on persists via existing
   account settings; a missing setting in an old account gets the off default.
2. Confirm ordered IDs, zero, invalid IDs, Unicode fallback and resets produce
   stable seeds. A large burst must use constant stream storage.
3. Watch normal streaming, pauses, bursts, thinking/tool/idle transitions and
   replay. Check the small mark as well as the enlarged fixture: a rich large
   image can become noise at the production size.
4. Check both themes, a narrow viewport, reduced motion, unavailable WebGPU,
   device failure and repeated mount/unmount. Confirm fallback and status text
   remain visible and no animation shifts the row.
5. Build the WASM with zero warnings, check asset consistency, typecheck, lint
   and run the Next production build. Capture screenshots for Discord.

WebGPU requires browser/driver support and a secure origin (localhost qualifies).
The legacy fallback is necessary on unsupported or restricted devices.
Responsive emulation does not establish performance on a physical phone. These
images do not represent model confidence, semantics or progress.
Changing a model/tokenizer can change its IDs, and the 32-bit hash can collide;
neither is a correctness problem for a decorative indicator.
