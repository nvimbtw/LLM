# Live throbber

`components/ui/throbber.tsx` selects the classic trellis or optional live art
using the account's **Use live throbber** Appearance setting. It defaults to
off; `/live-throbber on|off` controls the same preference. The existing settings
JSON stores it without a backend schema change. The wrapper changes only the
animation: the clock, token count, thinking label and tool label stay with their
existing owners. Both marks occupy the same 27 × 25 CSS pixel slot.

## Rendering boundary

The live mark is a clean 25 pixel circle containing procedural Random Art.
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

The renderer samples the latest accumulated seed at most ten times a second,
then exponentially approaches the new coefficients over a 420 ms time constant.
Every token affects the hash even when intermediate targets are coalesced.
Keeping the grammar fixed during a turn lets coefficients morph continuously;
retargeting does not restart an easing clock or build a backlog. Bursts therefore
stay bounded. Identical ordered inputs produce identical seeds and target
expressions; animation timing and GPU rounding can produce different pixels.

Frames are capped at 24 fps, backing resolution at twice CSS size and 192 pixels
per side. Each canvas allows only one unfinished GPU submission; slow devices
skip stale frames instead of accumulating a queue. The default footprint is much smaller. Activity modes change a gently
blended ambient pace; no fake token IDs are invented during a wait. Idle settles
and stops its loop. Hidden documents and offscreen canvases suspend drawing.
Unmount releases the surface/buffer, subscriptions, timers and observers.

Reduced motion removes ambient movement and interpolation, updating the still
image at most twice a second as real tokens arrive. Unsupported WebGPU, insecure
contexts, failed initialization/shader validation, or a device/render failure
show the existing trellis. Startup has an eight-second deadline and cancelled
initializations release their resources when they resolve. It stays visible until the first successful GPU frame
and is unmounted while live art is displayed. A device failure is latched for
the page lifetime; reloading retries initialization.

## Showcase

`/throbber` includes an enlarged 192 pixel canvas, the actual status-row mark,
classic/live comparison, all eight modes, pause/resume, replay, a 256-token burst
and ID/text source controls. IDs are explicitly synthetic fixtures. Replay resets
the same seed; pauses stop token input while the active mode keeps its ambient
motion. The original multi-state gallery lives in a separate **Classic showcase** tab,
so its SVG animations and counter timer are unmounted while reviewing live art. Those panels
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
bun test lib/live-throbber/stream.test.ts
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
lint, five stream tests (including a 100,000-token burst) and the webpack
production build pass. A production browser run on Intel Gen12LP verified
192/50/50 backing sizes at DPR 2, 300 submissions across three canvases over
five seconds, no submissions after idle settles or while offscreen, and two
updates per second per canvas under reduced motion. Device destruction selected
the classic fallback. Mocked account requests verified both setting values reach
the existing settings PUT. Dark/light and narrow viewport screenshots are in
[FEAT-1](https://discord.com/channels/1538659898383011961/1559193308117344298).
These frame counts are a bounded desktop observation, not a mobile benchmark.

## Review plan and risks

1. Confirm off/default uses the original animation and on persists via existing
   account settings; a missing setting in an old account gets the off default.
2. Confirm ordered IDs, zero, invalid IDs, Unicode fallback and resets produce
   stable seeds. A large burst must use constant stream storage.
3. Watch normal streaming, pauses, bursts, thinking/tool/idle transitions and
   replay. Check the small mark as well as the enlarged fixture: a rich large
   image can become noise at 25 pixels.
4. Check both themes, a narrow viewport, reduced motion, unavailable WebGPU,
   device failure and repeated mount/unmount. Confirm fallback and status text
   remain visible and no animation shifts the row.
5. Build the WASM with zero warnings, check asset consistency, typecheck, lint
   and run the Next production build. Capture screenshots for Discord.

WebGPU requires browser/driver support and a secure origin (localhost qualifies).
The legacy fallback is necessary on unsupported or restricted devices. Responsive emulation does not establish performance on a physical phone. These images do not represent model confidence, semantics or progress.
Changing a model/tokenizer can change its IDs, and the 32-bit hash can collide;
neither is a correctness problem for a decorative indicator.
