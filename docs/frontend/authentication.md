# Authentication pages

`/login` and `/signup` share an open, single-column composition outside the
signed-in chrome, with no header or sidebar. The Slate home link belongs to the
form, above its restrained serif heading. Fields use `Input variant="ruled"`:
a bottom rule, no box or fill, and a thicker, tinted rule on keyboard focus.
A separate rule introduces the alternate account link. Theme comes from the
existing provider's saved preference, falling back to the system preference;
these pages have no separate theme selector.

`AuthScene` places a very muted, evenly spaced ASCII character texture across
the page, interspersed with sparse assistant/model vocabulary. `AuthTexture`
renders a deterministic SVG pattern with fixed-size
glyphs, so larger viewports repeat the tile rather than enlarge its characters.
Mouse movement reveals a soft primary-colour halo and a smaller, brighter blue
core through concentric radial masks. A soft clearing in all layers keeps the
form readable.
The effect's stylesheet is imported by `AuthScene` so its masks and opacity
rules load with the interactive layers.
The artwork is decorative, ignores pointer events and is hidden from assistive
technology. `usePointerLight` updates CSS properties through refs, without
rerendering the form or rebuilding SVG geometry. Its trail retains at most eight
pointer positions and fades each over 800ms. The halo and core fade after 240ms
without mouse movement, using a smooth 900ms opacity transition; the frame loop stops when trail positions expire
and is cancelled on unmount. Touch and reduced-motion modes
keep the static characters without the moving highlight.

The pages and `AuthShell` render on the server; `AuthScene` owns pointer handling,
while the forms use the existing login/signup hooks. `AuthField` keeps password
visibility in the keyboard tab order and connects hints with `aria-describedby`.
Signup creates an account and redirects to login; it does not create a workspace.
