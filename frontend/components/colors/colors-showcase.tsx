import { cn } from "@/lib/utils"

/**
 * The palette as a fixture: every token laddered, both themes at once.
 *
 * Each panel wraps itself in the mode it shows — `.dark` for one, `.theme-light`
 * (`:root`'s twin, declared beside it in globals.css) for the other — so the
 * fixture shows both themes at once and shows the *same* two whichever mode the
 * app itself happens to be in. Without the light wrapper the "light" panel is
 * only light while the app is, which is exactly the moment nobody is looking.
 *
 * The light panel ghosts the pre-change value under every token that moved,
 * with the OKLCH deltas frozen from that change; the table below carries the
 * same numbers plus the WCAG pair each token is judged by. Token names are
 * shown as the vars themselves so a name here is the token there. The ghosts
 * and deltas are the *light* theme's — the change they record never touched a
 * dark value — so the dark panel carries its own hexes and no ghosts.
 *
 * `LIGHT_TOKENS` / `DARK_TOKENS` are the fixture's snapshot of the current
 * hexes — keep them in sync when the palette changes, the same way /markup has
 * to track the special-token vocabulary. Deliberately not read from the
 * stylesheet: a fixture that reads its own source would only ever show itself
 * that everything is fine.
 */

/** token → current hex in `:root`, for the light panel's captions. */
const LIGHT_TOKENS: Record<string, string> = {
  "--background": "#f5f1e1",
  "--card": "#fdfbf5",
  "--popover": "#fdfbf5",
  "--secondary": "#e6e2d0",
  "--muted": "#e9e6d6",
  "--accent": "#e6e2d0",
  "--border": "#b6b3aa",
  "--input": "#8c897c",
  "--sidebar": "#eee9d8",
  "--sidebar-border": "#afada4",
  "--foreground": "#4f4f58",
  "--muted-foreground": "#66645f",
  "--primary": "#4d699b",
  "--primary-foreground": "#f5f1e1",
  "--destructive": "#c43b4f",
  "--chart-1": "#bf7c00",
  "--chart-2": "#6f894e",
  "--chart-3": "#4d699b",
  "--chart-4": "#b35b79",
  "--chart-5": "#766b90",
  "--chart-approved": "#3f6bb8",
  "--chart-rejected": "#bf7c00",
  "--tier-low": "#ea9971",
  "--tier-med": "#c94f26",
  "--tier-high": "#883120",
  "--kind-train": "#3f5f96",
  "--kind-benchmark": "#4e6733",
  "--kind-chat": "#655a80",
  "--kind-tokenize": "#86582a",
  "--throbber-spark": "#14958a",
  "--throbber-base": "#2f4f7a",
}

/** token → current hex in `.dark`. Untouched by the change the ghosts record. */
const DARK_TOKENS: Record<string, string> = {
  "--background": "#181616",
  "--card": "#232221",
  "--popover": "#2a2827",
  "--secondary": "#333130",
  "--muted": "#2a2827",
  "--accent": "#333130",
  "--border": "#514d4c",
  "--input": "#68625f",
  "--sidebar": "#12120f",
  "--sidebar-border": "#494644",
  "--foreground": "#c5c9c5",
  "--muted-foreground": "#a6a69c",
  "--primary": "#8ba4b0",
  "--primary-foreground": "#181616",
  "--destructive": "#c4746e",
  "--chart-1": "#c4b28a",
  "--chart-2": "#87a987",
  "--chart-3": "#8ba4b0",
  "--chart-4": "#a292a3",
  "--chart-5": "#8992a7",
  "--chart-approved": "#6d92da",
  "--chart-rejected": "#b98a2c",
  "--tier-low": "#b1452f",
  "--tier-med": "#d97145",
  "--tier-high": "#f1a355",
  "--kind-train": "#8ba4b0",
  "--kind-benchmark": "#8fb08f",
  "--kind-chat": "#ada0ae",
  "--kind-tokenize": "#c2a184",
  "--throbber-spark": "#79cbb8",
  "--throbber-base": "#5e7480",
}

interface Ghost {
  hex: string
  d: { L: string; C: string; H: string }
}

/** token → the value it replaced, with the OKLCH deltas of that move. */
const OLD: Record<string, Ghost> = {
  "--card": { hex: "#fbf8ee", d: { L: "+.009", C: "−.005", H: "−1.5°" } },
  "--popover": { hex: "#fbf8ee", d: { L: "+.009", C: "−.005", H: "−1.5°" } },
  "--secondary": { hex: "#e9e2c4", d: { L: "+.001", C: "−.016", H: "+0.3°" } },
  "--accent": { hex: "#e9e2c4", d: { L: "+.001", C: "−.016", H: "+0.3°" } },
  "--muted": { hex: "#ece6cc", d: { L: "+.000", C: "−.013", H: "+1.8°" } },
  "--border": { hex: "#beb48a", d: { L: "−.001", C: "−.045", H: "−4.7°" } },
  "--input": { hex: "#95894f", d: { L: "+.002", C: "−.060", H: "−1.7°" } },
  "--sidebar": { hex: "#f5f1e1", d: { L: "−.024", C: "+.002", H: "−1.9°" } },
  "--sidebar-border": { hex: "#beb48a", d: { L: "−.021", C: "−.045", H: "+0.2°" } },
  "--muted-foreground": { hex: "#676550", d: { L: "+.001", C: "−.024", H: "−14.3°" } },
}

const SURFACES = [
  "--background",
  "--card",
  "--popover",
  "--secondary",
  "--muted",
  "--accent",
  "--sidebar",
  "--border",
  "--sidebar-border",
  "--input",
]
const TEXT = ["--foreground", "--muted-foreground", "--primary", "--primary-foreground", "--destructive"]
const ACCENTS = [
  "--chart-1",
  "--chart-2",
  "--chart-3",
  "--chart-4",
  "--chart-5",
  "--chart-approved",
  "--chart-rejected",
  "--tier-low",
  "--tier-med",
  "--tier-high",
  "--kind-train",
  "--kind-benchmark",
  "--kind-chat",
  "--kind-tokenize",
  "--throbber-spark",
  "--throbber-base",
]

// The frozen record of the change this fixture was built for: the OKLCH deltas
// and the WCAG pair each token is held to, before → after, both modes included
// where a pair has one.
const DELTA_ROWS = [
  ["--border", "#beb48a", "#b6b3aa", "−.001", "−.045", "−4.7°", "1.84 → 1.85 on paper"],
  ["--input", "#95894f", "#8c897c", "+.002", "−.060", "−1.7°", "3.11 → 3.10 on paper"],
  ["--sidebar-border", "#beb48a", "#afada4", "−.021", "−.045", "+0.2°", "1.85 on the sunk sidebar"],
  ["--muted-foreground", "#676550", "#66645f", "+.001", "−.024", "−14.3°", "4.54 → 4.55 on secondary"],
  ["--card / --popover", "#fbf8ee", "#fdfbf5", "+.009", "−.005", "−1.5°", "fg 7.63 → 7.83 on card"],
  ["--sidebar", "#f5f1e1", "#eee9d8", "−.024", "+.002", "−1.9°", "fg 7.16 → 6.67 on sidebar"],
  ["--secondary / --accent", "#e9e2c4", "#e6e2d0", "+.001", "−.016", "+0.3°", "fg 6.23 → 6.23"],
  ["--muted", "#ece6cc", "#e9e6d6", "+.000", "−.013", "+1.8°", "fg 6.47 → 6.46"],
] as const

function TokenRow({ token, dark }: { token: string; dark?: boolean }) {
  // The ghost is a light-theme record; in the dark panel there is nothing to
  // ghost and the caption has to read the dark hex, not the light one.
  const g = dark ? undefined : OLD[token]
  const hex = dark ? DARK_TOKENS[token] : LIGHT_TOKENS[token]
  return (
    <div className="flex items-center gap-3">
      <div
        className="h-9 w-9 shrink-0 rounded-(--radius-control-sm) border"
        style={{ backgroundColor: `var(${token})`, borderColor: "var(--border)" }}
      />
      {g && (
        <div
          className="h-9 w-9 shrink-0 rounded-(--radius-control-sm) border border-dashed"
          style={{ backgroundColor: g.hex, borderColor: "var(--muted-foreground)" }}
          title={`was ${g.hex}`}
        />
      )}
      <div className="min-w-0">
        <div className="font-mono text-xs text-foreground">{token}</div>
        <div className="font-mono text-2xs text-muted-foreground">
          {hex}
          {g && (
            <span className="pl-1" title={`ΔL ${g.d.L} · ΔC ${g.d.C} · ΔH ${g.d.H} vs ${g.hex}`}>
              · ΔL {g.d.L} ΔC {g.d.C} ΔH {g.d.H}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

function Sample({ className }: { className?: string }) {
  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-sm" style={{ color: "var(--foreground)" }}>
        The body ink: a paragraph set in the app&apos;s sans, at the size and
        leading body copy reads at. It should sit still while you read it —
        no shimmer at the letter edges, no grey-on-paper dimness.
      </p>
      <p className="text-xs" style={{ color: "var(--muted-foreground)" }}>
        The eyebrow: the same sentence at 12px, letterspaced, in the muted ink
        — the smallest regular text the app sets.
      </p>
    </div>
  )
}

function Panel({
  title,
  dark,
  bodyText,
}: {
  title: string
  dark?: boolean
  bodyText?: boolean
}) {
  return (
    <div
      className={cn("rounded-(--radius-sheet) border p-5", dark ? "dark" : "theme-light")}
      style={{ backgroundColor: "var(--background)", borderColor: "var(--border)", color: "var(--foreground)" }}
    >
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <span className="font-mono text-2xs text-muted-foreground">{dark ? "dragon" : "warm paper"}</span>
      </div>

      <h3 className="eyebrow mb-2 mt-4 text-muted-foreground">Surfaces</h3>
      <div className="grid gap-2.5">
        {SURFACES.map((t) => (
          <TokenRow key={t} token={t} dark={dark} />
        ))}
      </div>

      <h3 className="eyebrow mb-2 mt-5 text-muted-foreground">Text</h3>
      <div className="grid gap-2.5">
        {TEXT.map((t) => (
          <TokenRow key={t} token={t} dark={dark} />
        ))}
      </div>

      {bodyText && (
        <>
          <h3 className="eyebrow mb-2 mt-5 text-muted-foreground">
            Body sample
          </h3>
          <Sample />
        </>
      )}

      <h3 className="eyebrow mb-2 mt-5 text-muted-foreground">Accents</h3>
      <div className="grid gap-2.5">
        {ACCENTS.map((t) => (
          <TokenRow key={t} token={t} dark={dark} />
        ))}
      </div>
    </div>
  )
}

export function ColorsShowcase() {
  return (
    <div className="space-y-8">
      <div className="grid gap-6 md:grid-cols-2">
        <Panel title="Light — warm paper" bodyText />
        <Panel title="Dark — dragon" dark bodyText />
      </div>

      <div>
        <h2 className="text-base font-semibold text-foreground">What moved in the light theme</h2>
        <div className="mt-3 overflow-x-auto rounded-(--radius-sheet) border" style={{ borderColor: "var(--border)" }}>
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                {["token", "old", "new", "ΔL", "ΔC", "ΔH", "WCAG pair"].map((h) => (
                  <th
                    key={h}
                    className="tag px-3 py-2 text-left font-medium text-muted-foreground"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {DELTA_ROWS.map(([token, old, next, dl, dc, dh, pair]) => (
                <tr
                  key={token}
                  className="border-b last:border-b-0"
                  style={{ borderColor: "color-mix(in oklab, var(--border) 55%, transparent)" }}
                >
                  <td className="px-3 py-2 font-mono text-foreground">{token}</td>
                  <td className="px-3 py-2 font-mono text-muted-foreground line-through decoration-dotted">{old}</td>
                  <td className="px-3 py-2 font-mono text-foreground">{next}</td>
                  <td className="px-3 py-2 font-mono text-muted-foreground">{dl}</td>
                  <td className="px-3 py-2 font-mono text-muted-foreground">{dc}</td>
                  <td className="px-3 py-2 font-mono text-muted-foreground">{dh}</td>
                  <td className="px-3 py-2 text-muted-foreground">{pair}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
