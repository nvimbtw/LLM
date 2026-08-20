import { cn } from "@/lib/utils"

/**
 * The palette as a fixture: every token laddered, both themes at once.
 *
 * The dark panel renders inside a `.dark` subtree (the app's custom variant is
 * `&:is(.dark *)`, so one wrapper flips its whole contents to the dark tokens)
 * — the fixture shows both themes without toggling.
 *
 * The light panel ghosts the pre-change value under every token that moved,
 * with the OKLCH deltas frozen from that change; the table below carries the
 * same numbers plus the WCAG pair each token is judged by. Token names are
 * shown as the vars themselves so a name here is the token there.
 *
 * `TOKENS` is the fixture's snapshot of the current hexes — keep it in sync
 * when the palette changes, the same way /markup has to track the special-token
 * vocabulary. Deliberately not read from the stylesheet: a fixture that reads
 * its own source would only ever show itself that everything is fine.
 */

/** token → current hex, for the swatches' captions. */
const TOKENS: Record<string, string> = {
  "--background": "#f5f1e1",
  "--card": "#fbf8ee",
  "--popover": "#fbf8ee",
  "--secondary": "#e9e2c4",
  "--muted": "#ece6cc",
  "--accent": "#e9e2c4",
  "--border": "#beb48a",
  "--input": "#95894f",
  "--foreground": "#4f4f58",
  "--muted-foreground": "#676550",
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

interface Ghost {
  hex: string
  d: { L: string; C: string; H: string }
}

/** token → the value it replaced, with the OKLCH deltas of that move. */
const OLD: Record<string, Ghost> = {
  "--background": { hex: "#f7f4e4", d: { L: "−.008", C: "+.000", H: "−2.8°" } },
  "--card": { hex: "#fdfbf1", d: { L: "−.008", C: "+.000", H: "−4.5°" } },
  "--popover": { hex: "#fdfbf1", d: { L: "−.008", C: "+.000", H: "−4.5°" } },
  "--border": { hex: "#c1b88c", d: { L: "−.011", C: "−.002", H: "−1.6°" } },
  "--input": { hex: "#988d58", d: { L: "−.013", C: "+.006", H: "0.0°" } },
  "--foreground": { hex: "#545464", d: { L: "−.021", C: "−.011", H: "+0.5°" } },
  "--primary-foreground": { hex: "#f7f4e4", d: { L: "−.008", C: "+.000", H: "−2.8°" } },
  "--destructive": { hex: "#c63e52", d: { L: "−.007", C: "+.001", H: "+0.4°" } },
  "--chart-1": { hex: "#c37e00", d: { L: "−.009", C: "−.002", H: "+0.5°" } },
  "--chart-rejected": { hex: "#c37e00", d: { L: "−.009", C: "−.002", H: "+0.5°" } },
  "--kind-tokenize": { hex: "#8a5a2b", d: { L: "−.009", C: "−.002", H: "+0.7°" } },
}

const SURFACES = ["--background", "--card", "--popover", "--secondary", "--muted", "--accent", "--border", "--input"]
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
  ["--background", "#f7f4e4", "#f5f1e1", "−.008", "+.000", "−2.8°", "Y .901 → .878"],
  ["--foreground", "#545464", "#4f4f58", "−.021", "−.011", "+0.5°", "6.72 → 7.16 on paper"],
  ["--card / --popover", "#fdfbf1", "#fbf8ee", "−.008", "+.000", "−4.5°", "fg 7.16 → 7.63 on card"],
  ["--primary-foreground", "#f7f4e4", "#f5f1e1", "−.008", "+.000", "−2.8°", "4.99 → 4.87 on primary"],
  ["--destructive", "#c63e52", "#c43b4f", "−.007", "+.001", "+0.4°", "4.51 → 4.54 on paper"],
  ["--border", "#c1b88c", "#beb48a", "−.011", "−.002", "−1.6°", "1.81 → 1.84 on paper"],
  ["--input", "#988d58", "#95894f", "−.013", "+.006", "0.0°", "3.03 → 3.11 on paper"],
  ["--chart-1 / --chart-rejected", "#c37e00", "#bf7c00", "−.009", "−.002", "+0.5°", "3.02 → 3.04 on paper"],
  ["--kind-tokenize", "#8a5a2b", "#86582a", "−.009", "−.002", "+0.7°", "4.53 → 4.61 on its tint"],
] as const

function TokenRow({ token }: { token: string }) {
  const g = OLD[token]
  return (
    <div className="flex items-center gap-3">
      <div
        className="h-9 w-9 shrink-0 rounded-md border"
        style={{ backgroundColor: `var(${token})`, borderColor: "var(--border)" }}
      />
      {g && (
        <div
          className="h-9 w-9 shrink-0 rounded-md border border-dashed"
          style={{ backgroundColor: g.hex, borderColor: "var(--muted-foreground)" }}
          title={`was ${g.hex}`}
        />
      )}
      <div className="min-w-0">
        <div className="font-mono text-xs text-foreground">{token}</div>
        <div className="font-mono text-[10px] text-muted-foreground">
          {TOKENS[token]}
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
      className={cn("rounded-xl border p-5", dark && "dark")}
      style={{ backgroundColor: "var(--background)", borderColor: "var(--border)", color: "var(--foreground)" }}
    >
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <span className="font-mono text-[10px] text-muted-foreground">{dark ? "dragon" : "warm paper"}</span>
      </div>

      <h3 className="mb-2 mt-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Surfaces</h3>
      <div className="grid gap-2.5">
        {SURFACES.map((t) => (
          <TokenRow key={t} token={t} />
        ))}
      </div>

      <h3 className="mb-2 mt-5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Text</h3>
      <div className="grid gap-2.5">
        {TEXT.map((t) => (
          <TokenRow key={t} token={t} />
        ))}
      </div>

      {bodyText && (
        <>
          <h3 className="mb-2 mt-5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            Body sample
          </h3>
          <Sample />
        </>
      )}

      <h3 className="mb-2 mt-5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Accents</h3>
      <div className="grid gap-2.5">
        {ACCENTS.map((t) => (
          <TokenRow key={t} token={t} />
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
        <p className="mt-1 text-sm text-muted-foreground">
          OKLCH deltas and the WCAG pair each token is held to, before → after, measured
          at the change this fixture documents. Every move was a lightness or chroma move —
          hue nearly untouched, which is what keeps the scheme Kanagawa.
        </p>
        <div className="mt-3 overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                {["token", "old", "new", "ΔL", "ΔC", "ΔH", "WCAG pair"].map((h) => (
                  <th
                    key={h}
                    className="px-3 py-2 text-left font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground"
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