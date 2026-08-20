import Link from "next/link"

import { ColorsShowcase } from "@/components/colors/colors-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * A dev page, like /markup and /throbber: no data, no auth, no link in the
 * sidebar. Visit it directly to judge the palette — both themes at once, the
 * current tokens and the frozen pre-change snapshot side by side — without
 * toggling themes or hunting through the app for a surface to land on.
 */
export default function ColorsPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Colours"
        description={
          <>
            <p>
              The theme&apos;s token ladder as a fixture, so a palette change
              can be judged in one screen against the values it replaced. Both
              themes render at once — the dark one in a{" "}
              <code className="text-xs">.dark</code> subtree, not by toggling —
              and the light panel ghosts the previous value beneath every token
              that moved, with the measured deltas (OKLCH) from the change the
              fixture was built for.
            </p>
            <p>
              The numbers behind any future change live in the two audits:{" "}
              <code className="text-xs">node scripts/audit-contrast.mjs</code>{" "}
              is the WCAG gate,{" "}
              <code className="text-xs">node scripts/audit-theme.mjs</code>{" "}
              adds APCA, the OKLCH ladder and the chromatic-aberration check
              that caught the old violet-on-yellow body ink.
            </p>
            <p className="text-xs">
              Sibling dev pages:{" "}
              <Link href="/markup" className="underline underline-offset-4">
                /markup
              </Link>{" "}
              ·{" "}
              <Link href="/throbber" className="underline underline-offset-4">
                /throbber
              </Link>
            </p>
          </>
        }
      />
      <ColorsShowcase />
    </PageShell>
  )
}