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
      <PageHeader title="Colours" />
      <ColorsShowcase />
    </PageShell>
  )
}
