import { Panel } from "@/components/ui/panel"
import { type AblationReport, formatNumber } from "@/lib/benchmark"

/**
 * What each sublayer contributes, from silencing them one at a time.
 *
 * It answers a question the composite score cannot: not how good the model is,
 * but which part of it is doing the work. A layer with no delta is a layer that
 * trained into doing nothing.
 */
export function AblationPanel({ ablation }: { ablation: AblationReport }) {
  const worst = Math.max(...ablation.units.map((u) => u.ce_delta), 0.0001)
  const ordered = [...ablation.units].sort((a, b) => b.ce_delta - a.ce_delta)

  return (
    <Panel
      title="Sublayer contribution"
      description={`${ablation.layers} layers · ${ablation.window_tokens} tokens`}
    >
      {ablation.learned_nats < 0.1 ? (
        <p className="mb-4 text-xs text-muted-foreground">
          The intact model sits {formatNumber(ablation.learned_nats, 3)} nats
          below a uniform guess, so the table below is measuring noise.
        </p>
      ) : null}

      <dl className="flex flex-col gap-2.5">
        {ordered.map((unit) => (
          <div key={unit.unit} className="flex items-center gap-3">
            <dt className="w-40 shrink-0 truncate text-xs">
              <span className="text-muted-foreground">L{unit.layer}</span>{" "}
              {unit.kind}
            </dt>
            <dd className="flex min-w-0 flex-1 items-center gap-3">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border/60">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${(unit.ce_delta / worst) * 100}%` }}
                />
              </div>
              <span className="w-14 shrink-0 text-right text-xs tabular-nums">
                {formatNumber(unit.ce_delta, 3)}
              </span>
            </dd>
          </div>
        ))}
      </dl>

      <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
        Cross-entropy added by silencing one sublayer, holding everything else
        fixed. Intact: {formatNumber(ablation.intact_ce, 3)} against{" "}
        {formatNumber(ablation.chance_ce, 3)} for a uniform guess.
      </p>
    </Panel>
  )
}
