import { Panel } from "@/components/ui/panel"
import type { PracticeReport } from "@/lib/practice"

export interface PracticePanelProps {
  report: PracticeReport
}

const STATUS = {
  locked: "Prerequisites first",
  practice: "Practising",
  mastered: "Mastered",
  needs_sft: "Needs SFT examples",
}
/** Formats measured practice accuracy without replacing absence with zero. */
const percent = (value: number | null | undefined) =>
  value == null ? "—" : `${(value * 100).toFixed(0)}%`

/** Presents per-skill progression and fixed before/after evidence without changing practice or grading. */
export function PracticePanel({ report }: PracticePanelProps) {
  if (!report.mastery) return null
  return (
    <Panel
      variant="open"
      title="Math progression"
      description={
        report.practice_phase ??
        "Mastery is checked separately from training accuracy."
      }
    >
      <div className="space-y-5">
        <p className="text-sm">
          Fixed test: {percent(report.test_before?.accuracy)} before →{" "}
          {percent(report.test_after?.accuracy)} after. The same questions and
          decoding settings are used at both ends; this test never changes the
          lessons.
        </p>
        <p className="text-xs text-muted-foreground">
          Validation uses fresh seeds and held-out wording. Elementary facts can
          recur: mastery of these facts does not establish general reasoning.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted-foreground">
              <tr>
                <th className="py-2">Skill</th>
                <th>Status</th>
                <th>Validation</th>
                <th>Invalid / capped</th>
                <th>Checks</th>
                <th>Test before / after</th>
              </tr>
            </thead>
            <tbody>
              {report.mastery.map((skill) => (
                <tr key={skill.task_type} className="align-top">
                  <td className="py-2 pr-3">{skill.label}</td>
                  <td className="pr-3">{STATUS[skill.status]}</td>
                  <td className="pr-3 tabular-nums">
                    {percent(skill.accuracy)}
                    {skill.samples > 0 ? ` (${skill.samples})` : ""}
                  </td>
                  <td className="pr-3 tabular-nums">
                    {skill.samples
                      ? `${skill.invalid_answers ?? 0} / ${skill.truncated ?? 0}`
                      : "—"}
                  </td>
                  <td>{skill.checks}</td>
                  <td className="tabular-nums">
                    {percent(
                      report.test_before?.tasks.find(
                        (t) => t.task_type === skill.task_type
                      )?.accuracy
                    )}{" "}
                    /{" "}
                    {percent(
                      report.test_after?.tasks.find(
                        (t) => t.task_type === skill.task_type
                      )?.accuracy
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted-foreground">
          Invalid / capped counts answers that could not be parsed and replies
          that reached the token or context limit in the latest validation.
          These counts can overlap.
        </p>
        {report.mastery.some((s) => s.status === "needs_sft") && (
          <p className="text-sm">
            Some skills rarely produce correct answers. Generate demonstrations
            for those skills and fine-tune before another practice run. No SFT
            job is started automatically.
          </p>
        )}
        {!!report.curriculum_events?.length && (
          <ul className="space-y-1 text-xs text-muted-foreground">
            {report.curriculum_events.map((event, i) => (
              <li key={i}>
                Iteration {event.step}: {event.task_type} — {event.action},{" "}
                {event.correct}/{event.samples} correct.
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  )
}
