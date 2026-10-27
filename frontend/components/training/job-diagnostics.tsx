"use client"

import type {
  DiagPoint,
  ModelDiagnosticOccurrence,
  ModelDiagPoint,
  TrainingDiagnosticPoint,
} from "@/lib/training"
import { Panel } from "@/components/ui/panel"
import { DiagnosticsPanel } from "./diagnostics-panel"

/** Checks bounded owner measurements before rendering interpreted custom-model telemetry. */
function ownerMeasurements(value: unknown): value is Record<string, number> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const entries = Object.entries(value)
  return (
    entries.length <= 4096 &&
    entries.every(
      ([name, measurement]) =>
        name.length > 0 &&
        name.length <= 512 &&
        typeof measurement === "number" &&
        Number.isFinite(measurement) &&
        measurement >= 0
    )
  )
}

/** Checks one actual occurrence's bounded identity and tensor summaries without inferring layer indices. */
function occurrence(value: ModelDiagnosticOccurrence): boolean {
  return (
    typeof value?.instance_path === "string" &&
    value.instance_path.length <= 512 &&
    Array.isArray(value.call_path) &&
    value.call_path.length <= 64 &&
    value.call_path.every(
      (index) => Number.isSafeInteger(index) && index >= 0
    ) &&
    Array.isArray(value.shapes) &&
    value.shapes.length <= 32 &&
    value.shapes.every(
      (shape) =>
        Array.isArray(shape) &&
        shape.length <= 8 &&
        shape.every((size) => Number.isSafeInteger(size) && size >= 0)
    ) &&
    Array.isArray(value.dtypes) &&
    value.dtypes.length === value.shapes.length &&
    value.dtypes.every(
      (dtype) => typeof dtype === "string" && dtype.length <= 64
    ) &&
    Array.isArray(value.output_rms) &&
    value.output_rms.length <= value.shapes.length &&
    value.output_rms.every(
      (rms) => typeof rms === "number" && Number.isFinite(rms) && rms >= 0
    )
  )
}

/** Recognizes the custom-model wire variant, preserving omitted measurements and rejecting malformed summaries. */
function modelDiagnostics(
  point: TrainingDiagnosticPoint
): point is ModelDiagPoint {
  return (
    "parameter_grad_norm" in point &&
    ownerMeasurements(point.parameter_grad_norm) &&
    (point.parameter_update_ratio === undefined ||
      ownerMeasurements(point.parameter_update_ratio)) &&
    (point.occurrences === undefined ||
      (Array.isArray(point.occurrences) &&
        point.occurrences.length <= 128 &&
        point.occurrences.every(occurrence))) &&
    (point.step === undefined ||
      (Number.isSafeInteger(point.step) && point.step >= 0))
  )
}

/** Requires the legacy arrays consumed by Transformer charts; sparse/model reports never enter those charts. */
function legacyDiagnostics(point: TrainingDiagnosticPoint): point is DiagPoint {
  return (
    !("parameter_grad_norm" in point) &&
    [
      point.grad_layers,
      point.update_ratio_layers,
      point.act_rms_layers,
      point.attn_entropy,
      point.act_hist,
      point.emb_pca,
      point.emb_pca3,
    ].every(Array.isArray)
  )
}

/** Formats a measured finite scalar, keeping an absent measurement distinct from an actual zero. */
function measurement(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value)
    ? "Unavailable"
    : value.toPrecision(4)
}

/** Presents bounded latest owner measurements, counting shared parameters by their reported owner identity. */
function ParameterOwners({ point }: { point: ModelDiagPoint }) {
  const owners = [
    ...new Set([
      ...Object.keys(point.parameter_grad_norm),
      ...Object.keys(point.parameter_update_ratio ?? {}),
    ]),
  ].sort()
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <caption className="pb-2 text-left font-medium">
          Parameter owners
        </caption>
        <thead>
          <tr>
            <th scope="col">Owner</th>
            <th scope="col">Gradient norm</th>
            <th scope="col">Update / weight</th>
          </tr>
        </thead>
        <tbody>
          {owners.slice(0, 128).map((owner) => (
            <tr key={owner} className="border-t border-border">
              <th scope="row" className="py-2 font-mono font-normal break-all">
                {owner}
              </th>
              <td className="tabular-nums">
                {measurement(point.parameter_grad_norm[owner])}
              </td>
              <td className="tabular-nums">
                {measurement(point.parameter_update_ratio?.[owner])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {owners.length > 128 && (
        <p className="text-sm">
          Showing 128 of {owners.length} parameter owners.
        </p>
      )}
    </div>
  )
}

/** Shows each measured execution occurrence separately, including repeated calls sharing one module. */
function Occurrences({ rows }: { rows: ModelDiagnosticOccurrence[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <caption className="pb-2 text-left font-medium">
          Execution occurrences
        </caption>
        <thead>
          <tr>
            <th scope="col">Module / call</th>
            <th scope="col">Outputs</th>
            <th scope="col">Output RMS</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={`${row.instance_path}:${row.call_path.join("/")}:${index}`}
              className="border-t border-border"
            >
              <th scope="row" className="py-2 font-mono font-normal break-all">
                {row.instance_path} · {row.call_path.join("/") || "root"}
              </th>
              <td>
                {row.shapes
                  .map(
                    (shape, output) =>
                      `[${shape.join(", ")}] ${row.dtypes[output]}`
                  )
                  .join("; ")}
              </td>
              <td className="tabular-nums">
                {row.output_rms.length
                  ? row.output_rms.map(measurement).join(", ")
                  : "Unavailable"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Selects actual reported diagnostics without inventing Transformer measurements for custom models. */
export function JobDiagnostics({ diag }: { diag: TrainingDiagnosticPoint[] }) {
  const models = diag.filter(modelDiagnostics)
  const legacy = diag.filter(legacyDiagnostics)
  const latest = models.at(-1)
  const losses = [
    "loss_masked",
    "loss_unmasked",
    "pretrain_loss",
    "base_kl",
  ] as const
  return (
    <>
      {(legacy.length > 0 || diag.length === 0) && (
        <DiagnosticsPanel diag={legacy} />
      )}
      {latest && (
        <Panel variant="open" title="Model diagnostics">
          <div className="grid gap-5">
            <p className="text-sm">
              {latest.step === undefined
                ? "Latest model report"
                : `Step ${latest.step}`}
            </p>
            <ParameterOwners point={latest} />
            {latest.occurrences && <Occurrences rows={latest.occurrences} />}
            {losses.some((key) => latest[key] !== undefined) && (
              <dl className="grid gap-2 text-sm">
                {losses
                  .filter((key) => latest[key] !== undefined)
                  .map((key) => (
                    <div key={key}>
                      <dt>
                        {
                          {
                            loss_masked: "Assistant-token loss",
                            loss_unmasked: "Prompt-token loss",
                            pretrain_loss: "Held-out pretraining loss",
                            base_kl: "Base model KL",
                          }[key]
                        }
                      </dt>
                      <dd>{measurement(latest[key])}</dd>
                    </div>
                  ))}
              </dl>
            )}
          </div>
        </Panel>
      )}
      {diag.length > 0 && !latest && legacy.length === 0 && (
        <p className="text-sm">
          No compatible model diagnostics were reported.
        </p>
      )}
    </>
  )
}
