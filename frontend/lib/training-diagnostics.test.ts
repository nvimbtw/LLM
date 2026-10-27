import { test } from "node:test"
import assert from "node:assert/strict"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import { JobDiagnostics } from "../components/training/job-diagnostics"
import {
  createJobTelemetry,
  jobTelemetryEvent,
  reduceJobTelemetry,
} from "./job-telemetry"
import type { DiagPoint, TrainingDiagnosticPoint } from "./training"
import modelReport from "./fixtures/model-diagnostics.json"

/** Renders actual diagnostic variants through the training page's adapter/component boundary. */
function renderReport(diag: TrainingDiagnosticPoint): string {
  const event = jobTelemetryEvent({ type: "train", step: 5, diag })
  assert.ok(event)
  const state = reduceJobTelemetry(createJobTelemetry(), event)
  return renderToStaticMarkup(
    createElement(JobDiagnostics, { diag: state.diag })
  )
}

test("live custom-model diagnostics render owner and repeated-call identities without Transformer arrays", () => {
  const markup = renderReport(modelReport)
  assert.match(markup, /Model diagnostics/)
  assert.match(markup, /core\.gate\.weight/)
  assert.match(markup, /core · 1\/0\/0\/0\/0/)
  assert.match(markup, /core · 1\/0\/0\/0\/1/)
  assert.match(markup, /\[1, 64, 8192\] torch.float32/)
  assert.match(markup, /Latest model report/)
  assert.doesNotMatch(markup, /Attention entropy|Embedding anisotropy|Layer 0/)
  assert.equal("grad_layers" in modelReport, false)
})

test("custom diagnostics retain measured zero and omit unavailable measurements rather than inventing values", () => {
  const markup = renderReport({
    step: 7,
    parameter_grad_norm: { "shared.weight": 0 },
    loss_masked: 0,
  })
  assert.match(markup, /Step 7/)
  assert.match(markup, /0\.000/)
  assert.match(markup, /Unavailable/)
  assert.match(markup, /Assistant-token loss/)
  assert.doesNotMatch(markup, /Prompt-token loss|Execution occurrences/)
})

test("malformed custom summaries are refused and valid owner presentation stays bounded", () => {
  assert.match(
    renderReport({ ...modelReport, parameter_grad_norm: { bad: Number.NaN } }),
    /No compatible model diagnostics/
  )
  assert.match(
    renderReport({
      ...modelReport,
      occurrences: [{ ...modelReport.occurrences[0], shapes: [[-1]] }],
    }),
    /No compatible model diagnostics/
  )
  const markup = renderReport({
    parameter_grad_norm: Object.fromEntries(
      Array.from({ length: 129 }, (_, index) => [`owner.${index}`, index])
    ),
  })
  assert.match(markup, /Showing 128 of 129 parameter owners/)
})

test("legacy Transformer diagnostics still select the existing figures", () => {
  const legacy: DiagPoint = {
    step: 3,
    grad_global: 1,
    grad_layers: [1],
    grad_embedding: 1,
    grad_unembed: 1,
    update_ratio_layers: [0.01],
    act_rms_layers: [1],
    attn_entropy: [[0.5]],
    emb_cos_mean: 0,
    emb_cos_p5: 0,
    emb_cos_p95: 0,
    act_hist: [],
    emb_pca: [],
    emb_pca3: [],
  }
  const markup = renderReport(legacy)
  assert.doesNotMatch(markup, /Model diagnostics|Parameter owners/)
  assert.match(markup, /sampled per report \(1\)/)
})
