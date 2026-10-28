import type { TrainConfig } from "./training"
import { decodeArchitecture, identityDigest } from "./architecture/decoder"
import type {
  ArchitectureSelection,
  TrainingArchitectureRequest,
  Recipe,
} from "./architecture/contracts.generated"
import { parameterError } from "./architecture/parameters"

export interface TrainingArchitectureCheck {
  key: string
  errors: string[]
}

/** Identifies browser selection/numerical inputs only for stale-result detection; runtime/tokenizer compatibility and receipt authorization remain engine-owned. */
export function trainingSelectionKey(config: TrainConfig): string {
  return identityDigest({
    selection: config.architecture
      ? {
          revision_id: config.architecture.revision_id,
          parameters: config.architecture.parameters,
        }
      : null,
    tokenizer_id: config.tokenizer_id || null,
    batch_size: config.batch_size,
    precision: config.precision ?? "bf16",
    seed: config.seed ?? 0,
    allow_tf32: config.allow_tf32 ?? true,
  })
}

/** Interprets one immutable selected-recipe request and drops stale browser-carried evidence instead of treating old receipt IDs as authorization. */
export function selectedArchitectureForEngine(
  config: TrainConfig
): ArchitectureSelection {
  if (!config.architecture)
    throw new Error("Choose an accepted architecture revision.")
  const selection = decodeArchitecture(
    "ArchitectureSelection",
    config.architecture
  )
  return {
    ...selection,
    validation_run_id:
      config.architecture_validation_key === trainingSelectionKey(config)
        ? selection.validation_run_id
        : null,
  }
}

/** Reads engine-owned prospective bindings/profile from explicit training controls, excluding any old authorization receipt from the metadata request. */
export function trainingPreviewRequest(
  config: TrainConfig
): TrainingArchitectureRequest {
  const selection = selectedArchitectureForEngine(config)
  return decodeArchitecture("TrainingArchitectureRequest", {
    schema_version: 1,
    selection: { ...selection, validation_run_id: null },
    batch_size: config.batch_size,
    precision: config.precision ?? "bf16",
    seed: config.seed ?? 0,
    allow_tf32: config.allow_tf32 ?? true,
    tokenizer_id: config.tokenizer_id || null,
  })
}

/** Restores reusable settings/pins without transporting authorization across preset/model loads or changed runtime sessions. */
export function withoutTrainingReceipt(config: TrainConfig): TrainConfig {
  const value = { ...config }
  delete value.architecture_validation_key
  return config.architecture
    ? {
        ...value,
        architecture: {
          ...decodeArchitecture("ArchitectureSelection", config.architecture),
          validation_run_id: null,
        },
      }
    : value
}

/** Checks loaded declared parameter inputs without source parsing or default legacy-shape assumptions; optional defaults remain engine-owned bindings. */
export function trainingParameterErrors(
  selection: ArchitectureSelection,
  recipe: Recipe
): Record<string, string> {
  const errors: Record<string, string> = {}
  for (const name of Object.keys(selection.parameters))
    if (!(name in recipe.parameters)) errors[name] = "Unknown recipe parameter."
  for (const [name, declaration] of Object.entries(recipe.parameters)) {
    const value = selection.parameters[name]
    const error =
      value === undefined
        ? declaration.required
          ? "Required parameter."
          : null
        : parameterError(declaration, value)
    if (error) errors[name] = error
  }
  return errors
}
