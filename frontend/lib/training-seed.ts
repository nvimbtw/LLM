import { getJob, trainConfig } from "./jobs"
import { withoutTrainingReceipt } from "./training-selection"
import type { TrainConfig } from "./training"

/** Loads reusable settings for one captured-session run, preserving immutable selection and explicit resume overrides while discarding old validation authorization. */
export async function loadTrainingSeed(
  id: string,
  resume: string | null,
  request: { token: string; signal?: AbortSignal }
): Promise<TrainConfig> {
  const job = await getJob(id, request)
  const config = trainConfig(job)
  if (!config)
    throw new Error("This execution has no reusable training settings.")
  const reusable = withoutTrainingReceipt(config)
  return resume ? { ...reusable, resume_from: resume } : reusable
}
