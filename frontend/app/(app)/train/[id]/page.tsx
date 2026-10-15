import { ExecutionView } from "@/components/jobs/execution-view"

/** Opens the domain execution while preserving its existing queue and telemetry behavior. */
export default async function ExecutionPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return <ExecutionView id={id} kind="train" />
}
