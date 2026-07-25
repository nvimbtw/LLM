import { ModelDetail } from "@/components/models/model-detail";

export default async function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:px-6">
      <ModelDetail modelId={id} />
    </div>
  );
}
