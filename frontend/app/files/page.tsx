import { FileUpload } from "@/components/files/file-upload"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function FilesPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Files"
        description="Upload documents to send to the server."
      />
      <FileUpload />
    </PageShell>
  )
}
