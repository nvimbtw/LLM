import { FileUpload } from "@/components/files/file-upload";

export default function FilesPage() {
  return (
    <div className="mx-auto w-full max-w-2xl flex-1 px-4 py-8 sm:px-6">
      <header className="mb-8">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Files
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload documents to send to the server.
        </p>
      </header>
      <FileUpload />
    </div>
  );
}
