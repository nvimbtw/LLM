"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { CheckCircle2, Upload } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useAuth } from "@/hooks/use-auth"
import { useFileUpload } from "@/hooks/use-file-upload"
import { FileDropzone } from "@/components/files/file-dropzone"
import { FileList } from "@/components/files/file-list"

export function FileUpload() {
  const router = useRouter()
  const { isAdmin, isLoading } = useAuth()

  const {
    files,
    status,
    progress,
    error,
    addFiles,
    removeFile,
    clear,
    upload,
  } = useFileUpload()
  const [rejected, setRejected] = React.useState<string[]>([])

  // Admin-only: bounce non-admins once auth has resolved. The sidebar already
  // hides the link, but the route is directly reachable.
  React.useEffect(() => {
    if (!isLoading && !isAdmin) router.replace("/")
  }, [isLoading, isAdmin, router])

  if (isLoading || !isAdmin) return null

  const uploading = status === "uploading"

  function handleFiles(incoming: FileList | File[]) {
    setRejected(addFiles(incoming))
  }

  async function handleUpload() {
    try {
      await upload()
    } catch {
      // Error surfaced via the hook's `error` state.
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload files</CardTitle>
        <CardDescription>
          Add one or more files to send to the server.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <FileDropzone onFiles={handleFiles} disabled={uploading} />

        {rejected.length > 0 && (
          <p className="text-sm text-destructive">
            Skipped unsupported {rejected.length === 1 ? "file" : "files"}:{" "}
            {rejected.join(", ")}
          </p>
        )}

        <FileList files={files} onRemove={removeFile} disabled={uploading} />

        {uploading && <Progress value={progress} />}

        {status === "error" && error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
        {status === "success" && (
          <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <CheckCircle2 className="size-4" />
            Files uploaded.
          </p>
        )}

        <div className="flex items-center gap-2">
          <Button
            type="button"
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
          >
            {uploading ? (
              <>
                {/* No mark: the button is disabled and its label has changed,
                    which is the whole of what a spinner in here was saying. */}
                Uploading…
              </>
            ) : (
              <>
                <Upload className="size-4" />
                Upload {files.length > 0 ? `(${files.length})` : ""}
              </>
            )}
          </Button>
          {files.length > 0 && !uploading && (
            <Button type="button" variant="ghost" onClick={clear}>
              Clear
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
