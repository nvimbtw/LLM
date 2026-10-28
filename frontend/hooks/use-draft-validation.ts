"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { useValidationSubmission } from "./use-validation-submission"
import type { ArchitectureKind } from "./use-architecture-catalog"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import { listTokenizers, type TokenizerInfo } from "@/lib/tokenizers"
import { getArchitectureTokenizerBinding } from "@/lib/model-validation"
import { identityDigest } from "@/lib/architecture/decoder"
import {
  configuredDraftValidation,
  draftAcceptance,
  suggestedInputShape,
} from "@/lib/architecture/validation-form"
import type { ParameterValue } from "@/lib/architecture/parameters"
import type { ArchitectureDraft, DraftSnapshot } from "@/lib/architecture/draft"
import type {
  ArchitectureSchema,
  BlockDeclaration,
  Recipe,
  ResourceDetail,
  RuntimeProfile,
  ValidationReport,
} from "@/lib/architecture/contracts.generated"

export interface DraftValidationInputs {
  kind: ArchitectureKind
  id: string
  document: ArchitectureDraft
  draft: DraftSnapshot
  readOnly: boolean
  onAccepted: (detail: ResourceDetail) => void
}

interface ProspectiveMetadata {
  schema: ArchitectureSchema
  tokenizers: TokenizerInfo[]
  tokenizerError: string | null
}

/** Owns this saved draft's metadata reads, configuration, frozen submissions and exact-receipt acceptance; cleanup only detaches browser observation. */
export function useDraftValidation({
  kind,
  id,
  document,
  draft,
  readOnly,
  onAccepted,
}: DraftValidationInputs) {
  const token = useSessionToken()
  const { owner: submission, state: submitted } = useValidationSubmission(
    kind,
    id
  )
  const [metadata, setMetadata] = React.useState<ProspectiveMetadata | null>(
    null
  )
  const [declaration, setDeclaration] = React.useState<
    BlockDeclaration | Recipe | null
  >(null)
  const [profile, setProfile] = React.useState<RuntimeProfile | null>(null)
  const [parameters, setParameters] = React.useState<
    Record<string, ParameterValue | undefined>
  >({})
  const [shapes, setShapes] = React.useState<Record<string, string>>({})
  const [tokenizer, setTokenizer] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const [busy, setBusy] = React.useState(false)
  const [accepted, setAccepted] = React.useState<string | null>(null)
  const [metadataEpoch, reloadMetadata] = React.useReducer(
    (value: number) => value + 1,
    0
  )
  const lifetime = React.useRef<{
    active: boolean
    command: boolean
    reads: ArchitectureReadOwner
  } | null>(null)
  const dirty = draft.text !== draft.savedText
  const subject = { kind: "draft" as const, ...draft.saved }
  const formKey = `${draft.editRevision}:${identityDigest({ parameters, profile, shapes, tokenizer })}`
  const stale = submitted?.key !== formKey || dirty
  const unavailable = readOnly || dirty || draft.saving || draft.conflict

  React.useEffect(() => {
    if (!token) return
    const owner = {
      active: true,
      command: false,
      reads: new ArchitectureReadOwner({
        token,
        resource: `${kind}:${id}:${draft.saved.draft_version}`,
      }),
    }
    lifetime.current = owner
    void owner.reads
      .read(async (signal) => {
        const [schema, tokenizers] = await Promise.allSettled([
          architectures.schema({ token, signal }),
          listTokenizers(signal, token),
        ])
        if (schema.status === "rejected") throw schema.reason
        return {
          schema: schema.value,
          tokenizers: tokenizers.status === "fulfilled" ? tokenizers.value : [],
          tokenizerError:
            tokenizers.status === "fulfilled"
              ? null
              : "Vocabulary metadata is unavailable. Reload metadata before selecting a vocabulary.",
        }
      })
      .then((value) => {
        if (!value || !owner.active) return
        setMetadata(value)
        const prospective = value.schema.default_profile
        setProfile(
          kind === "block"
            ? {
                ...prospective,
                domains: prospective.domains.map((domain) => ({
                  ...domain,
                  batch_max: domain.batch_min,
                  sequence_max: domain.sequence_min,
                })),
              }
            : prospective
        )
      })
      .catch((cause: unknown) => {
        if (owner.active)
          setError(
            cause instanceof Error
              ? cause.message.slice(0, 2000)
              : "Prospective metadata did not load."
          )
      })
    return () => {
      owner.active = false
      owner.reads.dispose()
    }
  }, [token, kind, id, draft.saved.draft_version, metadataEpoch])

  /** Explicitly reloads prospective metadata after a failed read without discarding editor bytes or submitting a job. */
  function retryMetadata() {
    if (lifetime.current?.command) return
    setError(null)
    reloadMetadata()
  }

  /** Retains declarations only for this saved-source component; optional defaults are form inputs and never rewrite source. */
  const setDeclaredInputs = React.useCallback(
    (value: BlockDeclaration | Recipe) => {
      setDeclaration(value)
      setParameters(
        Object.fromEntries(
          Object.entries(value.parameters)
            .filter(([, item]) => item.default != null)
            .map(([name, item]) => [name, item.default!])
        )
      )
      setShapes({})
    },
    []
  )

  /** Inspects exact saved Python through the supervised queue or asks the metadata-only YAML reader; neither action accepts a revision. */
  async function inspect() {
    const owner = lifetime.current
    if (
      !token ||
      !owner?.active ||
      owner.command ||
      unavailable ||
      busy ||
      submitted?.pending
    )
      return
    owner.command = true
    setError(null)
    setBusy(true)
    try {
      if (kind === "block")
        await submission?.submit(`inspect:${draft.editRevision}`, {
          schema_version: 1,
          mode: "inspect",
          subject,
        })
      else {
        const result = await owner.reads.read((signal) =>
          architectures.declaration(id, subject, { token: token!, signal })
        )
        if (result && owner.active) setDeclaredInputs(result.recipe)
      }
    } catch (cause) {
      if (owner.active)
        setError(
          cause instanceof Error
            ? cause.message.slice(0, 2000)
            : "Declaration inspection failed."
        )
    } finally {
      owner.command = false
      if (owner.active) setBusy(false)
    }
  }

  /** Queues one complete frozen form snapshot, deriving tokenizer facts only from the owned artifact reader and leaving context resolution to the recipe. */
  async function validate() {
    const owner = lifetime.current
    if (
      !token ||
      !owner?.active ||
      owner.command ||
      !declaration ||
      !profile ||
      !submission ||
      unavailable ||
      busy ||
      submitted?.pending
    )
      return
    owner.command = true
    setError(null)
    setBusy(true)
    const capturedRevision = document.getSnapshot().editRevision
    try {
      const binding = tokenizer
        ? await owner.reads.read((signal) =>
            getArchitectureTokenizerBinding(tokenizer, {
              token: token!,
              signal,
            })
          )
        : null
      if (
        !owner.active ||
        document.getSnapshot().editRevision !== capturedRevision
      )
        return
      const request = configuredDraftValidation(
        subject,
        declaration,
        { parameters, profile, shapes },
        binding ?? null
      )
      await submission.submit(formKey, request)
    } catch (cause) {
      if (owner.active)
        setError(
          cause instanceof Error
            ? cause.message.slice(0, 2000)
            : "Validation could not be submitted."
        )
    } finally {
      owner.command = false
      if (owner.active) setBusy(false)
    }
  }

  /** Supplies explicit synthetic shape suggestions only for literals and batch/sequence; undeclared dimensions remain empty for user input. */
  function inputShape(name: string): string {
    if (shapes[name] !== undefined) return shapes[name]
    return suggestedInputShape(declaration!.inputs[name], profile!)
  }

  /** Accepts only the current fully valid report and refreshes metadata without replacing any newer local source document. */
  async function accept(report: ValidationReport) {
    const owner = lifetime.current
    const receipt = report.receipt
    if (
      !token ||
      !owner?.active ||
      owner.command ||
      !receipt ||
      report.verdict !== "valid" ||
      stale ||
      unavailable ||
      busy ||
      submitted?.key !== formKey ||
      document.getSnapshot().editRevision !== draft.editRevision
    )
      return
    owner.command = true
    const api = kind === "block" ? blocks : architectures
    setBusy(true)
    setError(null)
    try {
      const result = await owner.reads.read(async (signal) => {
        const result = await api.accept(
          id,
          draftAcceptance(document.getSnapshot(), submitted!, formKey, report),
          { token: token!, signal }
        )
        if (
          result.reference.resource_id !== id ||
          result.reference.content_hash !== draft.saved.content_hash ||
          result.acceptance_identity_hash !== receipt.identity_hash ||
          result.acceptance_receipt_id !== receipt.receipt_id
        )
          throw new Error(
            "Accepted revision differs from the validated configuration. Reload metadata to check the server state."
          )
        return {
          result,
          detail: await api.detail(id, { token: token!, signal }),
        }
      })
      if (result && owner.active) {
        setAccepted(result.result.reference.revision_id)
        onAccepted(result.detail)
      }
    } catch (cause) {
      if (owner.active)
        setError(
          cause instanceof Error
            ? cause.message.slice(0, 2000)
            : "Revision was not accepted."
        )
    } finally {
      owner.command = false
      if (owner.active) setBusy(false)
    }
  }

  return {
    metadata,
    declaration,
    profile,
    parameters,
    tokenizer,
    error,
    busy,
    accepted,
    submitted,
    submission,
    stale,
    unavailable,
    inspect,
    validate,
    accept,
    inputShape,
    setDeclaredInputs,
    setParameters,
    setProfile,
    setShapes,
    setTokenizer,
    retryMetadata,
  }
}
