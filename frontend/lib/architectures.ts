import {
  architectureRequest,
  architectureQuery,
  type ArchitectureRequest,
} from "./architecture/http"
import { decodeArchitecture, encodeArchitecture } from "./architecture/decoder"
import { draftDigest } from "./architecture/draft"
import type {
  AcceptRevision,
  ArchitectureCreate,
  ArchitectureDraftSave,
  CatalogQuery,
  MetadataPatch,
  ValidationRequest,
  DraftSubject,
  TrainingArchitectureRequest,
} from "./architecture/contracts.generated"
/** Provides bounded architecture catalog and immutable revision operations through Rust. */
export const architectures = {
  selected: async (revision: string, request: ArchitectureRequest) => {
    const value = await architectureRequest(
      `/architectures/revisions/${encodeURIComponent(revision)}`,
      "ArchitectureRevisionView",
      request
    )
    const reference = value.accepted.reference
    if (
      reference.kind !== "architecture" ||
      reference.revision_id !== revision.toLowerCase() ||
      reference.content_hash !== draftDigest(value.yaml)
    )
      throw new Error("Selected recipe differs from its immutable revision.")
    return value
  },
  trainingPreview: async (
    value: TrainingArchitectureRequest,
    request: ArchitectureRequest
  ) => {
    const expected = decodeArchitecture("TrainingArchitectureRequest", value)
    const result = await architectureRequest(
      `/architectures/revisions/${encodeURIComponent(expected.selection.revision_id)}/training-preview`,
      "TrainingArchitecturePreview",
      request,
      {
        method: "POST",
        body: encodeArchitecture("TrainingArchitectureRequest", expected),
      }
    )
    if (
      result.architecture.revision_id !==
        expected.selection.revision_id.toLowerCase() ||
      result.profile.domains[0].batch_max !== expected.batch_size ||
      result.profile.numerical_seed !== expected.seed ||
      (expected.tokenizer_id == null
        ? result.tokenizer != null
        : result.tokenizer?.tokenizer_id !==
          expected.tokenizer_id.toLowerCase())
    )
      throw new Error(
        "Training preview differs from the selected revision or numerical inputs."
      )
    return result
  },
  schema: (request: ArchitectureRequest) =>
    architectureRequest("/architectures/schema", "ArchitectureSchema", request),
  declaration: async (
    id: string,
    subject: DraftSubject,
    request: ArchitectureRequest
  ) => {
    const result = await architectureRequest(
      `/architectures/${encodeURIComponent(id)}/declaration`,
      "RecipeDraftInspection",
      request,
      { method: "POST", body: encodeArchitecture("DraftSubject", subject) }
    )
    if (
      subject.resource_id !== id ||
      result.subject.resource_id !== id ||
      result.subject.draft_version !== subject.draft_version ||
      result.subject.content_hash !== subject.content_hash
    )
      throw new Error("Recipe declaration belongs to a different saved draft.")
    return result
  },
  list: (query: CatalogQuery, request: ArchitectureRequest) =>
    architectureRequest(
      `/architectures?${architectureQuery("CatalogQuery", query)}`,
      "CatalogPage",
      request
    ),
  detail: (id: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}`,
      "ResourceDetail",
      request
    ),
  draft: (id: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}/draft`,
      "ArchitectureDraftView",
      request
    ),
  revisions: (
    id: string,
    request: ArchitectureRequest,
    cursor?: string | null
  ) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}/revisions?${new URLSearchParams({ limit: "50", ...(cursor ? { cursor } : {}) })}`,
      "RevisionPage",
      request
    ),
  revision: (id: string, revision: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revision)}`,
      "ArchitectureRevisionView",
      request
    ),
  create: (value: ArchitectureCreate, request: ArchitectureRequest) =>
    architectureRequest("/architectures", "ResourceCreated", request, {
      method: "POST",
      body: encodeArchitecture("ArchitectureCreate", value),
    }),
  save: (
    id: string,
    value: ArchitectureDraftSave,
    request: ArchitectureRequest
  ) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}/draft`,
      "DraftSaved",
      request,
      {
        method: "PUT",
        body: encodeArchitecture("ArchitectureDraftSave", value),
      }
    ),
  patch: (id: string, value: MetadataPatch, request: ArchitectureRequest) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}`,
      "ResourceDetail",
      request,
      { method: "PATCH", body: encodeArchitecture("MetadataPatch", value) }
    ),
  validate: (
    id: string,
    value: ValidationRequest,
    request: ArchitectureRequest
  ) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}/validations`,
      "ValidationRun",
      request,
      { method: "POST", body: encodeArchitecture("ValidationRequest", value) }
    ),
  accept: (id: string, value: AcceptRevision, request: ArchitectureRequest) =>
    architectureRequest(
      `/architectures/${encodeURIComponent(id)}/revisions`,
      "AcceptedRevision",
      request,
      { method: "POST", body: encodeArchitecture("AcceptRevision", value) }
    ),
}
