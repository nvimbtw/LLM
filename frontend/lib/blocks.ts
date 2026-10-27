import {
  architectureRequest,
  architectureQuery,
  type ArchitectureRequest,
} from "./architecture/http"
import { encodeArchitecture } from "./architecture/decoder"
import type {
  AcceptRevision,
  BlockCreate,
  BlockDraftSave,
  CatalogQuery,
  MetadataPatch,
  ValidationRequest,
} from "./architecture/contracts.generated"
/** Provides bounded block catalog and immutable revision operations through Rust. */
export const blocks = {
  list: (query: CatalogQuery, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks?${architectureQuery("CatalogQuery", query)}`,
      "CatalogPage",
      request
    ),
  detail: (id: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}`,
      "ResourceDetail",
      request
    ),
  draft: (id: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}/draft`,
      "BlockDraftView",
      request
    ),
  revisions: (
    id: string,
    request: ArchitectureRequest,
    cursor?: string | null
  ) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}/revisions?${new URLSearchParams({ limit: "50", ...(cursor ? { cursor } : {}) })}`,
      "RevisionPage",
      request
    ),
  revision: (id: string, revision: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revision)}`,
      "BlockRevisionView",
      request
    ),
  copySource: (id: string, revision: string, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revision)}/copy-source`,
      "BlockCreate",
      request
    ),
  create: (value: BlockCreate, request: ArchitectureRequest) =>
    architectureRequest("/blocks", "ResourceCreated", request, {
      method: "POST",
      body: encodeArchitecture("BlockCreate", value),
    }),
  save: (id: string, value: BlockDraftSave, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}/draft`,
      "DraftSaved",
      request,
      { method: "PUT", body: encodeArchitecture("BlockDraftSave", value) }
    ),
  patch: (id: string, value: MetadataPatch, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}`,
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
      `/blocks/${encodeURIComponent(id)}/validations`,
      "ValidationRun",
      request,
      { method: "POST", body: encodeArchitecture("ValidationRequest", value) }
    ),
  accept: (id: string, value: AcceptRevision, request: ArchitectureRequest) =>
    architectureRequest(
      `/blocks/${encodeURIComponent(id)}/revisions`,
      "AcceptedRevision",
      request,
      { method: "POST", body: encodeArchitecture("AcceptRevision", value) }
    ),
}
