import type {
  CatalogPage,
  GraphFragment,
  GraphQuery,
  ResourceDetail,
  ValidationReport,
} from "@/lib/architecture/contracts.generated"
import type {
  ConfigurationIdentity,
  ResourceIdentity,
  RunIdentity,
} from "@/lib/architecture/read-owner"
export interface ArchitectureGraphProps {
  identity: ConfigurationIdentity | RunIdentity
  fragment: GraphFragment
  loading: boolean
  error: string | null
  onQuery: (query: GraphQuery) => void
  onSource: (revisionId: string, line: number | null) => void
}
export interface ArchitectureCatalogProps {
  identity: ResourceIdentity
  page: CatalogPage
  loading: boolean
  error: string | null
  onNextPage: (cursor: string) => void
}
export interface ArchitectureResourceProps {
  identity: ResourceIdentity
  detail: ResourceDetail
}
export interface ArchitectureReportProps {
  identity: RunIdentity
  report: ValidationReport
  stale: boolean
}
