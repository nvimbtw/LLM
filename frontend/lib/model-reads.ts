import { API_URL, apiFetch } from "./api"

/** Reads public model metadata anonymously or with the captured bearer identity; anonymous requests never borrow a newer account's credentials. */
export function readModel(
  path: string,
  token: string | null,
  signal?: AbortSignal
): Promise<Response> {
  return token
    ? apiFetch(path, { signal, headers: { Authorization: `Bearer ${token}` } })
    : fetch(`${API_URL}${path}`, {
        signal,
        headers: { "Content-Type": "application/json" },
      })
}
