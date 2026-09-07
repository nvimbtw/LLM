// The sandbox behind the backend's `bash` tool.
//
//   POST /run   {"session": "<id>", "command": "ls -l", "timeout_ms": 10000}
//            -> {"stdout": "...", "stderr": "...", "exit_code": 0, "timed_out": false}
//   GET  /health -> {"ok": true, "sessions": 3}
//
// A Node service and not part of the backend because just-bash is a TypeScript
// library: it *simulates* bash over a virtual filesystem, with `cat`, `grep`,
// `sed`, `jq` and the rest implemented in JS and no `execve` under any of them.
// That is the property the whole tool rests on — a model that writes `rm -rf /`
// deletes an object in a Map.
//
// It sits beside `deploy/searxng/` for the same reason: the backend reaching a
// tool over loopback HTTP is the shape this project already has, and it keeps
// the arrow pointing away from the frontend (frontend → backend → service).
//
// Bound to 127.0.0.1. An open shell endpoint, sandboxed or not, is not
// something to leave listening on a public interface: it is unauthenticated,
// and CPU is a real resource even when the filesystem is imaginary.

import { createServer } from "node:http"

import { Bash } from "just-bash"

const HOST = process.env.BASH_HOST ?? "127.0.0.1"
// 8899 because this project already holds 8080 (backend), 8001 (engine),
// 8002 (data engine), 8888 (SearXNG) and 3000 (UI).
const PORT = Number(process.env.BASH_PORT ?? 8899)

// How many conversations keep a filesystem at once, and for how long after
// their last command. A session is a chat: cheap, but not free, and a server
// left running for a week should not hold every chat anyone ever had.
const MAX_SESSIONS = Number(process.env.BASH_SESSIONS ?? 256)
const SESSION_TTL_MS = Number(process.env.BASH_SESSION_TTL_SECS ?? 3600) * 1000

// The sandbox's own ceiling, when the caller names none. The backend sends one
// (`TOOL_TIMEOUT_SECS`); this is what stops a direct caller hanging the process.
const DEFAULT_TIMEOUT_MS = Number(process.env.BASH_TIMEOUT_MS ?? 10_000)
const MAX_TIMEOUT_MS = Number(process.env.BASH_MAX_TIMEOUT_MS ?? 30_000)

// Enough output to be worth cutting down on the far side, and not so much that
// a `find /` fills a socket. The model's real budget is `BASH_BUDGET_CHARS` in
// the backend, which is two orders of magnitude smaller than this.
const MAX_OUTPUT_BYTES = Number(process.env.BASH_MAX_OUTPUT_BYTES ?? 65_536)

const MAX_BODY_BYTES = 256 * 1024

/**
 * One sandbox per conversation, so a file written in one round is still there
 * in the next — which is the difference between a shell and a calculator.
 *
 * Insertion-ordered, which a JS Map is: the oldest key is the first one out
 * when the cap is hit. Eviction is silent by design — the next command simply
 * gets a fresh filesystem, the same as a machine that rebooted, and telling the
 * model its files are gone is worth less than the tokens it would cost.
 */
const sessions = new Map()

function sandboxFor(key) {
  const now = Date.now()

  for (const [id, entry] of sessions) {
    if (now - entry.touched > SESSION_TTL_MS) sessions.delete(id)
  }

  const existing = sessions.get(key)
  if (existing) {
    existing.touched = now
    // Re-inserted so it counts as the most recent for eviction.
    sessions.delete(key)
    sessions.set(key, existing)
    return existing.bash
  }

  while (sessions.size >= MAX_SESSIONS) {
    sessions.delete(sessions.keys().next().value)
  }

  // Defaults everywhere: an in-memory filesystem, no network, and no Python or
  // JavaScript runtime. Every one of those is a capability, and a capability
  // this project has not decided to hand a language model is one it does not
  // have. Turning any of them on is an edit here, deliberately.
  const bash = new Bash()
  sessions.set(key, { bash, touched: now })
  return bash
}

function clamp(text) {
  if (typeof text !== "string") return ""
  return Buffer.byteLength(text) <= MAX_OUTPUT_BYTES
    ? text
    : Buffer.from(text).subarray(0, MAX_OUTPUT_BYTES).toString("utf8")
}

async function run({ session, command, timeout_ms }) {
  const budget = Math.min(Number(timeout_ms) || DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS)
  const bash = sandboxFor(String(session ?? "default"))

  const controller = new AbortController()
  const stop = setTimeout(() => controller.abort(), budget)
  try {
    const result = await bash.exec(String(command), { signal: controller.signal })
    return {
      stdout: clamp(result.stdout),
      stderr: clamp(result.stderr),
      exit_code: result.exitCode ?? 0,
      timed_out: false,
    }
  } catch (e) {
    // An abort is the timeout firing, and it is an answer rather than a fault:
    // the model asked for something that does not finish, and "it was still
    // running" is what it needs to read. Anything else is this service failing,
    // which the backend should see as a 500 and turn into its own caveat.
    if (controller.signal.aborted) {
      return { stdout: "", stderr: "", exit_code: 124, timed_out: true }
    }
    throw e
  } finally {
    clearTimeout(stop)
  }
}

function json(response, code, payload) {
  const body = JSON.stringify(payload)
  response.writeHead(code, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  })
  response.end(body)
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    let size = 0
    const chunks = []
    request.on("data", (chunk) => {
      size += chunk.length
      if (size > MAX_BODY_BYTES) {
        reject(new Error("request body too large"))
        request.destroy()
        return
      }
      chunks.push(chunk)
    })
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")))
    request.on("error", reject)
  })
}

const server = createServer(async (request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    json(response, 200, { ok: true, sessions: sessions.size })
    return
  }
  if (request.method !== "POST" || request.url !== "/run") {
    json(response, 404, { error: "POST /run or GET /health" })
    return
  }

  try {
    const payload = JSON.parse(await readBody(request))
    if (typeof payload.command !== "string" || !payload.command.trim()) {
      json(response, 400, { error: "command is required" })
      return
    }
    json(response, 200, await run(payload))
  } catch (e) {
    // The message and nothing else: it goes into a prompt on the far side.
    json(response, 500, { error: e instanceof Error ? e.message : "failed" })
  }
})

server.listen(PORT, HOST, () => {
  console.log(`just-bash sandbox on http://${HOST}:${PORT}`)
  console.log(`set BASH_URL=http://${HOST}:${PORT} in backend/.env`)
})
