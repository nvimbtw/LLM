import { after, afterEach, beforeEach, test } from "node:test"
import assert from "node:assert/strict"

import {
  SESSION_EXPIRED_EVENT,
  clearSession,
  handleSessionExpired,
  setToken,
  subscribeSessionExpiry,
} from "./api"

const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window")
const originalStorage = Object.getOwnPropertyDescriptor(
  globalThis,
  "localStorage"
)

let token: string | null = null
let pathname = "/admin"
const assigned: string[] = []
const expiryListeners = new Set<() => void>()
let unsubscribe: (() => void) | null = null

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: {
    location: {
      get pathname() {
        return pathname
      },
      assign: (path: string) => assigned.push(path),
    },
    addEventListener: (type: string, listener: () => void) => {
      if (type === SESSION_EXPIRED_EVENT) expiryListeners.add(listener)
    },
    removeEventListener: (type: string, listener: () => void) => {
      if (type === SESSION_EXPIRED_EVENT) expiryListeners.delete(listener)
    },
    dispatchEvent: (event: Event) => {
      if (event.type === SESSION_EXPIRED_EVENT)
        for (const listener of [...expiryListeners]) listener()
      return true
    },
  },
})
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (key: string) => (key === "auth_token" ? token : null),
    setItem: (key: string, value: string) => {
      if (key === "auth_token") token = value
    },
    removeItem: (key: string) => {
      if (key === "auth_token") token = null
    },
  },
})

beforeEach(() => {
  pathname = "/admin"
  assigned.length = 0
  token = "account-a"
  setToken("account-a")
})
afterEach(() => {
  unsubscribe?.()
  unsubscribe = null
})
after(() => {
  if (originalWindow)
    Object.defineProperty(globalThis, "window", originalWindow)
  if (originalStorage)
    Object.defineProperty(globalThis, "localStorage", originalStorage)
})

test("a rejected session clears the token and reports once for its path", () => {
  let reported = 0
  unsubscribe = subscribeSessionExpiry(() => {
    reported += 1
  })

  assert.equal(handleSessionExpired(), true)
  assert.equal(token, null)
  assert.equal(reported, 1)
  assert.deepEqual(assigned, [])

  assert.equal(handleSessionExpired(), false)
  assert.equal(reported, 1)
})

test("a late rejection from a replaced session leaves the newer token alone", () => {
  setToken("account-b")
  assert.equal(handleSessionExpired("account-a"), false)
  assert.equal(token, "account-b")
  assert.deepEqual(assigned, [])
})

test("without a mounted shell the fallback loads /login", () => {
  assert.equal(handleSessionExpired(), true)
  assert.deepEqual(assigned, ["/login"])
})

test("anonymous rejections on a new path are reported again", () => {
  clearSession()
  let reported = 0
  unsubscribe = subscribeSessionExpiry(() => {
    reported += 1
  })

  assert.equal(handleSessionExpired(null), false)
  pathname = "/models"
  assert.equal(handleSessionExpired(null), false)
  assert.equal(reported, 2)
})
