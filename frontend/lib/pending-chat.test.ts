import { test } from "node:test"
import assert from "node:assert/strict"
import { PendingChatMessages, type PendingMessage } from "./pending-chat"

const message: PendingMessage = {
  content: "First question",
  modelId: "checkpoint-42",
  budget: "high",
  attachments: [
    {
      id: "upload-42",
      filename: "notes.txt",
      mediaType: "text/plain",
      size: 128,
    },
  ],
}

test("ordinary and project chats keep their exact first-message choices and deliver once", () => {
  const pending = new PendingChatMessages()
  pending.setSession("a")
  pending.put("a", "project-chat", message)
  assert.equal(pending.take("a", "project-chat"), message)
  assert.equal(pending.take("a", "project-chat"), null)
  assert.equal(message.modelId, "checkpoint-42")
  assert.equal(message.budget, "high")
  assert.equal(message.attachments[0].id, "upload-42")
})

test("replacement accounts discard pending drafts and an old create response cannot insert into the new account", () => {
  const pending = new PendingChatMessages()
  pending.setSession("a")
  pending.put("a", "a-chat", message)
  assert.equal(pending.take("b", "a-chat"), null)
  assert.throws(
    () => pending.put("a", "late-a-chat", message),
    /account changed/
  )
  pending.put("b", "b-chat", message)
  pending.setSession(null)
  assert.equal(pending.take("b", "b-chat"), null)
})
