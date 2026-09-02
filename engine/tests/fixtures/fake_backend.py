"""A stand-in for the backend's `/internal` routes.

The engine reads the text it trains on from the data engine, through the
backend. Every build path therefore has an HTTP call in it, and the choice for
the tests is to stub that call or to answer it. This answers it: a real socket,
real pages, real JSON, so what is exercised is the conversation the engine
actually has — paging, caps, ordering — rather than a mock asserting that the
code calls the functions it calls.

Holds the same text the fixture tree used to hold on disk, so a corpus packed in
a test is byte-identical to the one packed before the move.
"""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

#: Fixed rather than minted, so a key derived from a selection is stable across
#: runs of the suite and a failure names the same dataset every time.
TINY_DATASET_ID = "11111111-1111-4111-8111-111111111111"
TINY_CHAT_ID = "22222222-2222-4222-8222-222222222222"
TINY_GYM_ID = "33333333-3333-4333-8333-333333333333"

#: The data engine's own cap on a page (`MAX_PAGE` in its datasets router). The
#: stand-in enforces it too — a client that asked for more would get a 422 from
#: the real one and the tests must not be the only place that works.
MAX_PAGE = 200


class Dataset:
    def __init__(self, dataset_id: str, name: str, kind: str, documents: list[dict]):
        self.id = dataset_id
        self.name = name
        self.kind = kind
        self.documents = documents

    def view(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "document_count": len(self.documents),
            "byte_size": sum(d.get("byte_size", 0) for d in self.documents),
            "updated_at": "2026-08-19T00:00:00+00:00",
        }


def text_documents(texts: list[str]) -> list[dict]:
    return [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_OID, f"text-{i}")),
            "ordinal": i,
            "content": text,
            "messages": None,
            "byte_size": len(text.encode("utf-8")),
        }
        for i, text in enumerate(texts)
    ]


def chat_documents(conversations: list[list[dict]]) -> list[dict]:
    """Conversations as the data engine holds them: `[{role, content}]`, which is
    not the `{turns: [{role, text}]}` the renderer wants — `sft._conversation` is
    what bridges the two, and it only gets exercised if the fixture is honest
    about the difference."""
    return [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_OID, f"chat-{i}")),
            "ordinal": i,
            "content": None,
            "messages": messages,
            "byte_size": sum(len(m["content"].encode("utf-8")) for m in messages),
        }
        for i, messages in enumerate(conversations)
    ]


def demonstration_documents(count: int) -> list[dict]:
    """Gym demonstrations as the data engine holds them, which is *not* how the
    other chat fixtures are shaped.

    A tool turn here names the call it answers and carries the body alone —
    `POST /datasets/demonstrations` stores exactly this — where a converted Hub
    corpus carries the whole rendered block and names nothing. Both reach
    `sft._turns`, and only one of them is rendered there. A fixture that used
    the pre-rendered shape for both would never exercise that branch.

    **The assistant turn before it names the same call**, which is the shape
    that decides whether these demonstrations teach tool use at all: SFT trains
    assistant turns and masks tool turns, so a command stored only on the tool
    turn is a command the model gets no gradient on. This fixture used to omit
    it, which is how the corpus went a whole release teaching a model to narrate
    a search and then close its turn.
    """
    conversations = []
    for i in range(count):
        needle = f"ANSWER_{i:06x}"
        conversations.append(
            [
                {"role": "system", "content": "You have a shell."},
                {"role": "user", "content": f"Which file contains {needle}?"},
                {
                    "role": "assistant",
                    "content": "<|par|>Searching the files.<|/par|>",
                    "tool": "bash",
                    "argument": f"grep -rn {needle} .",
                },
                {
                    "role": "tool",
                    "tool": "bash",
                    "argument": f"grep -rn {needle} .",
                    "content": f"<|codeblock|>./notes{i}.txt:3:{needle}<|/codeblock|>",
                },
                {"role": "assistant", "content": f"notes{i}.txt"},
            ]
        )
    return chat_documents(conversations)


class FakeBackend:
    """Serves `/internal/datasets/{id}` and `/internal/datasets/{id}/documents`.

    `fail_next` makes the next N requests fail, for the retry paths.
    """

    def __init__(self, datasets: list[Dataset]):
        self.datasets = {dataset.id: dataset for dataset in datasets}
        self.requests: list[dict] = []
        self.fail_next = 0
        self.fail_status = 500

        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                backend.requests.append(
                    {
                        "path": url.path,
                        "authorization": self.headers.get("Authorization", ""),
                        **{k: int(v) for k, v in query.items() if v.lstrip("-").isdigit()},
                    }
                )

                if backend.fail_next > 0:
                    backend.fail_next -= 1
                    return self._send(backend.fail_status, {"error": "not now"})

                parts = url.path.strip("/").split("/")
                if len(parts) < 3 or parts[0] != "internal" or parts[1] != "datasets":
                    return self._send(404, {"error": "no such route"})

                dataset = backend.datasets.get(parts[2])
                if dataset is None:
                    return self._send(404, {"error": f"no dataset {parts[2]}"})

                if len(parts) == 3:
                    return self._send(200, dataset.view())
                if parts[3] != "documents":
                    return self._send(404, {"error": "no such route"})

                offset = int(query.get("offset", 0))
                limit = int(query.get("limit", 20))
                if limit > MAX_PAGE or limit < 1:
                    return self._send(422, {"error": f"limit must be 1..{MAX_PAGE}"})
                return self._send(
                    200,
                    {
                        "documents": dataset.documents[offset : offset + limit],
                        "offset": offset,
                        "total": len(dataset.documents),
                    },
                )

            def _send(self, status: int, body: dict):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://{}:{}".format(*self._server.server_address)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def pages(self) -> list[dict]:
        return [r for r in self.requests if r["path"].endswith("/documents")]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
