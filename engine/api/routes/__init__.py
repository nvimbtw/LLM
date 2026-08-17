"""One router per resource. `main` includes them and adds nothing of its own."""

from api.routes import (
    benchmarks,
    catalogue,
    chats,
    inference,
    models,
    runs,
    tokenizers,
    usage,
)

ROUTERS = [
    runs.router,
    chats.router,
    models.router,
    tokenizers.router,
    benchmarks.router,
    inference.router,
    catalogue.router,
    usage.router,
]

__all__ = ["ROUTERS"]
