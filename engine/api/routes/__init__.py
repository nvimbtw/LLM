"""One router per resource. `main` includes them and adds nothing of its own."""

from api.routes import catalogue, chats, inference, models, runs, usage

ROUTERS = [
    runs.router,
    chats.router,
    models.router,
    inference.router,
    catalogue.router,
    usage.router,
]

__all__ = ["ROUTERS"]
