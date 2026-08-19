"""One router per resource. `main` includes them and adds nothing of its own."""

from api.routes import datasets, hub, processors, runs

ROUTERS = [
    hub.router,
    datasets.router,
    processors.router,
    runs.router,
]

__all__ = ["ROUTERS"]
