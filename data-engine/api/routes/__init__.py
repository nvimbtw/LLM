"""One router per resource. `main` includes them and adds nothing of its own."""

from api.routes import datasets, demonstrations, hub, processors, runs, search_cache

ROUTERS = [
    hub.router,
    datasets.router,
    processors.router,
    runs.router,
    search_cache.router,
    demonstrations.router,
]

__all__ = ["ROUTERS"]
