"""One router per resource. `main` includes them and adds nothing of its own."""

from api.routes import (
    architectures,
    attention,
    inference,
    measure,
    model_definitions,
    models,
    runs,
    tokenizers,
    validations,
)

ROUTERS = [
    architectures.router,
    validations.router,
    runs.router,
    models.router,
    model_definitions.router,
    attention.router,
    tokenizers.router,
    inference.router,
    measure.router,
]

__all__ = ["ROUTERS"]
