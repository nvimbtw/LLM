"""Bounded domain failures shared by source, resolution and model construction."""


class ArchitectureError(ValueError):
    """An actionable code/path without retaining custom objects or tensor values."""

    def __init__(self, code: str, message: str, path: str | None = None):
        """Record a stable failure code, bounded explanation and optional source/flow path."""
        super().__init__(message[:2000])
        self.code = code
        self.path = path
