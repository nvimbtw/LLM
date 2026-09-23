"""Pipelines. Each orchestrator runs one job end to end, step by step, calling
into the layers below it. No pipeline logic lives anywhere else — the API layer
parses a request and calls one of these; the layers below know nothing about
each other's order.
"""
