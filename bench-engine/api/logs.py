import logging
def quieten() -> None:
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
