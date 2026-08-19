"""The seam to the Hugging Face Hub: what is out there, and how to get it.

Split by question rather than by object — `search` answers "what exists",
`sample` answers "what is in it", `fetch` answers "give me the rows". Nothing
here touches the database; the handler that calls them is what writes.
"""

from hub.fetch import (
    MESSAGE_COLUMNS,
    TEXT_COLUMNS,
    as_document,
    iter_rows,
    pick_column,
    shards_under,
)
from hub.sample import sample, splits
from hub.search import HubError, detail, search

__all__ = [
    "MESSAGE_COLUMNS", "TEXT_COLUMNS", "HubError", "as_document", "detail",
    "iter_rows", "pick_column", "sample", "search", "shards_under", "splits",
]
