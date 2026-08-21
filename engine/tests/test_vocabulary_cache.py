"""What keeps counting a token cheap enough to do while someone types.

`/tokenize/count` is called for every message stored and again on every pause in
the composer, and each call used to re-parse the whole `tokenizer.json`. These
are the assertions that keep it parsed once — and that keep the cache from
outliving the file it was read from.
"""

import os
import shutil

import pytest

from llm import storage
from llm.corpus import bpe
from tests.fixtures.tiny_env import TINY_MODEL


def test_the_same_vocabulary_is_parsed_once():
    """The whole point: a second call is free, so a count is an encode and not
    a multi-MB JSON parse."""
    model_dir = storage.model_dir(TINY_MODEL)

    first = bpe.load(model_dir)
    second = bpe.load(model_dir)

    assert first is second


def test_a_rewritten_vocabulary_is_read_again(tmp_path):
    """An SFT run saves its vocabulary over the base model's, in place. Keyed on
    the path alone, every later count would encode under the vocabulary that
    file used to hold.

    The mtime is set rather than left to the write, because a filesystem's clock
    is coarser than two writes in a row and the assertion would turn on it."""
    model_dir = tmp_path / "rewritten"
    model_dir.mkdir()
    path = storage.tokenizer_file(model_dir)
    shutil.copy(storage.tokenizer_file(storage.model_dir(TINY_MODEL)), path)

    before = bpe.load(model_dir)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert bpe.load(model_dir) is not before


def test_a_missing_vocabulary_still_raises(tmp_path):
    """The cache sits behind the existence check, not in front of it — a model
    with no tokenizer has to stay a 404 rather than a KeyError."""
    with pytest.raises(FileNotFoundError):
        bpe.load(tmp_path)
