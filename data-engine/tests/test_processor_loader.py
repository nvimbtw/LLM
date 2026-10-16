"""Custom imports, atomic publication, invalid edits and active source hashes."""

import hashlib
import time

import pytest

from processors.loader import Catalog, Watcher

SOURCE = '''from processors.helpers import processor
@processor(name="example", title="Example", kinds=["pretrain", "sft"])
def run(text, p, ctx):
    """Uppercase input text."""
    return text.upper()
'''


def test_load_hash_and_bad_edit_keep_last_working_generation(tmp_path):
    """Failed edits preserve both active callback behavior and its original hash."""
    path = tmp_path / "example.py"
    path.write_text(SOURCE)
    catalog = Catalog({}, tmp_path)
    original = catalog.snapshot()
    assert (
        catalog["example"].version_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    path.write_text("def broken(")
    catalog.reload()
    assert catalog["example"] is original["example"]
    assert "SyntaxError" in catalog.error(path)
    assert catalog["example"].apply({"content": "hi"}, {}, {}) == {"content": "HI"}
    path.write_text(SOURCE.replace("text.upper()", "text.lower()"))
    catalog.reload()
    assert catalog["example"].version_hash != original["example"].version_hash
    assert catalog.error(path) is None
    assert original["example"].apply({"content": "hi"}, {}, {}) == {"content": "HI"}


def test_startup_collisions_raise_and_reload_collision_keeps_complete_registry(
    tmp_path,
):
    """Duplicates cannot publish a partial catalogue alongside existing processors."""
    (tmp_path / "one.py").write_text(SOURCE)
    catalog = Catalog({}, tmp_path)
    (tmp_path / "two.py").write_text(SOURCE)
    with pytest.raises(ValueError, match="two processors named"):
        Catalog({}, tmp_path)
    with pytest.raises(ValueError, match="two processors named"):
        Catalog({"example": catalog["example"]}, tmp_path)
    previous = catalog.snapshot()
    catalog.reload()
    assert catalog.snapshot() is previous
    assert "two processors" in catalog.error(tmp_path / "two.py")


@pytest.mark.parametrize(
    "prefix", ["import store\n", "from store import schema\n", "from . import other\n"]
)
def test_bad_import_is_quarantined_without_losing_good_files(tmp_path, prefix):
    """The import policy excludes storage and relative imports before execution."""
    (tmp_path / "good.py").write_text(SOURCE)
    bad = tmp_path / "bad.py"
    bad.write_text(prefix + SOURCE.replace('name="example"', 'name="bad"'))
    catalog = Catalog({}, tmp_path)
    assert set(catalog) == {"example"}
    assert "ImportError" in catalog.error(bad)


def test_watcher_reloads_stable_edits_and_joins_on_stop(tmp_path):
    """Lifespan-owned polling observes an edit and leaves no worker after shutdown."""
    path = tmp_path / "example.py"
    path.write_text(SOURCE)
    catalog = Catalog({}, tmp_path)
    watcher = Watcher(catalog)
    watcher.start()
    try:
        path.write_text(SOURCE.replace("text.upper()", "text.lower()"))
        deadline = time.monotonic() + 3
        while catalog["example"].apply({"content": "HI"}, {}, {})["content"] != "hi":
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        watcher.stop()
