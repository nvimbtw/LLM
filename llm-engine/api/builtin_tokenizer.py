"""Publish and index the real byte-alphabet tokenizer used by builtin recipe acceptance examples."""

import hashlib
from pathlib import Path
import shutil
import tempfile
from uuid import NAMESPACE_URL, UUID, uuid5

import store
from llm import storage
from llm.architecture.package_integrity import (
    MANIFEST_BYTES, decode_json, encode_json, package_file, publish_directory, read_bytes, sync_directory, write_bytes,
)
from llm.architecture.spec import FinalTokenizer
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES, binding_from_bytes
from llm.corpus import baseline, vocab
from store import architecture_queries as catalog, queries


def baseline_bytes() -> tuple[UUID,bytes]:
    """Identify exact real tokenizer bytes; changed reserved labels or tokenizer serialization receive a new artifact UUID."""
    raw = baseline.byte_alphabet().to_str().encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return uuid5(NAMESPACE_URL,f"slate:builtin-byte-alphabet-v1:{digest}"),raw


def verify_existing(binding: FinalTokenizer, raw: bytes) -> None:
    """Require complete exact previously published bytes/marker, never replacing an unrelated or partially written artifact."""
    directory = storage.tokenizer_build_dir(str(binding.tokenizer_id))
    metadata = decode_json(read_bytes(package_file(directory,"build.json"),MANIFEST_BYTES))
    actual = read_bytes(package_file(directory,"tokenizer.json"),TOKENIZER_BYTES)
    if actual != raw or not isinstance(metadata,dict) or metadata.get("source") != "builtin-byte-alphabet-v1":
        raise catalog.CatalogError("identity_conflict","builtin tokenizer identity has conflicting artifact bytes")
    if metadata.get("artifact_sha256") != binding.fingerprint or metadata.get("vocab_size") != binding.vocabulary_size:
        raise catalog.CatalogError("identity_conflict","builtin tokenizer publication metadata is inconsistent")


def publish_baseline(binding: FinalTokenizer, raw: bytes) -> None:
    """Own staging/fsync/rename under exclusive startup; exact complete orphans survive a later index failure."""
    destination = storage.tokenizer_build_dir(str(binding.tokenizer_id))
    root = storage.tokenizer_staging_dir()
    if root.exists():
        shutil.rmtree(root)
    if destination.exists():
        verify_existing(binding,raw)
    else:
        root.mkdir(parents=True,exist_ok=True)
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="builtin-byte-alphabet-",dir=root))
        metadata = {"tokenizer_id":str(binding.tokenizer_id),"source":"builtin-byte-alphabet-v1",
            "artifact_sha256":binding.fingerprint,"fingerprint":binding.fingerprint[:vocab.FINGERPRINT_CHARS],
            "vocab_size":binding.vocabulary_size,"max_vocab":binding.vocabulary_size,"merges":0,
            "datasets":[],"sampled_chunks":0,"sampled_bytes":0,"sample_bytes":0,
            "special_tokens_sha":vocab.special_tokens_sha()}
        try:
            write_bytes(staging/"tokenizer.json",raw)
            write_bytes(staging/"build.json",encode_json(metadata))
            sync_directory(staging)
            publish_directory(staging,destination)
            sync_directory(destination.parent)
            sync_directory(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def ensure_baseline() -> FinalTokenizer:
    """Publish and index a real example tokenizer, never substituting it for pending training bindings.

    The caller holds exclusive reconciled startup ownership. Complete filesystem
    orphans are adopted; conflicts fail and existing display edits survive.
    """
    tokenizer_id,raw = baseline_bytes()
    binding = binding_from_bytes(tokenizer_id,raw)
    publish_baseline(binding,raw)
    with store.session() as db:
        existing = queries.get_tokenizer(db,tokenizer_id)
        if existing is None:
            queries.upsert_tokenizer(db,tokenizer_id=tokenizer_id,name=f"Byte alphabet {binding.fingerprint[:8]} (builtin)",
                description="Byte-complete tokenizer with no learned merges, used for builtin validation examples.",
                vocab_size=binding.vocabulary_size,merges=0,max_vocab=binding.vocabulary_size,datasets=[],implicit=True,
                fingerprint=binding.fingerprint[:vocab.FINGERPRINT_CHARS])
        elif existing.vocab_size != binding.vocabulary_size or existing.fingerprint != binding.fingerprint[:vocab.FINGERPRINT_CHARS]:
            raise catalog.CatalogError("identity_conflict","builtin tokenizer index differs from its published artifact")
    return binding
