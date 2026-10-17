"""ByteLevel streaming must agree with whole-reply decoding at Unicode boundaries."""

import random
import unittest
from pathlib import Path

from tokenizers import Tokenizer, decoders

from llm.corpus import bpe


TOKENIZER_FILE = Path(__file__).parent / "fixtures" / "tiny" / "tokenizer.json"


class StreamDecoderTests(unittest.TestCase):
    """Exercise byte fragments, added markers and unsupported decoder fallback."""

    def setUp(self) -> None:
        """Load an independent pinned ByteLevel vocabulary for each case."""
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    def assert_decodes(self, ids: list[int]) -> None:
        """Compare concatenated incremental fragments with canonical decoding."""
        decoder = bpe.stream_decoder(self.tokenizer)
        self.assertIsNotNone(decoder)
        fragments = [decoder.push(token_id) for token_id in ids]
        self.assertEqual("".join(fragments) + decoder.finish(), bpe.decode(self.tokenizer, ids))

    def test_unicode_and_special_boundaries(self) -> None:
        """Complete Unicode arrives intact and unfinished bytes flush at markers."""
        for text in ("Café 😀 Привет 你好", "a\u0301\u200d👩‍💻", "<|assistant|>hello<|end|>"):
            with self.subTest(text=text):
                self.assert_decodes(self.tokenizer.encode(text).ids)
        lead = self.tokenizer.token_to_id("Ã")
        continuation = self.tokenizer.token_to_id("©")
        marker = self.tokenizer.token_to_id("<|endoftext|>")
        self.assertIsNotNone(lead)
        self.assertIsNotNone(continuation)
        self.assertIsNotNone(marker)
        self.assert_decodes([lead, continuation])
        self.assert_decodes([lead, marker])
        self.assert_decodes([lead])

    def test_vocab_and_generated_sequences(self) -> None:
        """All fixture IDs and mixed random sequences match the library decoder."""
        ids = list(self.tokenizer.get_vocab().values())
        for token_id in ids:
            self.assert_decodes([token_id])
        rng = random.Random(8)
        for _ in range(1000):
            self.assert_decodes(rng.choices(ids, k=rng.randint(2, 40)))

    def test_unsupported_decoder_uses_prefix_fallback(self) -> None:
        """A different decoder cannot enter the ByteLevel streaming path."""
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.assertIsNone(bpe.stream_decoder(self.tokenizer))


if __name__ == "__main__":
    unittest.main()
