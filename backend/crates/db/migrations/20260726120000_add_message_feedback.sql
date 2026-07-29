-- Thumbs up/down on an assistant reply.
--
-- This is the collection half of preference post-training: a label per reply is
-- exactly what a KTO-style objective consumes (unpaired desirable/undesirable),
-- and the benchmark's `preference` stage scores held-out rated replies to say
-- whether such a tune actually moved anything.
--
-- Nullable and two-valued rather than a NOT NULL three-valued column: NULL means
-- "not rated", which is the overwhelmingly common case, and it has to stay
-- distinguishable from a deliberate rating. Un-rating writes NULL back.
CREATE TYPE message_feedback AS ENUM ('up', 'down');

ALTER TABLE messages ADD COLUMN feedback message_feedback;

-- The export that feeds the benchmark asks for "every rated assistant reply",
-- which is a tiny fraction of the table — so the index only covers rated rows.
CREATE INDEX messages_feedback_idx ON messages (feedback) WHERE feedback IS NOT NULL;
