-- Replies generated purely to be rated, and the ratings people gave them.
--
-- A KTO tune needs thumbs, and thumbs only appear as a by-product of chatting —
-- which means a fresh model can never be preference-tuned, because nobody has had
-- enough conversations with it to have rated anything. That is not a data problem
-- to wait out; it is a missing step in the flow. A gather job generates short
-- replies to real prompts and parks them here unrated, and the rating panel walks
-- through them.
--
-- Deliberately NOT rows in `messages`. A gathered reply is not part of any
-- conversation: it has no chat, no ordering among other turns, and its prompt is
-- one field of the same row rather than "the most recent user turn before it".
-- Forcing it into `messages` would mean a synthetic chat per batch and a LATERAL
-- lookup that happens to resolve — a shape that reads as a conversation, isn't one,
-- and would show up in the sidebar.
--
-- `feedback` reuses the `message_feedback` enum rather than declaring a parallel
-- one: it is the same verdict, and `queue::export_preference_data` unions the two
-- sources into one training file, which is only sound if "up" means the same thing
-- on both sides.
CREATE TABLE signal_samples (
    id_sample  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- The batch this came from. SET NULL rather than CASCADE: deleting a finished
    -- job from the board must not silently destroy ratings that are already part of
    -- the training data — the export would then shrink for no visible reason.
    id_job     UUID REFERENCES jobs(id_job) ON DELETE SET NULL,
    -- Which model wrote it. Preference data is on-policy for the model that
    -- produced it, so a tune wants to know whose replies it is learning from.
    id_model   UUID REFERENCES models(id_model) ON DELETE SET NULL,
    prompt     TEXT NOT NULL,
    response   TEXT NOT NULL,
    -- NULL = generated but not yet rated, which is every row's initial state and
    -- the thing the rating panel queries for. Same two-valued/nullable reasoning as
    -- messages.feedback.
    feedback   message_feedback,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    rated_at   TIMESTAMPTZ
);

-- The rating panel's read: "what is still unrated", oldest first so a batch is
-- worked through in the order it was generated.
CREATE INDEX signal_samples_pending_idx ON signal_samples (created_at)
    WHERE feedback IS NULL;

-- The export's read, and the readiness count the KTO composer gates on.
CREATE INDEX signal_samples_rated_idx ON signal_samples (feedback)
    WHERE feedback IS NOT NULL;

-- A fifth kind of GPU work. Like autolabel it generates candidate replies, but it
-- does not judge them — a person does, which is the entire point: these are the
-- signals the rule-based judge cannot produce.
--
-- Same in-transaction caveat as the autolabel migration; IF NOT EXISTS makes it
-- re-runnable.
ALTER TYPE job_type ADD VALUE IF NOT EXISTS 'gather';
