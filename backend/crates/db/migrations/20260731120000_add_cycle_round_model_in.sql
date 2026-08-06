-- Which model generated the round's catalogue.
--
-- `id_model_out` records what a round produced; this records what it started
-- from, and it is not the same as `cycles.id_model_current` by the time the
-- round's second training stage runs — RAFT has replaced that with its own
-- output. Both stages must train on the *same* slice of the catalogue, so the
-- generating model has to be remembered rather than re-read.
--
-- Nullable, and left NULL for rounds that ran before this column existed: their
-- catalogue lines carry no provenance stamp either, so there is nothing to
-- reconstruct it from and a guess would silently scope a run to the wrong data.
ALTER TABLE cycle_rounds
    ADD COLUMN id_model_in UUID REFERENCES models(id_model) ON DELETE SET NULL;
