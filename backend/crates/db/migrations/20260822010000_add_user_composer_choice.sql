-- What the composer is set to send with — the model and the thinking budget —
-- as its own JSONB blob, not folded into `users.settings`. The two persist on
-- different rhythms (a model/budget pick fires immediately; a settings change
-- debounces) and a shared column would mean the later of two independent
-- writes clobbering the other's field. Opaque here for the same reason
-- `users.settings` is: the frontend shape can grow without a migration.
ALTER TABLE users ADD COLUMN composer_choice JSONB NOT NULL DEFAULT '{}'::jsonb;
