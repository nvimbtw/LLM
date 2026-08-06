-- Models start private and an admin publishes them. Default false so every
-- existing and newly-trained model is private until explicitly published.
ALTER TABLE models ADD COLUMN published BOOLEAN NOT NULL DEFAULT false;
