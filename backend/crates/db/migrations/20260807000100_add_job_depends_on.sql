-- What a `waiting` job is waiting for: the job whose model it will consume.
--
-- Unlike `ref_id` — which points into three different tables and is therefore
-- deliberately not a foreign key — this always names another row in this table, so
-- it is a real reference.
--
-- ON DELETE SET NULL rather than CASCADE. A chain's children outlive their parent:
-- deleting a finished pretrain must not erase the fine-tune it produced from the
-- history, which is exactly what CASCADE would do. The cost is that a *waiting*
-- child can be orphaned, and that is handled rather than prevented — `jobs::chain`
-- cancels a waiting row whose dependency has gone NULL, with a reason saying so,
-- because a job whose base model will now never exist can only fail later and more
-- expensively.
ALTER TABLE jobs
    ADD COLUMN depends_on UUID REFERENCES jobs(id_job) ON DELETE SET NULL;

-- The resolver's only lookup: "what waits on the job that just finished". Partial,
-- because the column is NULL for every ordinary job and those rows have nothing to
-- contribute to that question.
CREATE INDEX idx_jobs_depends_on ON jobs (depends_on) WHERE depends_on IS NOT NULL;
