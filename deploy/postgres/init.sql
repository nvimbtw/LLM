-- Runs once, on an empty data volume, via the postgres image's
-- docker-entrypoint-initdb.d hook. Delete the volume to re-run it.
--
-- Three databases because each module owns one and none reads another's — see
-- docs/decisions/. One *server* rather than three is a local-development
-- convenience only: the ownership boundary is enforced by the applications,
-- which are each handed a URL naming exactly one of these.
--
-- POSTGRES_DB creates the backend's, so only the other two are named here.
CREATE DATABASE slate_engine;
CREATE DATABASE slate_data;
CREATE DATABASE slate_gyms;
