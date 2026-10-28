# Data processing workbench

Open a dataset, add pipeline steps and select a step to inspect its parameters
and Python source. Catalogue forms retain declared types, defaults and ranges.
The overall before/after diff and `dropped_by` badge describe the pipeline's
outcome; there is no per-step trace.

The Data page's Processors link opens `/processors`. Each directory entry opens
`/processors/{processor_id}`, where the registered name is the stable identifier.
All signed-in users can inspect source. Administrators can edit complete custom
files in Monaco and save them; builtins stay read-only. Rust independently
enforces write permission.

Processors is a normal directory link beside the Datasets/Browse/Runs panel tabs.
Dataset entries use their whole main content as the open target. Detail pages
put the loaded dataset name, kind and status in the primary header, without
offering an unsupported rename operation. Administrator deletion uses the shared
[confirmation behavior](editing.md#navigation-catalogues-and-deletion).

`/processors/new-processor` is the dedicated administrator creation form, using
the same page/header primitives as project and training creation. It starts with
a source template; identifier, title, kinds, parameters and logic live in that
one file. A first version must load successfully. Invalid drafts remain in the
form; successful creation opens its detail page. Later edits retain the
save-as-supplied/last-working-version behavior.

Source shows its file and saved hash. After a failed reload it also shows the
active hash and error: bytes were saved, and preview continues with the last
working callback. Fix the source and save again. Successful HTTP saves refresh
the catalogue and automatically preview the same first eight stored documents,
including saves that produced a reload error.

`use-processor-source.ts` owns requests, draft state and cancellation for one
processor/session. Switching identity aborts requests and ignores stale save
completions. A workbench is keyed by dataset and session; its hook owns ordered
steps, a 250 ms preview debounce, cancellation and source refreshes. Old preview
responses cannot replace a newer pipeline or source revision. Pipelines have a
32-step budget matching the API.

Monaco 0.52.2 and its React adapter are pinned in `bun.lock`. Dev/build copies
the package's editor assets and license into ignored `public/monaco/` via
`scripts/prepare-monaco.mjs`. The loader uses this local path; source editing
requires no external editor CDN.

Compose keeps generated editor assets in the `frontend-monaco` volume, alongside
the separate Next build volume, so container writes do not change host asset
ownership. The tracked empty asset directory keeps native checkout ownership;
preparation replaces its contents without deleting the mount point.

After adding these dependencies in Docker development, rebuild from the repo root:

```sh
docker compose up -d --build --no-deps --force-recreate --renew-anon-volumes frontend
```

The Compose file masks host dependencies with an anonymous `/app/node_modules`
volume. Renewing it loads dependencies from the rebuilt image; ordinary recreation
preserves volumes. See [Compose up](https://docs.docker.com/reference/cli/docker/compose/up/).

Verification includes frontend checks plus browser coverage of read-only source,
valid save/repreview, invalid save/old-active preview, and the existing diff/drop
indicators. The backend's separate tests check read authorization and admin-only
writes. The complete wire contract is in
[processor authoring](../data-engine/processors.md).
