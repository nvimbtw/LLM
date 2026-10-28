# Notebook editing and resource actions

Resource names and descriptions live in the page header. Administrators edit
model, vocabulary and custom architecture/block metadata there; projects remain
private to their owning account. Builtin architecture metadata and ordinary
model/vocabulary readers remain read-only. Dataset names identify the page but
are not editable because the domain has no rename operation.

`EditableTitle` supplies the page-native title field and an accessible label.
Titles and other inline text fields use caret-only editing, with no focus
underline, outline, ring or adjacent pencil. `Textarea variant="inline"` keeps
descriptions and project notes unboxed. It grows with multiline input and uses a
measured fallback when `field-sizing: content` is unavailable; resizing the page
recalculates the fallback height.

`EditActions` supplies **Unsaved changes**, **Save changes** and **Cancel**, with
one size and emphasis. Saving disables the controls, failed writes retain the
attempt, and Cancel restores the current stored values. Creation keeps its
specific Create action. Architecture metadata and source drafts have separate
owners and explicit save actions: saving/cancelling metadata never replaces
source bytes, and reloading source never discards metadata edits.

## Draft recovery

Project, model, vocabulary and architecture metadata fields, plus unsaved
architecture/block source, are retained in `sessionStorage`. The key combines a
hash of the opaque bearer with the resource identity; credentials are not stored
in the key or draft. A draft stays in this browser tab across in-app navigation,
Back and reload. Replacement sessions cannot read the earlier session's draft.
Closing the tab ends this recovery lifetime. These field drafts are distinct from
the account-scoped training draft in [Jobs](jobs.md).

Writes are synchronous before subsequent navigation. Versioned payloads are
bounded and decoded before restoration. A changed server baseline preserves the
attempt with a review warning rather than silently discarding it. Confirmed
saves and explicit Cancel remove the retained copy. Storage failures remain
visible and request a loss confirmation for links/browser unload when dirty
fields cannot be retained. Ordinary navigation has no repetitive prompt while
recovery is working. Native browser Back under a storage failure is not
intercepted; the visible storage error remains the warning in that case.

Source recovery retains exact invalid Python/YAML, independent of optimistic
server save versions and validation receipts. A reload of the server source
explicitly discards its retained local copy. Navigating during a pending write
does not claim that the server rolled it back; returning reconciles against the
current stored baseline.

## Conversation titles

The composer, archive and sidebar use `ConversationRename`. The composer title
is a labelled text button; archive/sidebar menus expose Rename.
The inline editor has the same **Save changes** / **Cancel** controls. Enter
saves, Escape cancels, IME composition does not submit, and blur keeps editing.
A pending save cannot be duplicated. Failed writes leave the attempted title
and error visible for retry; they never imply a saved title. Leaving the editor
without saving or choosing Cancel abandons that title attempt.

## Navigation, catalogues and deletion

The signed-in frame has a compact, labelled mobile navigation opener; desktop
keeps its existing sidebar layout. Data's Processors destination is a normal link
beside the panel tabs. Dataset detail uses the loaded resource name, kind and
status in its primary header, with a back link during loading and errors.

Catalogues use open single-column entries with serif resource names, generous
main links and separate secondary actions. Gym names/descriptions open practice;
**Evaluate and view history** opens the source's evaluation/history page. General
stays first, with the same presentation as the other gyms.

`DeleteConfirmation` states the resource-specific consequences, uses **Cancel**
and a specific destructive action such as **Delete dataset**, and keeps pending
and failure feedback inside the confirmation. Failure permits retry; navigation
or removal happens only after acknowledgement. Project deletion keeps its
existing equivalent confirmation and preserves conversations. Dataset, model and
vocabulary delete controls remain administrator-only. Run deletion confirms
permanent history removal. Destructive controls use the shared theme-aware
variant, without hard-coded white text.

Quiet overflow actions remain visible on small screens and coarse pointers.
Attachment removal has a 32px hit area and a neutral keyboard focus fill.
Archive conversation menus use a 40px target. None depends on hover to be found
on touch screens.

## Verification

From `frontend/`, run typecheck, lint, unit tests and the production build. With
a running frontend, use `NOTEBOOK_URL=http://localhost:3000
CHROMIUM_PATH=/usr/bin/chromium node scripts/check-notebook-browser.mjs` for
fixture-driven routing, persistence, failed writes, focus, touch emulation and
multiline fallback checks. Fixtures do not certify service-side execution or
physical touch hardware; those remain separate checks.
