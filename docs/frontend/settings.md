# Settings and profile

The bottom-left profile button, Ctrl/⌘+K, and `/settings` open the same modal.
The profile button includes the ⌘K hint and opens Profile directly without
leaving the current page; there is no separate sidebar Settings button. The
left rail selects Appearance, Chat, Decoding, Usage, or Profile. Only the
selected section is shown on the right. The keyboard shortcut and slash
command reopen the current section.

Collapsed, the profile entry shows only its centered icon; the username,
loading placeholder, and shortcut hint are hidden. Its tooltip and accessible
name still identify the settings entry.

The rail stays visible while the content scrolls, including on narrow screens.
The selected section has a thin rule on its left, matching conversation
navigation. The dialog uses display headings and ruled account
fields instead of stacked cards. Reset to defaults is within the preference
sections; the close control stays available at the top right. Section tabs
support arrow-key navigation through the shared UI primitive. Account details,
password changes, account deletion confirmation, and logout
are available in Profile; successful logout or deletion closes the modal. Usage
has its own section with totals, the token chart, and the model breakdown. Its
sidebar entry is removed. Profile and Usage are available only in this modal;
the retired `/profile` and `/usage` page URLs return 404. Callers open the
matching tab through `openSettings("profile")` or `openSettings("usage")`.

`SettingsProvider` owns modal visibility and the selected section; section
selection is session UI state and is not saved with preferences. Profile and
Usage are available only to signed-in users.

`lib/settings.ts` is the single source of truth. One entry in `SETTINGS` (plus a
field on the `Settings` type) gives you a control in the ⌘K modal and, for a
toggle, a slash command — nothing else needs touching.

- An entry names a **`group`**, and `GROUPS` declares the sections in the order they
  appear. That is the only placement decision adding a setting requires.
- Sections and setting rows show their titles and controls without explanatory
  subtitles. `GROUPS` contains only section identifiers and labels.
- Theme is deliberately **not** a `SETTINGS` entry: next-themes owns and
  persists it (the no-flash script reads it before React exists), so declaring
  it here would give the app two answers to what the theme is. It gets a
  hand-written row in the `appearance` section that reads and writes the real one.

**Use live throbber** is an Appearance toggle, off by default, also exposed as
`/live-throbber on|off`. It selects the animation while preserving status text.
See [Live throbber](live-throbber.md) for capability fallback and the token contract.

Everything else in `Settings` is stored on the account. `SettingsProvider` uses
`useSessionToken()` from `hooks/use-auth.ts` to scope its loaded values, pending
writes and responses to the actual browser session identity. An unmatched or not
yet loaded session shows defaults. A failed settings GET does not authorize writes
over unknown account data; preference mutations stay disabled until a successful
read for that identity. Switching accounts, including direct token replacement,
loads the replacement account rather than treating two authenticated sessions as
one identity.

Changes write back with PUT, debounced 500ms by `lib/settings-writer.ts`, so a
slider or system prompt does not send one request per tick. The complete next value
is calculated before scheduling; React state updaters never start timers or writes.
Logout or token replacement discards the prior session's pending work, and late
responses are ignored. Unmounting with the same token flushes its pending value.
`lib/account-settings.ts` validates known preference fields and merges older blobs
over current defaults. Invalid response containers fail the read rather than
turning an unknown account copy into writable defaults. A later failed PUT leaves
local values intact; the next change retries the current complete value.

See [`../backend/README.md`](../backend/README.md) for account storage.
