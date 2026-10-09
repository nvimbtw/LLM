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
sidebar entry is removed. `/profile` and `/usage` remain available as standalone
pages.

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

Everything else in `Settings` is stored on the account, not the browser —
`components/settings/settings-provider.tsx` fetches `GET /user/me/settings` on
login and writes back with `PUT` on every change, debounced 500ms so a dragged
slider or a typed system prompt isn't one request per tick. Logged-out state is
*derived* (`isAuthenticated ? settings : DEFAULT_SETTINGS`) rather than written
in the effect that notices the session changed, same reasoning as
`ChatsProvider`'s empty list — a synchronous `setState` in that effect is what
the `react-hooks/set-state-in-effect` lint rule catches. A fetched blob is
merged over `DEFAULT_SETTINGS`, so a setting added after an account last saved
comes back with its default rather than `undefined`. See
[`../backend/README.md`](../backend/README.md) for where it lives server-side.
