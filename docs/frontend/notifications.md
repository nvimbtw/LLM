# Notifications

`ToastProvider` owns notification delivery and the center's modal. Existing callers
keep `toast(options)`, `toast.success`, `toast.error`, and `dismissToast(id?)`.
`type` selects severity; `lifetime` independently selects retention:

| Lifetime | Popup | History |
| --- | --- | --- |
| `sticky` | Until dismissed | Until dismissed |
| `timed` | Until its timeout | Until dismissed |
| `transient` | Until its timeout | Never retained |

Success defaults to transient for four seconds, ordinary and warning notices to
timed for six seconds, and errors to sticky. An explicit lifetime or timeout can
override these defaults; timeout zero selects sticky when no lifetime is supplied.
Popup expiry never dismisses retained history. Explicit dismissal removes an entry
from both views. Restoring history reopens sticky popups, while timed entries
remain in history. The Bell counts retained undismissed entries, including ones
already viewed; opening the center does not mark or dismiss anything.

`NavNotifications` supplies the separate sidebar footer entry. Ctrl+Alt+N opens
the modal. It and Settings share `components/ui/account-modal.tsx`: dimensions,
border, background, rail spacing, scrolling content, and vertical section marker.
Base UI supplies dialog focus management and accessible toast announcements,
close controls and swipe dismissal. Glass toasts retain the composer's translucent
card material, blur and saturation with the shared crisp overlay shadow.

## Account and storage ownership

`lib/notification-store.ts` owns history, popup visibility and one expiry timer.
Severity/lifetime defaults and storage validation live in `lib/notifications.ts`.
The store holds at most 100 entries. Stored text is bounded and versioned; only
retained plain text is serialized. ReactNode titles and descriptions retain their
live rendering, while textual children form their durable history representation.
Transient events never enter storage.

The actual bearer captured by `useSessionToken` is verified through `/user/me`.
Only a successful response with a durable `id_user` UUID permits account storage
access, using `slate-notifications:v1:<account UUID>`. A replacement token masks
old records immediately, clears previous popups, and ignores late profile results.
A different bearer for the same verified account restores the same history.
Logged-out events remain separate, memory-only state and are cleared on login.
No account history is stored on the backend or synchronized across devices.

A failed identity or history read never authorizes overwriting unknown history.
Invalid history leaves its stored bytes intact. Storage failures keep live entries
in memory, show a notification plus an actionable center error, and expose Try
again. A failed write retries the complete retained snapshot; a failed read or
identity lookup re-verifies and reloads before writing. The store deduplicates
explicit event keys and bounds both history and timer ownership.

`FormError` and `LoadError` route explanations through this system. Form errors
retain screen-reader focus text and existing caller-owned invalid-field state;
load errors keep retry controls at the failed content location. Mounted error
episodes emit once, including under StrictMode and ordinary rerenders. Clearing
an error starts a new episode. Standalone fixtures without ToastProvider retain
visible fallback explanations.

Continuously derived pre-submit constraints remain neutral guidance beside the
form so disabled actions stay understandable; they do not emit a notification on
initial render or each keystroke. Request and submission failures use feedback.
Password confirmation mismatch retains invalid-field association and focuses the
confirmation control. A failed Usage read leaves a Try again action in its section.

Verify lifecycle, expiry/dismissal, account replacement, opaque-session identity,
storage failures, retry and bounds with `bun test lib/notifications.test.ts` from
`frontend/`. Real browser verification remains necessary for announcement timing,
focus return, keyboard shortcuts, swipe dismissal and modal layout.
