# Gyms

## Two doors to a gym, and only one of them is a loop

The General gym is offered by the catalogue and mixes source questions across
episodes. Evaluation and practice forms expose source weights; task selection
still defaults to easy rungs. A composite gym's tools are decided per episode,
so its empty catalogue tool list does not force practice to one round. Math
episodes remain tool-free. General evaluations and mixed training reports show
scores by source alongside the overall result. Demonstrations are available
for every catalogue entry declaring that capability, including General. The
catalogue places General first in the same single-column open list as the source
gyms, using shared open rows without decorative separators. Name/description links open practice and
the separate **Evaluate and view history** link opens evaluation and prior runs.
The General RL job panel also shows the latest task-source sequence and a chart
per source. Those charts accumulate correct answers only when that source appears
in a batch; their adjacent correct/graded counts and percentage keep the count
from being mistaken for a pass-rate trend. The existing overall pass-rate and
signal chart remains below them. Completed runs read the compact history saved
in the RL result, while running jobs use live diagnostic frames.


The math practice composer defaults to automatic progression across all rungs.
Turning it off permits manual task selection, including which demonstrations
to generate. Demonstrations create or append an SFT dataset; they never start
a training job. A math episode has one response and no tools. The job page
shows per-skill mastery, promotion/review evidence and fixed before/after test
scores, read from the saved result after completion. Training pass rate is
labelled separately because its difficulty changes. See
[math practice](../llm-engine/math-practice.md).

`/gyms/sources/<gym slug>` composes **one** evaluation: the browser drives it over the socket below,
the episodes are graded, and it ends. Nothing it scores reaches the weights —
it is a measurement.

The loop is a **job**, composed at `/gyms/new-gym`
(`components/jobs/rl-composer.tsx`) and watched on the job page like every other
run. Each of its iterations is a gym run of its own, created by the LLM engine
rather than the browser and kept off the source evaluation history (`kind: "rl"`), so a
fifty-iteration run does not bury the evaluations anyone made.

That distinction is invisible from the run form, which is why the form links to
the composer in as many words: the shape of the question "why did my gym only
run once" is someone standing at the first door looking for the second.

**The catalogue leads with the loop.** Each entry at `/gyms` opens
`/gyms/new-gym?gym=<id>` — practising is what anyone comes to a gym for, and it
is one click from the list. The evaluation door is the card's second, quieter
link. The gym id travels as a search param read by the *page* (a server
component) and handed to the composer as a prop, rather than `useSearchParams`
inside it, which would put the page behind a Suspense boundary to learn one
string; an id no gym answers to falls back to the first, since the value comes
off a URL anyone can type.

Both practice execution UUIDs and evaluation UUIDs open at `/gyms/{uuid}`. The
route first checks the durable execution; only a 404 permits evaluation lookup.
It mounts that workflow's existing renderer and socket owner. Source slugs and
old nested evaluation URLs redirect with their query intact. See
[routing](routing.md); HTTP and socket API paths retain their existing contracts.

## Stream protocol (a gym run)

One socket per run, opened at `/ws/gyms/runs/{id}` (`lib/gym-socket.ts`), and
deliberately the chat vocabulary with four frames added:

```
server → { type: "episode_start", episode_id, index, task_type, prompt }
server → { type: "round_done",   episode_id, text, usage }
server → { type: "episode_done", episode_id, score, correct, answer, ground_truth }
server → { type: "run_done",     score, correct, graded, episode_count }
```

Two differences from chat, and both are the reason it is a socket of its own.
The LLM engine's `done` arrives as **`round_done`**, because an episode takes a
round per command and a client counting `done` would see an episode end every
time the model ran `ls`. And every frame carries its `episode_id`, so a run of
twenty tasks is one connection rather than twenty.

The socket carries only the sampling — the model and the tasks were fixed when
the run was created — and **closing it abandons the run**: episodes already
graded are stored, the rest are never run.

`context_multiplier` is in that sampling for a reason particular to gyms: a task
plus its system block is ~70 tokens before the shell has answered anything, and
half the window is reserved for the reply, so a model trained under 256 cannot
hold a task unless the window is stretched. `RunForm` says so before a run is
started rather than after it scores zero.

## Reading a gym episode

`components/gyms/episode-transcript.tsx` renders an episode with
`components/chat/trace.tsx` and `components/chat/rich-message.tsx`, composed the
way `chat-message.tsx` composes them: the task is a user bubble, the rounds that
called the shell are the trace, the round that stopped calling is the reply.

That reuse is the point rather than a saving. What a gym shows has to be what
the model actually produced — the same markup, the same reasoning panel, the
same command output in the same code block — because the model is doing in the
gym exactly what it does in a conversation. A renderer of its own here would be
a second opinion about that, and the first time the two disagreed the gym would
be the one lying.

`hooks/use-gym-run.ts` is what makes a live run and a reloaded one the same
shape: it builds a `Transcript` from frames while they arrive and from stored
steps on a reload, so the page below never learns which it is looking at.
