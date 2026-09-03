# Demonstrations — what the gym teaches, rather than measures

Split out of [`README.md`](README.md) when it passed the ~250-line rule (see
[`../README.md`](../README.md)). The README keeps the map; this file keeps the
expert trajectory and the task variety it depends on. Everything about what a
gym *measures* — the tasks, the rewards, the grading, the routes — is still
there.

## Demonstrations: the trajectory a model should have produced

`gyms/bash/expert.py` writes the correct trajectory for a task — the commands,
and the answer alone as the last turn. It costs nothing to produce because the
generator already holds the ground truth, so the gym can demonstrate itself:
behaviour cloning here needs no collected dataset and no larger model.

**What it teaches is not how to use a shell.** The first measured runs showed a
model that drove the shell competently enough to put the answer on screen in 4
of 20 episodes and then replied "let me run the script again" — all twenty final
replies were search narration. Nothing it had been trained on ended with a bare
answer. So the *shape* is the lesson: the last turn is the answer, and it is the
turn SFT computes loss on.

Two rules keep a demonstration honest, and `tests/test_expert.py` enforces both
against the real sandbox rather than the in-process fake:

- **Every command must run, and the evidence must actually appear in what comes
  back.** The first version searched with `grep -rl`, which prints the filename
  alone — so the needle never reached the transcript, and `SAW_TARGET` would not
  have fired on the one trajectory that is meant to be exemplary. The shaping
  and the demonstration have to agree about what progress is.
- **The answer must be readable off the transcript.** A demonstration whose
  final answer cannot be derived from what it was shown teaches guessing.

**A batch shares one workspace.** `POST /gym/trajectories` seeds, plays and
clears one sandbox session per request rather than a filesystem per task — the
sandbox keeps a bounded number and cloning wants thousands. An *episode* may
never do this: its id is its filesystem, which is the invariant the key exists
for.

**Nothing here renders an observation**, and that is what makes the output
trainable rather than merely plausible. The route answers with what the shell
said; `tools::observation` in Rust renders the body, and
`llm.chat.tools.render_result` in the engine wraps it in the block — the single
renderer for both the live path and the SFT data. A Python copy of either would
train the model for a shell it is never served.

## From a trajectory to a corpus

A demonstration becomes an **ordinary `sft` dataset in the data engine**, and
the engine's existing SFT build reads it with no new reader, no new key and no
new corpus path. That is not a convenience: the engine owns no corpora — a run
names dataset ids or it is refused — so a demonstration set living beside one
would be the engine owning a corpus again, which
[`../decisions/datasets-out-of-the-engine.md`](../decisions/datasets-out-of-the-engine.md)
exists to prevent. It also means demonstrations can be *mixed* into one
fine-tune with a chat corpus, which matters: three task types on their own train
a model that only answers gym questions.

```
POST /gym/trajectories        gyms — plays the trajectory, renders nothing
        ↓ raw stdout/stderr/exit
backend  crates/api/src/demonstrations.rs
         tools::observation renders each body — the same call a live round makes
        ↓ messages: [{role, content, tool?, argument?}]
POST /datasets/demonstrations data engine — kind=sft, origin=gym, status=ready
        ↓
engine   llm/corpus/sft.py — render_result adds the <|tool-call|> header
```

**Three services, and the assembly is in the middle one** because that is the
only place all three facts are available at once: gyms knows what the shell
said, the backend knows how it is rendered, the data engine owns every corpus —
and the two Python services never speak.

**A stored tool turn is not pre-rendered.** It carries `tool`, `argument` and
the body, and the engine adds the call header at build time. The body has to be
rendered in Rust, because that is the function serving uses; the header must not
be, because spelling it in Rust would be a fourth copy of `tools.call`. The
alternative — storing the finished block, as `convert-hermes` must — is what
`contracts/shell-observation.json` now exists to make safe, and it is a cost
worth avoiding where there is a choice.

**A set is a name that accumulates.** A second request under the same name
appends, so `document_count` and `updated_at` move — which is what the engine's
corpus key reads, and what stops a run reusing a corpus built from fewer
demonstrations and reporting success. The provenance on the dataset records the
gym, the task types, the seed base and `max_output_chars`: a body is clamped
when it is written, so a set rendered at one budget is not interchangeable with
one rendered at another, and nothing in the corpus key can see that.

A trajectory that does not end on an answer is **dropped by the backend and
refused by the data engine**, rather than stored. The engine would drop it at
build time anyway — the last assistant turn is the entire training signal — but
a 422 naming the row beats a lower conversation count noticed weeks later.

## A task type asks more than one question

Every generator draws its surface — the phrasing, the filename, the marker, the
needle's spelling — from a pool rather than an f-string. This is load-bearing
and not cosmetic. Before it, a seed varied the haystack and never the question,
and `bash_code` produced a byte-identical prompt for every seed it was ever
asked for. A few thousand demonstrations over three templates teach three
sentences: the model memorises the phrasing and answers the next one from
memory, which reads afterwards as a model that solved the gym.

`tests/test_tasks.py` pins a floor on the distinct prompts a type produces, and
holds every phrasing to the same ~80-character budget for the reason the system
block is short.

## Composition and mutation

The compositional task types leave the historical generators unchanged. The
new write primitive uses one redirection command; medium tasks use exactly one
pipe; hard tasks use three or more pipeline stages. Names, markers, values,
layouts and prompt fragments are generated from the seed, rather than selected
from a small scenario list, so repeated SFT batches can use fresh seed ranges.

Edit tasks store a mutation expectation in the task configuration. The episode
grader performs a post-answer `cat` and requires the expected content to be
present, so a correct final answer without the requested file change scores
zero. Pipelines run inside one shell call because just-bash starts a fresh shell
for every gym step; a pipe cannot span two steps.
