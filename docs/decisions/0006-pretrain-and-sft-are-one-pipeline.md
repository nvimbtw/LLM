# Training modes (pretrain vs SFT)

A run has a `mode` (`TrainConfig.mode` in `training.rs`): **pretrain** (default) or
**sft** (supervised fine-tuning). Both are two-phase — build a dataset, then train
on the GPU — and share the job/WebSocket progress plumbing (`jobs.rs`); they differ
only in phase 1 and which `llm::train` entry point phase 2 drives. The route
(`routes/training.rs::start_training`) branches on `mode`.

- **Pretrain** — `build_dataset` (trains a fresh BPE tokenizer over the uploaded
  books + synthetic docs → `datasets/corpus.bin` + `artifacts/tokenizer/merges.bin`)
  then `llm::train::train` (fresh random weights). Loss is computed over every
  position.
- **SFT** — fine-tunes an *existing* base model on chat conversations, training the
  loss **only on the assistant's reply tokens**. The request carries a
  `base_model_id` (validated present + resolvable, else `400`). Phase 1 is
  `build_sft_dataset(base_model_dir)`; phase 2 is `llm::train::sft_train`, which
  loads the base weights and drives the masked step. SFT reuses the base's
  tokenizer — it never retrains BPE (that would renumber the vocab and make the
  fine-tune gibberish) — and copies `merges.bin` next to the fine-tuned weights so
  the output model loads standalone. SFT typically wants a lower `lr` and fewer
  `epochs` than pretraining.

**The masking convention** (shared by the whole SFT path, from
`chat_template::build_training_example` through the CUDA loss kernels): predicting
token `i+1` from position `i`, `mask[i] = 1` iff token `i+1` is an assistant-reply
token — from the first token after an `<|assistant|>` marker through the closing
`<|/assistant|>` marker inclusive — and `0` everywhere else (system, user,
scaffold, cross-example boundaries). Masked-out positions contribute no loss and no
gradient.
