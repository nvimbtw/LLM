// The SFT corpus, measured.
//
// Generation itself is a job (queued through `POST /jobs` with kind `synth`, run on
// the network lane), so this file is only the read side: what is in the corpus
// right now, and how much of it the model that will train on it can actually hold.
//
// The token counts come from the real tokenizer and the real chat template, which
// is the whole point — a character count cannot answer "does this conversation fit
// the window", and that is the question that decides whether an SFT run trains on
// conversations or on fragments of them.

use actix_web::{HttpResponse, Responder, get, web};
use serde::Deserialize;
use uuid::Uuid;

use crate::extractors::AdminUser;
use synth;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(get_corpus);
}

#[derive(Deserialize)]
struct CorpusQuery {
    // The model whose tokenizer and context window to measure against. Without it
    // the file and conversation counts still come back, and the token distribution
    // is simply absent rather than guessed.
    #[serde(default)]
    id_model: Option<Uuid>,
}

#[get("/corpus")]
async fn get_corpus(_admin: AdminUser, query: web::Query<CorpusQuery>) -> impl Responder {
    let id_model = query.into_inner().id_model;

    // Reading every conversation and BPE-encoding it is real CPU work — tens of
    // milliseconds over a few hundred conversations, and unbounded in principle —
    // so it goes to a blocking thread rather than stalling an actix worker.
    let result = web::block(move || synth::corpus_stats(id_model)).await;

    match result {
        Ok(Ok(stats)) => HttpResponse::Ok().json(stats),
        // A named model with no tokenizer on disk, or an unreadable meta.json: the
        // caller picked a model that cannot measure anything, which is their input
        // being wrong rather than the server failing.
        Ok(Err(message)) => HttpResponse::BadRequest().body(message),
        Err(e) => {
            eprintln!("corpus stats join error: {}", e);
            HttpResponse::InternalServerError().body("failed to read the SFT corpus")
        }
    }
}

// There is deliberately no endpoint to prepare the chat data for training. A
// pretrain renders its own share of the conversations when it starts
// (`training::build_dataset`), so the corpus is always derived from the
// conversations as they are now — the same way the run derives its tokenizer and
// its embeddings. Nothing to trigger, nothing to forget.
