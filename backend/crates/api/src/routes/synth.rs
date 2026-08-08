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
async fn get_corpus(
    _admin: AdminUser,
    engine: web::Data<engine::Client>,
    query: web::Query<CorpusQuery>,
) -> impl Responder {
    let id_model = query.into_inner().id_model;

    // The measuring is the engine's now, so this awaits HTTP rather than
    // occupying a blocking thread.
    let result = synth::corpus_stats(engine.get_ref(), id_model).await;

    match result {
        Ok(stats) => HttpResponse::Ok().json(stats),
        // A named model with no tokenizer, or an unreadable meta.json: the caller
        // picked a model that cannot measure anything, which is their input being
        // wrong rather than the server failing.
        Err(message) => HttpResponse::BadRequest().body(message),
    }
}

// There is deliberately no endpoint to prepare the chat data for training. A
// pretrain renders its own share of the conversations when it starts
// (`training::build_dataset`), so the corpus is always derived from the
// conversations as they are now — the same way the run derives its tokenizer and
// its embeddings. Nothing to trigger, nothing to forget.
