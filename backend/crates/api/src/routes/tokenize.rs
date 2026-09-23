//! How long a text is under a model's own vocabulary.
//!
//! Not `tokenizers.rs`, which is the registry of built vocabularies — that one
//! is keyed by tokenizer id and answers what exists. This asks a *model* to
//! measure a string, and loads no weights to do it.

use actix_web::{post, web};
use llm_engine::Proxy;
use serde_json::Value;

use crate::extractors::AuthUser;
use crate::upstream::{answer, caller, Answer};

/// `AuthUser`, unlike `/runs/tokenize` next door, which queues a vocabulary
/// build and is admin work: the composer draws its context meter from this on
/// every pause in typing, so anyone who can hold a conversation can ask.
#[post("/count")]
async fn count(user: AuthUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&caller(&user), "/tokenize/count", body.into_inner())
            .await?,
    )
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(count);
}
