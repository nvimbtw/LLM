//! Built vocabularies. Listing one is reading; deleting one takes the artifacts
//! a run may still be training against with it, so that stays with an admin.

use actix_web::{HttpRequest, delete, get, patch, web};
use llm_engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{Answer, answer, caller, with_query};

/// Lists vocabulary facts for the authenticated caller.
#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &with_query("/tokenizers", &req))
            .await?,
    )
}

/// Reads one vocabulary and its recorded model references.
#[get("/{tokenizer_id}")]
async fn get_one(user: AuthUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &format!("/tokenizers/{id}"))
            .await?,
    )
}

/// Forwards administrator-authorized display edits without reimplementing engine validation.
#[patch("/{tokenizer_id}")]
async fn edit(
    _admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    answer(
        engine
            .patch(
                &auth::Caller::System,
                &format!("/tokenizers/{id}"),
                body.into_inner(),
            )
            .await?,
    )
}

/// Removes vocabulary artifacts after administrator authorization.
#[delete("/{tokenizer_id}")]
async fn remove(_admin: AdminUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .delete(&auth::Caller::System, &format!("/tokenizers/{id}"))
            .await?,
    )
}

/// Registers vocabulary reads and admin-only metadata/deletion routes.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(edit);
    cfg.service(remove);
}
