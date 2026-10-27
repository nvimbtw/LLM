//! Administrator-only, opaque adapters for engine-owned model authoring and inspection.

use actix_web::{get, patch, post, put, web, HttpRequest};
use llm_engine::{Method, Proxy};
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::AdminUser;
use crate::upstream::{admin_json as forward, with_query, Answer};

const REQUEST_BYTES: usize = 1024 * 1024;

/// Identify one of the two catalogs sharing the engine's versioned authoring protocol.
struct Catalog(&'static str);

/// Read an engine-owned catalog page; filters and cursor interpretation stay upstream.
#[get("")]
async fn list(
    admin: AdminUser,
    req: HttpRequest,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &with_query(family.0, &req),
        None,
    )
    .await
}

/// Save a catalog resource, including drafts that are not yet valid for execution.
#[post("")]
async fn create(
    admin: AdminUser,
    body: web::Json<Value>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::POST,
        family.0,
        Some(body.into_inner()),
    )
    .await
}

/// Read one resource's metadata without importing its source in Rust.
#[get("/{id}")]
async fn detail(
    admin: AdminUser,
    id: web::Path<Uuid>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("{}/{id}", family.0),
        None,
    )
    .await
}

/// Forward an optimistic metadata update with its exact actor identity.
#[patch("/{id}")]
async fn metadata(
    admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::PATCH,
        &format!("{}/{id}", family.0),
        Some(body.into_inner()),
    )
    .await
}

/// Read the saved authoring draft, including invalid source or YAML.
#[get("/{id}/draft")]
async fn draft(
    admin: AdminUser,
    id: web::Path<Uuid>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("{}/{id}/draft", family.0),
        None,
    )
    .await
}

/// Save opaque draft bytes; the engine enforces source limits and optimistic versions.
#[put("/{id}/draft")]
async fn save_draft(
    admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::PUT,
        &format!("{}/{id}/draft", family.0),
        Some(body.into_inner()),
    )
    .await
}

/// Read immutable revision summaries with engine-owned pagination.
#[get("/{id}/revisions")]
async fn revisions(
    admin: AdminUser,
    id: web::Path<Uuid>,
    req: HttpRequest,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &with_query(&format!("{}/{id}/revisions", family.0), &req),
        None,
    )
    .await
}

/// Accept a revision only when the engine's bound validation evidence permits it.
#[post("/{id}/revisions")]
async fn accept(
    admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::POST,
        &format!("{}/{id}/revisions", family.0),
        Some(body.into_inner()),
    )
    .await
}

/// Read a pinned revision using validated resource and revision UUIDs.
#[get("/{id}/revisions/{revision_id}")]
async fn revision(
    admin: AdminUser,
    ids: web::Path<(Uuid, Uuid)>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    let (id, revision_id) = ids.into_inner();
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("{}/{id}/revisions/{revision_id}", family.0),
        None,
    )
    .await
}

/// Resolve one explicitly selected immutable recipe revision from a preset or retry without a mutable latest lookup.
#[get("/revisions/{id}")]
async fn selected_recipe(
    admin: AdminUser,
    id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("/architectures/revisions/{id}"),
        None,
    )
    .await
}

/// Read prospective engine-owned training bindings/profile; this request creates no job or numerical evidence.
#[post("/revisions/{id}/training-preview")]
async fn training_preview(
    admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::POST,
        &format!("/architectures/revisions/{id}/training-preview"),
        Some(body.into_inner()),
    )
    .await
}

/// Read eligible editable wrapper source for a pinned builtin without creating a custom resource.
#[get("/{id}/revisions/{revision_id}/copy-source")]
async fn copy_source(
    admin: AdminUser,
    ids: web::Path<(Uuid, Uuid)>,
    engine: web::Data<Proxy>,
) -> Answer {
    let (id, revision_id) = ids.into_inner();
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("/blocks/{id}/revisions/{revision_id}/copy-source"),
        None,
    )
    .await
}

/// Enqueue durable validation; observing the response never owns worker cancellation.
#[post("/{id}/validations")]
async fn validate(
    admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    family: web::Data<Catalog>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::POST,
        &format!("{}/{id}/validations", family.0),
        Some(body.into_inner()),
    )
    .await
}

/// Read the engine's authoritative recipe schema before the UUID route can match it.
#[get("/schema")]
async fn schema(admin: AdminUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &with_query("/architectures/schema", &req),
        None,
    )
    .await
}

/// Read exact saved YAML declarations without executing custom dependencies or treating metadata as acceptance evidence.
#[post("/{id}/declaration")]
async fn declaration(
    admin: AdminUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::POST,
        &format!("/architectures/{id}/declaration"),
        Some(body.into_inner()),
    )
    .await
}

/// Read authoritative tokenizer facts for architecture validation without trusting browser vocabulary guesses.
#[get("/tokenizers/{id}/architecture-binding")]
async fn tokenizer_binding(
    admin: AdminUser,
    id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &with_query(&format!("/tokenizers/{id}/architecture-binding"), &req),
        None,
    )
    .await
}

/// Read a durable validation report without exposing its diagnostics to ordinary users.
#[get("/validations/{id}")]
async fn report(admin: AdminUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("/validations/{id}"),
        None,
    )
    .await
}

/// Read a bounded graph page with its exact cursor and scope query.
#[get("/validations/{id}/graph")]
async fn validation_graph(
    admin: AdminUser,
    id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &with_query(&format!("/validations/{id}/graph"), &req),
        None,
    )
    .await
}

/// Read saved model-definition metadata under administrator authorization.
#[get("/models/{id}/definition")]
async fn definition(admin: AdminUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("/models/{id}/definition"),
        None,
    )
    .await
}

/// Read an existing model graph artifact; the engine never recaptures on this read.
#[get("/models/{id}/graph")]
async fn model_graph(
    admin: AdminUser,
    id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    forward(
        &admin,
        &engine,
        Method::GET,
        &with_query(&format!("/models/{id}/graph"), &req),
        None,
    )
    .await
}

/// Read one frozen model source revision without consulting catalog heads or interpreting code in Rust.
#[get("/models/{id}/sources/{revision_id}")]
async fn model_source(
    admin: AdminUser,
    ids: web::Path<(Uuid, Uuid)>,
    engine: web::Data<Proxy>,
) -> Answer {
    let (id, revision_id) = ids.into_inner();
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("/models/{id}/sources/{revision_id}"),
        None,
    )
    .await
}

/// Read one published validation snapshot source under the same administrative authorization as its report.
#[get("/validations/{id}/sources/{revision_id}")]
async fn validation_source(
    admin: AdminUser,
    ids: web::Path<(Uuid, Uuid)>,
    engine: web::Data<Proxy>,
) -> Answer {
    let (id, revision_id) = ids.into_inner();
    forward(
        &admin,
        &engine,
        Method::GET,
        &format!("/validations/{id}/sources/{revision_id}"),
        None,
    )
    .await
}

/// Register shared catalog actions under one catalog-specific scope.
fn catalog(cfg: &mut web::ServiceConfig) {
    cfg.service(list)
        .service(create)
        .service(detail)
        .service(metadata)
        .service(draft)
        .service(save_draft)
        .service(revisions)
        .service(accept)
        .service(revision)
        .service(validate);
}

/// Register private authoring routes and local JSON limits before generic model paths.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/blocks")
            .app_data(web::Data::new(Catalog("/blocks")))
            .app_data(web::JsonConfig::default().limit(REQUEST_BYTES))
            .service(copy_source)
            .configure(catalog),
    );
    cfg.service(
        web::scope("/architectures")
            .app_data(web::Data::new(Catalog("/architectures")))
            .app_data(web::JsonConfig::default().limit(REQUEST_BYTES))
            .service(schema)
            .service(selected_recipe)
            .service(training_preview)
            .service(declaration)
            .configure(catalog),
    );
    cfg.service(report)
        .service(tokenizer_binding)
        .service(validation_graph)
        .service(definition)
        .service(model_graph)
        .service(model_source)
        .service(validation_source);
}
