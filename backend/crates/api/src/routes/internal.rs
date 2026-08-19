//! What a service may ask the backend for.
//!
//! One direction, one consumer: the LLM engine, packing a corpus, reading
//! documents that belong to the data engine. It cannot ask the data engine
//! directly — the two Python services never speak — so the request arrives here
//! and is re-minted, carrying the same acting caller into the other service.
//!
//! Nothing here is reachable with a browser session. `ServiceCaller` accepts a
//! service token and only a service token, which is why these routes live under
//! their own scope rather than beside the `/data` routes a person uses: the same
//! path with two authentication schemes is the kind of thing that gets one of
//! them wrong.
//!
//! Reads only. A run consumes datasets; it does not create, clean or delete
//! them, and a service that cannot write cannot corrupt what a person curated.

use actix_web::{HttpRequest, get, web};
use data_engine::Proxy;

use crate::extractors::ServiceCaller;
use crate::upstream::{Answer, answer, with_query};

#[get("/datasets/{id}")]
async fn dataset(
    service: ServiceCaller,
    id: web::Path<String>,
    data: web::Data<Proxy>,
) -> Answer {
    answer(data.get(&service.caller, &format!("/datasets/{id}")).await?)
}

/// One page of documents. The query string is passed through untouched: paging
/// is the data engine's to define, and a run sizes its own pages by what the
/// last one measured.
#[get("/datasets/{id}/documents")]
async fn documents(
    service: ServiceCaller,
    id: web::Path<String>,
    req: HttpRequest,
    data: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/datasets/{id}/documents"), &req);
    answer(data.get(&service.caller, &path).await?)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(dataset);
    cfg.service(documents);
}
