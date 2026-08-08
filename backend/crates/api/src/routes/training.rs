#![allow(non_snake_case)]

// What's left of the training routes now that runs are queued jobs.
//
// Starting a run, listing runs and following one's progress all moved to
// `routes/jobs.rs` — a training run is a `train` job like any other, and having a
// second way to start one would mean a second way to bypass the queue. What stays
// is the one thing that is about training rather than about scheduling: the list
// of datasets the composer offers.

use actix_web::{HttpResponse, Responder, get, web};

use crate::extractors::AdminUser;

// The corpora the engine has on disk, so the params rail can render a checkbox
// per dataset. Proxied rather than read here: the engine owns the data, and a
// second filesystem view on this side could only ever disagree with it.
#[get("/datasets")]
async fn list_datasets(_admin: AdminUser, engine: web::Data<engine::Client>) -> impl Responder {
    match engine.datasets().await {
        Ok(list) => HttpResponse::Ok().json(list),
        Err(e) => {
            eprintln!("list datasets error: {}", e);
            HttpResponse::InternalServerError().body("failed to list datasets")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list_datasets);
}
