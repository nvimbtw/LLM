//! Datasets, before they are training data.
//!
//! Every route here forwards to the data engine and models none of what it
//! returns. Reading is open to any signed-in user; everything that pulls,
//! processes or deletes is an admin's, because a pull writes a corpus into a
//! database and a delete removes one.

use actix_web::{HttpRequest, delete, get, post, web};
use data_engine::Proxy;
use serde_json::Value;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{Answer, admin_caller, answer, caller, with_query};

#[get("/hub")]
async fn hub(user: AuthUser, req: HttpRequest, data: web::Data<Proxy>) -> Answer {
    answer(
        data.get(&caller(&user), &with_query("/hub/datasets", &req))
            .await?,
    )
}

#[get("/hub/detail")]
async fn hub_detail(user: AuthUser, req: HttpRequest, data: web::Data<Proxy>) -> Answer {
    answer(
        data.get(&caller(&user), &with_query("/hub/datasets/detail", &req))
            .await?,
    )
}

#[get("/hub/sample")]
async fn hub_sample(user: AuthUser, req: HttpRequest, data: web::Data<Proxy>) -> Answer {
    answer(
        data.get(&caller(&user), &with_query("/hub/datasets/sample", &req))
            .await?,
    )
}

#[get("/datasets")]
async fn index(user: AuthUser, data: web::Data<Proxy>) -> Answer {
    answer(data.get(&caller(&user), "/datasets").await?)
}

#[get("/datasets/{id}")]
async fn show(user: AuthUser, id: web::Path<String>, data: web::Data<Proxy>) -> Answer {
    answer(data.get(&caller(&user), &format!("/datasets/{id}")).await?)
}

#[get("/datasets/{id}/documents")]
async fn documents(
    user: AuthUser,
    id: web::Path<String>,
    req: HttpRequest,
    data: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/datasets/{id}/documents"), &req);
    answer(data.get(&caller(&user), &path).await?)
}

#[post("/datasets/pull")]
async fn pull(user: AdminUser, body: web::Json<Value>, data: web::Data<Proxy>) -> Answer {
    answer(
        data.post(&admin_caller(&user), "/datasets/pull", body.into_inner())
            .await?,
    )
}

#[post("/datasets/{id}/process")]
async fn process(
    user: AdminUser,
    id: web::Path<String>,
    body: web::Json<Value>,
    data: web::Data<Proxy>,
) -> Answer {
    let path = format!("/datasets/{id}/process");
    answer(data.post(&admin_caller(&user), &path, body.into_inner()).await?)
}

#[delete("/datasets/{id}")]
async fn destroy(user: AdminUser, id: web::Path<String>, data: web::Data<Proxy>) -> Answer {
    answer(
        data.delete(&admin_caller(&user), &format!("/datasets/{id}"))
            .await?,
    )
}

#[get("/processors")]
async fn processors(user: AuthUser, req: HttpRequest, data: web::Data<Proxy>) -> Answer {
    answer(
        data.get(&caller(&user), &with_query("/processors", &req))
            .await?,
    )
}

#[post("/preview")]
async fn preview(user: AuthUser, body: web::Json<Value>, data: web::Data<Proxy>) -> Answer {
    answer(data.post(&caller(&user), "/preview", body.into_inner()).await?)
}

#[get("/runs")]
async fn runs(user: AuthUser, req: HttpRequest, data: web::Data<Proxy>) -> Answer {
    answer(data.get(&caller(&user), &with_query("/runs", &req)).await?)
}

#[post("/runs/{id}/cancel")]
async fn cancel(user: AdminUser, id: web::Path<String>, data: web::Data<Proxy>) -> Answer {
    answer(
        data.post(&admin_caller(&user), &format!("/runs/{id}/cancel"), Value::Null)
            .await?,
    )
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // `/hub` before `/datasets/{id}` is not ordering-sensitive here — the paths
    // do not overlap — but `pull` must precede `{id}` or a POST to it would try
    // to parse "pull" as a dataset id.
    cfg.service(hub);
    cfg.service(hub_detail);
    cfg.service(hub_sample);
    cfg.service(pull);
    cfg.service(index);
    cfg.service(documents);
    cfg.service(process);
    cfg.service(show);
    cfg.service(destroy);
    cfg.service(processors);
    cfg.service(preview);
    cfg.service(cancel);
    cfg.service(runs);
}
