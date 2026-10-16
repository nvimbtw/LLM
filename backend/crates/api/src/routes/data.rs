//! Datasets, before they are training data.
//!
//! Every route here forwards to the data engine and models none of what it
//! returns. Reading is open to any signed-in user; everything that pulls,
//! processes or deletes is an admin's, because a pull writes a corpus into a
//! database and a delete removes one.

use actix_web::{HttpRequest, delete, get, post, put, web};
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
    answer(
        data.post(&admin_caller(&user), &path, body.into_inner())
            .await?,
    )
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

/// Create a custom processor from a valid first source version for administrators.
#[post("/processors")]
async fn create_processor(
    user: AdminUser,
    body: web::Json<Value>,
    data: web::Data<Proxy>,
) -> Answer {
    answer(
        data.post(&admin_caller(&user), "/processors", body.into_inner())
            .await?,
    )
}

#[post("/preview")]
async fn preview(user: AuthUser, body: web::Json<Value>, data: web::Data<Proxy>) -> Answer {
    answer(
        data.post(&caller(&user), "/preview", body.into_inner())
            .await?,
    )
}

/// Read one processor's source through the authenticated data-service hop.
#[get("/processors/{name}/source")]
async fn processor_source(
    user: AuthUser,
    name: web::Path<String>,
    data: web::Data<Proxy>,
) -> Answer {
    answer(
        data.get(&caller(&user), &format!("/processors/{name}/source"))
            .await?,
    )
}

/// Save custom processor source for administrators; the service owns reload policy.
#[put("/processors/{name}/source")]
async fn save_processor_source(
    user: AdminUser,
    name: web::Path<String>,
    body: web::Json<Value>,
    data: web::Data<Proxy>,
) -> Answer {
    answer(
        data.put(
            &admin_caller(&user),
            &format!("/processors/{name}/source"),
            body.into_inner(),
        )
        .await?,
    )
}

#[get("/runs")]
async fn runs(user: AuthUser, req: HttpRequest, data: web::Data<Proxy>) -> Answer {
    answer(data.get(&caller(&user), &with_query("/runs", &req)).await?)
}

#[post("/runs/{id}/cancel")]
async fn cancel(user: AdminUser, id: web::Path<String>, data: web::Data<Proxy>) -> Answer {
    answer(
        data.post(
            &admin_caller(&user),
            &format!("/runs/{id}/cancel"),
            Value::Null,
        )
        .await?,
    )
}

/// Register data routes, keeping literal pull paths ahead of dataset identity routes.
pub fn configure(cfg: &mut web::ServiceConfig) {
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
    cfg.service(create_processor);
    cfg.service(processor_source);
    cfg.service(save_processor_source);
    cfg.service(preview);
    cfg.service(cancel);
    cfg.service(runs);
}

#[cfg(test)]
mod source_tests {
    use super::*;
    use actix_web::{App, HttpResponse, http::StatusCode, test};
    use auth::{AccessLevel, token, token_cipher};
    use serde_json::json;
    use uuid::Uuid;

    /// Mint the encrypted browser credential for a test user's access level.
    fn bearer(level: AccessLevel) -> String {
        let signed = token::issue_user_token(Uuid::new_v4(), level, "web-secret").unwrap();
        format!(
            "Bearer {}",
            token_cipher::encrypt(&signed, "web-secret").unwrap()
        )
    }

    /// Echo the service hop's method and path after verifying its separate service token.
    async fn echo_source(req: HttpRequest) -> HttpResponse {
        let header = req
            .headers()
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap();
        token::verify_service_token(
            header.strip_prefix("Bearer ").unwrap(),
            "data-engine",
            "service-secret",
        )
        .expect("the browser token is replaced with a service token");
        HttpResponse::Ok().json(json!({"method":req.method().as_str(),"path":req.path()}))
    }

    /// Exercise source forwarding and deny ordinary-user writes before contacting the service.
    #[actix_web::test]
    async fn source_routes_forward_reads_and_gate_writes() {
        let upstream = actix_test::start(|| App::new().default_service(web::to(echo_source)));
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new("web-secret".to_owned()))
                .app_data(web::Data::new(Proxy::new(
                    upstream.url("").trim_end_matches('/'),
                    Some("service-secret".into()),
                )))
                .service(web::scope("/data").configure(configure)),
        )
        .await;
        let path = "/data/processors/example/source";
        let read = test::TestRequest::get()
            .uri(path)
            .insert_header(("authorization", bearer(AccessLevel::User)))
            .to_request();
        let body: Value = test::call_and_read_body_json(&app, read).await;
        assert_eq!(
            body,
            json!({"method":"GET","path":"/processors/example/source"})
        );
        for (method, path) in [
            (
                actix_web::http::Method::PUT,
                "/data/processors/example/source",
            ),
            (actix_web::http::Method::POST, "/data/processors"),
        ] {
            for level in [AccessLevel::User, AccessLevel::Admin] {
                let allowed = matches!(level, AccessLevel::Admin);
                let request = test::TestRequest::default()
                    .method(method.clone())
                    .uri(path)
                    .insert_header(("authorization", bearer(level)))
                    .set_json(json!({"source":"operator Python"}))
                    .to_request();
                let response = test::call_service(&app, request).await;
                if allowed {
                    let body: Value = test::read_body_json(response).await;
                    assert_eq!(
                        body,
                        json!({"method":method.as_str(),"path":path.strip_prefix("/data").unwrap()})
                    );
                } else {
                    assert_eq!(response.status(), StatusCode::FORBIDDEN);
                }
            }
        }
    }
}
