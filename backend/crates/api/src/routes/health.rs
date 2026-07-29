use actix_web::{HttpResponse, Responder, get, web};

#[get("")]
async fn check() -> impl Responder {
    HttpResponse::Ok().body("ok")
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(check);
}
