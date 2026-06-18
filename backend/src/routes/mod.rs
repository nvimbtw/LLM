pub mod auth;
pub mod chat;
pub mod health;
pub mod user;

use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::scope("/health").configure(health::configure));
    cfg.service(web::scope("/user").configure(user::configure));
    cfg.service(web::scope("/auth").configure(auth::configure));
    cfg.service(web::scope("/chat").configure(chat::configure));
}
