pub mod auth;
pub mod chat;
pub mod files;
pub mod health;
pub mod training;
pub mod user;

use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::scope("/health").configure(health::configure));
    cfg.service(web::scope("/user").configure(user::configure));
    cfg.service(web::scope("/auth").configure(auth::configure));
    cfg.service(web::scope("/chat").configure(chat::configure));
    cfg.service(web::scope("/files").configure(files::configure));
    cfg.service(web::scope("/training").configure(training::configure));
}
