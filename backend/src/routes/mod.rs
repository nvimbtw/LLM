pub mod auth;
pub mod chat;
pub mod files;
pub mod health;
pub mod model;
pub mod preset;
pub mod training;
pub mod usage;
pub mod user;
pub mod validate;

use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::scope("/health").configure(health::configure));
    cfg.service(web::scope("/user").configure(user::configure));
    cfg.service(web::scope("/auth").configure(auth::configure));
    cfg.service(web::scope("/chat").configure(chat::configure));
    cfg.service(web::scope("/files").configure(files::configure));
    cfg.service(web::scope("/model").configure(model::configure));
    // Registered before "/training" so the more specific prefix matches first.
    cfg.service(web::scope("/training/presets").configure(preset::configure));
    cfg.service(web::scope("/training").configure(training::configure));
    cfg.service(web::scope("/usage").configure(usage::configure));
}
