pub mod auth;
pub mod chats;
pub mod datasets;
pub mod health;
pub mod models;
pub mod preset;
pub mod runs;
pub mod stream;
pub mod usage;
pub mod user;
pub mod validate;

use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    // Owned here: people, their sessions, and the presets they compose runs from.
    cfg.service(web::scope("/health").configure(health::configure));
    cfg.service(web::scope("/auth").configure(auth::configure));
    cfg.service(web::scope("/user").configure(user::configure));
    cfg.service(web::scope("/presets").configure(preset::configure));

    // Owned by the engine, reached through it.
    cfg.service(web::scope("/chats").configure(chats::configure));
    cfg.service(web::scope("/messages").configure(chats::configure_messages));
    cfg.service(web::scope("/models").configure(models::configure));
    cfg.service(web::scope("/runs").configure(runs::configure));
    cfg.service(web::scope("/usage").configure(usage::configure));
    cfg.service(web::scope("/ws").configure(stream::configure));
    cfg.configure(datasets::configure);
}
