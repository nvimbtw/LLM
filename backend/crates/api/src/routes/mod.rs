pub mod attachments;
pub mod auth;
pub mod benchmarks;
pub mod chats;
pub mod data;
pub mod gym;
pub mod health;
pub mod internal;
pub mod models;
pub mod preset;
pub mod projects;
pub mod runs;
pub mod stream;
pub mod tokenize;
pub mod tokenizers;
pub mod usage;
pub mod user;
pub mod validate;

use actix_web::web;

/// Register backend-owned resources and authenticated adapters to domain services.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::scope("/health").configure(health::configure));
    cfg.service(web::scope("/auth").configure(auth::configure));
    cfg.service(web::scope("/user").configure(user::configure));
    cfg.service(web::scope("/presets").configure(preset::configure));
    cfg.service(web::scope("/projects").configure(projects::configure));
    cfg.service(web::scope("/chats").configure(chats::configure));
    cfg.service(web::scope("/attachments").configure(attachments::configure));
    cfg.service(web::scope("/messages").configure(chats::configure_messages));
    cfg.service(web::scope("/models").configure(models::configure));
    cfg.service(web::scope("/benchmarks").configure(benchmarks::configure));
    cfg.service(web::scope("/runs").configure(runs::configure));
    cfg.service(web::scope("/tokenize").configure(tokenize::configure));
    cfg.service(web::scope("/tokenizers").configure(tokenizers::configure));
    cfg.service(web::scope("/usage").configure(usage::configure));
    cfg.service(web::scope("/ws").configure(stream::configure));

    cfg.service(web::scope("/data").configure(data::configure));

    cfg.service(web::scope("/internal").configure(internal::configure));

    cfg.service(web::scope("/gym").configure(gym::configure));
}
