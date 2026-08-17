use actix_web::{get, web};
use engine::Proxy;

use crate::extractors::AuthUser;
use crate::upstream::{Answer, answer, caller};

#[get("/datasets")]
async fn pretrain(user: AuthUser, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), "/datasets").await?)
}

#[get("/sft/datasets")]
async fn sft(user: AuthUser, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), "/sft/datasets").await?)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(pretrain);
    cfg.service(sft);
}
