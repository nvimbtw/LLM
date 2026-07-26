use actix_cors::Cors;
use actix_web::{App, HttpServer, web};
use std::sync::Arc;

mod autolabel;
mod crypto;
mod db;
mod extractors;
mod gather;
mod queue;
mod routes;
mod storage;
mod train_worker;
mod training;
mod ws;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let jwt_secret = std::env::var("JWT_SECRET").expect("JWT_SECRET must be set");

    let pool = db::create_pool(&database_url)
        .await
        .expect("failed to connect to database / run migrations");

    // Nothing is in flight at startup by definition: a worker lives on a thread
    // with no durable state, so any row still marked running belongs to a process
    // that died. Left alone they would hold the head of the queue forever and show
    // a throbber that never resolves. Queued rows are untouched — the runner picks
    // them up below, which is the whole point of a queue that survives a restart.
    match db::job::fail_orphaned(&pool).await {
        Ok(0) => {}
        Ok(n) => println!("closed {} interrupted job(s)", n),
        Err(e) => eprintln!("failed to close interrupted jobs: {}", e),
    }
    // The same reconciliation for benchmark history rows, which are opened by the
    // runner and so can be orphaned by the same crash.
    match db::benchmark::fail_orphaned(&pool).await {
        Ok(0) => {}
        Ok(n) => println!("closed {} interrupted benchmark run(s)", n),
        Err(e) => eprintln!("failed to close interrupted benchmark runs: {}", e),
    }

    // The single serialization point for the GPU. Created once and shared across
    // every worker (cloning web::Data clones the Arc, not the queue), so a job
    // enqueued on one worker is visible to a socket that lands on another.
    let queue = Arc::new(queue::Queue::new(pool.clone()));
    actix_web::rt::spawn(queue.clone().run());

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000")
            .allowed_methods(vec!["GET", "POST", "PUT", "PATCH", "DELETE"])
            .allowed_headers(vec![
                actix_web::http::header::CONTENT_TYPE,
                actix_web::http::header::AUTHORIZATION,
            ])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(web::Data::new(pool.clone()))
            .app_data(web::Data::new(jwt_secret.clone()))
            .app_data(web::Data::new(queue.clone()))
            .configure(routes::configure)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
