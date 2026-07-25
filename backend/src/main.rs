use actix_cors::Cors;
use actix_web::{App, HttpServer, web};

mod benchmarks;
mod crypto;
mod db;
mod extractors;
mod jobs;
mod routes;
mod storage;
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

    // Training runs on blocking OS threads that can't touch the async pool. They
    // hand a finished model to this channel; the consumer below owns a pool clone
    // and performs the insert on the runtime that owns the pool. Unbounded so the
    // worker never blocks, and so a finish can't be lost behind a full buffer.
    let (model_tx, mut model_rx) = tokio::sync::mpsc::unbounded_channel();
    let persist_pool = pool.clone();
    actix_web::rt::spawn(async move {
        while let Some(record) = model_rx.recv().await {
            if let Err(e) = db::model::upsert(&persist_pool, &record).await {
                eprintln!("failed to persist model {}: {}", record.name, e);
            }
        }
    });

    // Created once and shared across all workers (cloning web::Data clones the
    // Arc, not the registry) so a job started on one worker is visible to a
    // socket that lands on another.
    let jobs = web::Data::new(jobs::JobRegistry::new(model_tx));

    // A benchmark run lives on a worker thread with no durable state of its own,
    // so a process that died mid-run left a row stuck on "running". Nothing is
    // in flight at startup by definition — close any such row out now, or the
    // page shows a spinner that never resolves and `is_busy` blocks new runs.
    match db::benchmark::fail_orphaned(&pool).await {
        Ok(0) => {}
        Ok(n) => println!("closed {} interrupted benchmark run(s)", n),
        Err(e) => eprintln!("failed to close interrupted benchmark runs: {}", e),
    }

    // Benchmarks follow the same bridge as training: the blocking worker thread
    // can't touch the async pool, so it sends its outcome here and the consumer
    // performs the update on the runtime that owns the pool.
    let (benchmark_tx, mut benchmark_rx) = tokio::sync::mpsc::unbounded_channel();
    let benchmark_pool = pool.clone();
    actix_web::rt::spawn(async move {
        while let Some(outcome) = benchmark_rx.recv().await {
            let result = match &outcome {
                benchmarks::Outcome::Done { id_run, score, report } => {
                    db::benchmark::complete(&benchmark_pool, *id_run, *score, report).await
                }
                benchmarks::Outcome::Failed { id_run, error } => {
                    eprintln!("benchmark {} failed: {}", id_run, error);
                    db::benchmark::fail(&benchmark_pool, *id_run, error).await
                }
            };
            if let Err(e) = result {
                eprintln!("failed to persist benchmark outcome: {}", e);
            }
        }
    });

    let benchmarks = web::Data::new(benchmarks::BenchmarkRegistry::new(benchmark_tx));

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000")
            .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"])
            .allowed_headers(vec![
                actix_web::http::header::CONTENT_TYPE,
                actix_web::http::header::AUTHORIZATION,
            ])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(web::Data::new(pool.clone()))
            .app_data(web::Data::new(jwt_secret.clone()))
            .app_data(jobs.clone())
            .app_data(benchmarks.clone())
            .configure(routes::configure)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
