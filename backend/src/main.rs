use actix_cors::Cors;
use actix_web::{App, HttpServer, web};

/// Loopback unless told otherwise, because `/internal` is left open when
/// `BACKEND_SERVICE_SECRET` is unset and the binding is the only thing keeping
/// it private. A container has to bind `0.0.0.0` for its published port to
/// reach anything, which is why `docker-compose.yml` sets the secrets too.
const DEFAULT_BIND_ADDR: &str = "127.0.0.1";
const PORT: u16 = 8080;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let jwt_secret = std::env::var("JWT_SECRET").expect("JWT_SECRET must be set");
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_string());

    let pool = db::create_pool(&database_url)
        .await
        .expect("failed to connect to database / run migrations");

    // Spawned here, as children of the server, because the engine holds the GPU
    // for as long as the server lives — and held to the end of `main`, because
    // dropping a handle kills the process it supervises.
    let engine_config = engine::config();
    let engine_secret = engine_config.service_secret.clone();
    let engine = service::Supervisor::start(engine_config)
        .await
        .expect("failed to start the python engine");
    println!("engine ready at {}", engine.base_url());

    let data_config = data_engine::config();
    let data_secret = data_config.service_secret.clone();
    let data = service::Supervisor::start(data_config)
        .await
        .expect("failed to start the data engine");
    println!("data engine ready at {}", data.base_url());

    let bench_config = bench_engine::config();
    let bench_secret = bench_config.service_secret.clone();
    let bench = match service::Supervisor::start(bench_config).await {
        Ok(s) => {
            println!("bench engine ready at {}", s.base_url());
            Some(s)
        }
        Err(e) => {
            eprintln!("bench engine not available: {e} — benchmarks will return 503 until it is up");
            eprintln!("to fix: ensure slate_bench exists: docker exec slate-postgres-1 psql -U ${{POSTGRES_USER:-slate}} -c \"CREATE DATABASE slate_bench;\"  or recreate volume with docker compose down -v");
            eprintln!("and then: docker compose up bench-engine --build");
            None
        }
    };

    let proxy = engine::Proxy::new(engine.base_url(), engine_secret);
    let data_proxy = data_engine::Proxy::new(data.base_url(), data_secret);
    let bench_proxy = {
        let bench_secret_clone = bench_secret.clone();
        match &bench {
            Some(s) => bench_engine::Proxy::new(s.base_url(), bench_secret_clone),
            None => {
                let cfg = bench_engine::config();
                bench_engine::Proxy::new(cfg.base_url(), bench_secret_clone)
            }
        }
    };

    // The reverse hop: the engine asks for the documents a run has reached the
    // point of needing. Said out loud for the reason the services say it — an
    // unauthenticated seam should be a thing you were told about, not a thing
    // you discover.
    let inbound = service::InboundSecret::from_env();
    match inbound.enforced() {
        true => println!("inbound service calls authenticated"),
        false => println!(
            "inbound service calls are UNAUTHENTICATED — set BACKEND_SERVICE_SECRET \
             here and in the engine to close /internal"
        ),
    }

    let search_cache = std::sync::Arc::new(api::search_cache::DataEngineCache::new(data_proxy.clone()));
    let registry = tools::Registry::from_env(Some(search_cache));
    match registry.origins() {
        [] => println!("no tools configured — replies that call one get a caveat back"),
        origins => println!("tools ready: {}", origins.join(", ")),
    }

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
            .app_data(web::Data::new(proxy.clone()))
            .app_data(web::Data::new(data_proxy.clone()))
            .app_data(web::Data::new(bench_proxy.clone()))
            .app_data(web::Data::new(inbound.clone()))
            .app_data(web::Data::new(registry.clone()))
            .configure(api::configure)
    })
    .bind((bind_addr.as_str(), PORT))?
    .run()
    .await?;

    // Explicit, so a graceful shutdown also stops the services rather than
    // leaving them holding the ports the next boot needs.
    engine.shutdown();
    data.shutdown();
    if let Some(b) = bench {
        b.shutdown();
    }
    Ok(())
}
