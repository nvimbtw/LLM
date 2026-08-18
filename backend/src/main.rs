use actix_cors::Cors;
use actix_web::{App, HttpServer, web};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let jwt_secret = std::env::var("JWT_SECRET").expect("JWT_SECRET must be set");

    let pool = db::create_pool(&database_url)
        .await
        .expect("failed to connect to database / run migrations");

    // Spawned here, as a child of the server, because it holds the GPU for as
    // long as the server lives — and held to the end of `main`, because dropping
    // the handle kills it.
    let engine_config = engine::EngineConfig::from_env();
    let service_secret = engine_config.service_secret.clone();
    let engine = engine::Engine::start(engine_config)
        .await
        .expect("failed to start the python engine");
    println!("engine ready at {}", engine.base_url());

    let proxy = engine::Proxy::new(engine.base_url(), engine::AUDIENCE, service_secret);

    let registry = tools::Registry::from_env();
    match registry.names().as_slice() {
        [] => println!("no tools configured — replies that call one get a caveat back"),
        names => println!("tools ready: {}", names.join(", ")),
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
            .app_data(web::Data::new(registry.clone()))
            .configure(api::configure)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await?;

    // Explicit, so a graceful shutdown also stops the engine rather than leaving
    // it holding the port the next boot needs.
    engine.shutdown();
    Ok(())
}
