mod startup;

use actix_web::{App, HttpServer, web};

/// Loopback is the development trust boundary when reverse-hop authentication is disabled.
const DEFAULT_BIND_ADDR: &str = "127.0.0.1";
const PORT: u16 = 8080;

/// Connect backend storage, start service siblings and serve the API until graceful shutdown completes.
/// Database and server startup failures are fatal; unavailable services are monitored while the API runs.
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let jwt_secret = std::env::var("JWT_SECRET").expect("JWT_SECRET must be set");
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_owned());
    let pool = db::create_pool(&database_url)
        .await
        .expect("failed to connect to database / run migrations");
    let services = startup::Services::start();
    let application = services.application();
    HttpServer::new(move || {
        App::new()
            .wrap(startup::cors())
            .app_data(web::Data::new(pool.clone()))
            .app_data(web::Data::new(jwt_secret.clone()))
            .configure(|cfg| application.configure(cfg))
    })
    .bind((bind_addr.as_str(), PORT))?
    .run()
    .await?;
    services.shutdown();
    Ok(())
}
