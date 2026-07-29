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

    // Reconciles orphaned rows and starts the runners and the cycle driver. The
    // two handles come back for the HTTP layer to share: cloning `web::Data`
    // clones the Arc, not the driver, so a job enqueued on one worker is visible
    // to a socket that lands on another.
    let jobs = jobs::start(pool.clone()).await;

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
            .app_data(web::Data::new(jobs.queue.clone()))
            .app_data(web::Data::new(jobs.cycles.clone()))
            .configure(api::configure)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
