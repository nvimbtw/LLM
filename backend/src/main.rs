use actix_cors::Cors;
use actix_web::{App, HttpServer, web};
use sqlx::PgPool;

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

    create_admin(&pool).await;

    // Created once and shared across all workers (cloning web::Data clones the
    // Arc, not the registry) so a job started on one worker is visible to a
    // socket that lands on another.
    let jobs = web::Data::new(jobs::JobRegistry::new());

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
            .configure(routes::configure)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

async fn create_admin(pool: &PgPool) {
    let pwd_hash = match crypto::password::hash_password("admin") {
        Ok(hash) => hash,
        Err(e) => panic!("Failed to hash password {}", e),
    };

    match db::user::create(
        &pool,
        "admin",
        pwd_hash.as_str(),
        Some(db::user::AccessLevel::Admin),
    )
    .await
    {
        Ok(user) => println!("{}", user.username),
        Err(_) => println!("Admin already exists"),
    }
}
