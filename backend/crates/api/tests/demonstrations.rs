use actix_web::{App, HttpResponse, web};
use auth::AccessLevel;
use serde_json::{Value, json};
use uuid::Uuid;

const JWT_SECRET: &str = "demonstrations-test-secret";

async fn trajectories(body: web::Json<Value>) -> HttpResponse {
    if body["gym_id"] == "general" {
        assert_eq!(body["gym_weights"]["math"], 2);
        assert_eq!(body["task_weights"]["math_expressions_easy"], 3);
    }
    HttpResponse::Ok().json(json!({"selection": {"mode": "mixture", "sources": [{"gym_id": "math", "weight": 2}]}, "trajectories": [{
        "gym_id": "math",
        "task_type": "math_expressions_easy", "seed": 0,
        "system": "Solve the question.", "prompt": "2 + 3 + 4?",
        "moves": [], "answer": "Answer: 9"
    }]}))
}

async fn store(body: web::Json<Value>) -> HttpResponse {
    if !body["description"].is_string() {
        return HttpResponse::UnprocessableEntity().finish();
    }
    HttpResponse::Created().json(body.into_inner())
}

#[actix_web::test]
async fn omitted_description_defaults_to_empty_and_supplied_description_survives() {
    let gyms =
        actix_test::start(|| App::new().route("/gym/trajectories", web::post().to(trajectories)));
    let data =
        actix_test::start(|| App::new().route("/datasets/demonstrations", web::post().to(store)));
    let gyms_url = gyms.url("").trim_end_matches('/').to_string();
    let data_url = data.url("").trim_end_matches('/').to_string();
    let server = actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(JWT_SECRET.to_string()))
            .app_data(web::Data::new(gyms_engine::Proxy::new(
                gyms_url.clone(),
                None,
            )))
            .app_data(web::Data::new(data_engine::Proxy::new(
                data_url.clone(),
                None,
            )))
            .configure(api::configure)
    });
    let jwt = auth::token::issue_user_token(Uuid::new_v4(), AccessLevel::Admin, JWT_SECRET)
        .expect("issued");
    let token = auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed");

    for description in [None, Some(""), Some("Math foundations")] {
        let mut request = json!({
            "gym_id": "general", "task_types": ["math_expressions_easy"],
            "gym_weights": {"math": 2}, "task_weights": {"math_expressions_easy": 3},
            "dataset_name": "math-foundations", "count": 1, "seed_base": 0
        });
        if let Some(description) = description {
            request["description"] = json!(description);
        }
        let mut response = server
            .post("/gym/demonstrations")
            .insert_header(("Authorization", format!("Bearer {token}")))
            .send_json(&request)
            .await
            .expect("sent");
        assert!(response.status().is_success(), "{}", response.status());
        let stored: Value = response.json().await.expect("json");
        assert_eq!(stored["description"], description.unwrap_or(""));
        assert_eq!(stored["name"], "math-foundations");
        assert_eq!(stored["steps"][0]["selection"]["sources"][0]["weight"], 2);
        assert_eq!(stored["conversations"][0]["meta"]["gym_id"], "math");
        assert_eq!(
            stored["conversations"][0]["messages"][2]["content"],
            "Answer: 9"
        );
    }
}
