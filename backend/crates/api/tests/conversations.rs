//! These routes must work without any Python proxy registered in the application.
use actix_web::{http::StatusCode, test, web, App};
use auth::{token, token_cipher, AccessLevel};
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

const SECRET: &str = "conversation-api-fixture";

/// Issues a sealed browser bearer token for one isolated account.
fn bearer(owner: Uuid) -> String {
    let token = token::issue_user_token(owner, AccessLevel::User, SECRET).unwrap();
    format!("Bearer {}", token_cipher::encrypt(&token, SECRET).unwrap())
}

/// Recent slices preserve the complete archive endpoint and reject excessive or invalid limits.
#[sqlx::test(migrations = "../db/migrations")]
async fn recent_archive_limits_and_ownership(pool: PgPool) {
    let owner = db::user::create(&pool, "archive-owner", "hash", None)
        .await
        .unwrap()
        .id_user;
    let stranger = db::user::create(&pool, "archive-other", "hash", None)
        .await
        .unwrap()
        .id_user;
    for title in ["first", "second", "third"] {
        db::conversation::create(&pool, owner, title).await.unwrap();
    }
    db::conversation::create(&pool, stranger, "private")
        .await
        .unwrap();
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(pool))
            .app_data(web::Data::new(SECRET.to_owned()))
            .configure(api::configure),
    )
    .await;
    for (path, count) in [
        ("/chats", 3),
        ("/chats?limit=2", 2),
        ("/chats?limit=100", 3),
    ] {
        let request = test::TestRequest::get()
            .uri(path)
            .insert_header(("Authorization", bearer(owner)))
            .to_request();
        let response = test::call_service(&app, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value = test::read_body_json(response).await;
        let rows = result["chats"].as_array().unwrap();
        assert_eq!(rows.len(), count);
        assert!(rows.iter().all(|row| row["title"] != "private"));
    }
    for path in [
        "/chats?limit=0",
        "/chats?limit=-1",
        "/chats?limit=101",
        "/chats?limit=invalid",
    ] {
        let request = test::TestRequest::get()
            .uri(path)
            .insert_header(("Authorization", bearer(owner)))
            .to_request();
        assert_eq!(
            test::call_service(&app, request).await.status(),
            StatusCode::BAD_REQUEST
        );
    }
}

fn multipart(content: &[u8]) -> Vec<u8> {
    let mut body = b"--upload\r\nContent-Disposition: form-data; name=\"file\"; filename=\"photo.png\"\r\nContent-Type: image/png\r\n\r\n".to_vec();
    body.extend_from_slice(content);
    body.extend_from_slice(b"\r\n--upload--\r\n");
    body
}

#[sqlx::test(migrations = "../db/migrations")]
async fn history_and_files_work_without_inference(pool: PgPool) {
    let owner = db::user::create(&pool, "owner", "hash", None)
        .await
        .unwrap()
        .id_user;
    let stranger = db::user::create(&pool, "other", "hash", None)
        .await
        .unwrap()
        .id_user;
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(pool))
            .app_data(web::Data::new(SECRET.to_owned()))
            .configure(api::configure),
    )
    .await;
    let request = test::TestRequest::post()
        .uri("/chats")
        .insert_header(("Authorization", bearer(owner)))
        .set_json(json!({"title":"Persisted without inference"}))
        .to_request();
    let response = test::call_service(&app, request).await;
    assert_eq!(response.status(), StatusCode::CREATED);
    let chat: Value = test::read_body_json(response).await;
    let path = format!("/chats/{}", chat["chat_id"].as_str().unwrap());
    let request = test::TestRequest::get()
        .uri(&path)
        .insert_header(("Authorization", bearer(owner)))
        .to_request();
    let history: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(history["messages"], json!([]));
    let request = test::TestRequest::get()
        .uri(&path)
        .insert_header(("Authorization", bearer(stranger)))
        .to_request();
    assert_eq!(
        test::call_service(&app, request).await.status(),
        StatusCode::NOT_FOUND
    );

    let content = b"\x89PNG\r\n\0\xff";
    let request = test::TestRequest::post()
        .uri("/attachments")
        .insert_header(("Authorization", bearer(owner)))
        .insert_header(("Content-Type", "multipart/form-data; boundary=upload"))
        .set_payload(multipart(content))
        .to_request();
    let response = test::call_service(&app, request).await;
    assert_eq!(response.status(), StatusCode::CREATED);
    let file: Value = test::read_body_json(response).await;
    let path = format!("/attachments/{}", file["attachment_id"].as_str().unwrap());
    let request = test::TestRequest::get()
        .uri(&path)
        .insert_header(("Authorization", bearer(owner)))
        .to_request();
    let response = test::call_service(&app, request).await;
    assert_eq!(response.headers().get("Content-Type").unwrap(), "image/png");
    assert!(response
        .headers()
        .get("Content-Disposition")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("inline"));
    assert_eq!(test::read_body(response).await.as_ref(), content);
    let request = test::TestRequest::get()
        .uri(&path)
        .insert_header(("Authorization", bearer(stranger)))
        .to_request();
    assert_eq!(
        test::call_service(&app, request).await.status(),
        StatusCode::NOT_FOUND
    );

    for (bytes, status) in [
        (vec![], StatusCode::BAD_REQUEST),
        (vec![1; 50 * 1024 * 1024 + 1], StatusCode::PAYLOAD_TOO_LARGE),
    ] {
        let request = test::TestRequest::post()
            .uri("/attachments")
            .insert_header(("Authorization", bearer(owner)))
            .insert_header(("Content-Type", "multipart/form-data; boundary=upload"))
            .set_payload(multipart(&bytes))
            .to_request();
        assert_eq!(test::call_service(&app, request).await.status(), status);
    }
}
