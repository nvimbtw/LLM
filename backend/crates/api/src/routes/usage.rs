use actix_web::{HttpResponse, Responder, get, web};
use chrono::{Duration, Utc};
use serde::Serialize;
use sqlx::PgPool;
use std::collections::HashMap;

use crate::extractors::AuthUser;

// The usage page shows the trailing 30 days.
const WINDOW_DAYS: i32 = 30;

// These are local models with no real per-token billing, so cost is notional:
// a flat dollar rate per million tokens (input + output), purely for display.
const USD_PER_MILLION_TOKENS: f64 = 9.4;

#[derive(Serialize)]
struct UsageDay {
    date: String,
    input: i64,
    output: i64,
    requests: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelUsage {
    id: String,
    name: String,
    tokens: i64,
    requests: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct UsageResponse {
    total_requests: i64,
    total_tokens: i64,
    total_cost: f64,
    series: Vec<UsageDay>,
    by_model: Vec<ModelUsage>,
}

#[get("")]
async fn get_usage(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    let user_id = auth.claims.sub;
    let pool = pool.get_ref();

    let daily = match db::usage::daily_for_user(pool, user_id, WINDOW_DAYS).await {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("usage daily db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load usage");
        }
    };

    let by_model_rows = match db::usage::by_model_for_user(pool, user_id, WINDOW_DAYS).await {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("usage by-model db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load usage");
        }
    };

    // Build a contiguous series so the chart has every day, even empty ones.
    let by_day: HashMap<_, _> = daily.iter().map(|d| (d.day, d)).collect();
    let today = Utc::now().date_naive();
    let mut series = Vec::with_capacity(WINDOW_DAYS as usize);
    for i in (0..WINDOW_DAYS).rev() {
        let day = today - Duration::days(i as i64);
        match by_day.get(&day) {
            Some(d) => series.push(UsageDay {
                date: day.to_string(),
                input: d.input,
                output: d.output,
                requests: d.requests,
            }),
            None => series.push(UsageDay {
                date: day.to_string(),
                input: 0,
                output: 0,
                requests: 0,
            }),
        }
    }

    let total_requests: i64 = series.iter().map(|d| d.requests).sum();
    let total_tokens: i64 = series.iter().map(|d| d.input + d.output).sum();
    let total_cost =
        ((total_tokens as f64 / 1_000_000.0) * USD_PER_MILLION_TOKENS * 100.0).round() / 100.0;

    let by_model: Vec<ModelUsage> = by_model_rows
        .into_iter()
        .map(|m| ModelUsage {
            id: m.id_model.map(|id| id.to_string()).unwrap_or_else(|| "unknown".to_string()),
            name: m.name.unwrap_or_else(|| "Unknown model".to_string()),
            tokens: m.tokens,
            requests: m.requests,
        })
        .collect();

    HttpResponse::Ok().json(UsageResponse {
        total_requests,
        total_tokens,
        total_cost,
        series,
        by_model,
    })
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(get_usage);
}
