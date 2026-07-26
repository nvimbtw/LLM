// Preference signal: how much of it exists, what is still waiting to be rated, and
// the rating itself.
//
// Two readers. The KTO composer asks `GET /signals` before it will let a run be
// queued — the gate that replaced discovering an empty dataset three hours into the
// queue. The rating panel on a gather job's page walks `GET /signals/pending` and
// writes verdicts back one at a time.
//
// Admin-only, matching /jobs: these ratings become training data for the one model
// on the one GPU, so rating is an authoring action, not a per-user preference.

use actix_web::{HttpResponse, Responder, delete, get, put, web};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::db;
use crate::extractors::AdminUser;
use crate::gather;

// How many unrated samples one request returns. A batch is 16, so this covers a
// couple of them — enough that the panel never runs dry mid-session without
// shipping a backlog nobody will work through.
const PENDING_LIMIT: i64 = 50;

// The readiness answer, in the shape the composer renders.
//
// **Two stages, reported separately**, because they are different work and the UI
// has to know which one is next. The seed is hand-rated replies, enough to
// calibrate the judge. The catalogue is everything a tune trains on — seed plus
// auto-labels — and it is two orders of magnitude larger. A response that
// collapsed them into one "total" is what let a thirty-example run start.
#[derive(Serialize)]
struct SignalStatus {
    // ── seed: what a person rated ──
    desirable: i64,
    undesirable: i64,
    total: i64,
    // Generated and waiting for a verdict. Drives "finish rating what you have"
    // rather than "gather more" — the same total can mean either.
    pending: i64,
    required: i64,
    required_per_class: i64,
    // Enough hand-rated replies to calibrate the judge, i.e. auto-labelling can run.
    seeded: bool,
    seed_reason: Option<String>,

    // ── catalogue: what a tune would train on ──
    catalogue: i64,
    catalogue_approved: i64,
    catalogue_rejected: i64,
    // How much of it came from auto-labelling. The rest is the seed, which stays in
    // the training data — it is the most trustworthy part of it.
    catalogue_auto: i64,
    catalogue_required: i64,
    // Epochs this catalogue allows. Surfaced so the form can say the run will be
    // clamped instead of the clamp being a surprise in the job's config.
    max_epochs: i64,

    // Whether a KTO run can start: both bars cleared. Computed here rather than by
    // comparing numbers client-side, so the button and the enqueue check cannot
    // disagree.
    ready: bool,
    // Why not, when it isn't. Null when ready.
    blocked_reason: Option<String>,
}

#[get("")]
async fn get_signals(_admin: AdminUser, pool: web::Data<PgPool>) -> impl Responder {
    let counts = match db::signal::counts(pool.get_ref()).await {
        Ok(counts) => counts,
        Err(e) => {
            eprintln!("signal counts db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load signal counts");
        }
    };

    let seed_reason = gather::check_signals(counts.desirable, counts.undesirable).err();

    // The catalogue is the seed plus whatever auto-labelling has written. The seed
    // is counted in because it is genuinely part of the training data — and the
    // best part of it, being the only labels a person actually gave.
    let (auto_up, auto_down) = crate::autolabel::catalogue_counts();
    let approved = counts.desirable + auto_up;
    let rejected = counts.undesirable + auto_down;
    let catalogue_reason = gather::check_catalogue(approved, rejected).err();

    // Seed first: rating unblocks auto-labelling, which fills the catalogue, so
    // reporting the shortfalls in that order names the step that is actually next.
    let blocked_reason = seed_reason.clone().or(catalogue_reason);

    HttpResponse::Ok().json(SignalStatus {
        desirable: counts.desirable,
        undesirable: counts.undesirable,
        total: counts.desirable + counts.undesirable,
        pending: counts.pending,
        required: gather::MIN_SIGNALS,
        required_per_class: gather::MIN_PER_CLASS,
        seeded: seed_reason.is_none(),
        seed_reason,
        catalogue: approved + rejected,
        catalogue_approved: approved,
        catalogue_rejected: rejected,
        catalogue_auto: auto_up + auto_down,
        catalogue_required: gather::MIN_CATALOGUE,
        max_epochs: gather::max_epochs_for(approved + rejected),
        ready: blocked_reason.is_none(),
        blocked_reason,
    })
}

#[derive(Serialize)]
struct SampleResponse {
    id_sample: Uuid,
    id_job: Option<Uuid>,
    prompt: String,
    response: String,
}

#[derive(Deserialize)]
struct PendingQuery {
    // Narrows to one batch. Omitted, the whole unrated backlog comes back — so a
    // batch abandoned halfway can be finished from the composer without hunting for
    // the job that produced it.
    id_job: Option<Uuid>,
}

#[get("/pending")]
async fn list_pending(
    _admin: AdminUser,
    query: web::Query<PendingQuery>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::signal::list_pending(pool.get_ref(), query.id_job, PENDING_LIMIT).await {
        Ok(samples) => HttpResponse::Ok().json(
            samples
                .into_iter()
                .map(|s| SampleResponse {
                    id_sample: s.id_sample,
                    id_job: s.id_job,
                    prompt: s.prompt,
                    response: s.response,
                })
                .collect::<Vec<_>>(),
        ),
        Err(e) => {
            eprintln!("list pending signals db error: {}", e);
            HttpResponse::InternalServerError().body("failed to load samples")
        }
    }
}

// `{"feedback": "up" | "down" | null}` — null clears the verdict and puts the
// sample back in the queue, same shape and same reasoning as the chat message
// endpoint.
#[derive(Deserialize)]
struct FeedbackPayload {
    feedback: Option<db::message::Feedback>,
}

#[put("/{id}")]
async fn rate_sample(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    payload: web::Json<FeedbackPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_sample = path.into_inner();
    match db::signal::set_feedback(pool.get_ref(), id_sample, payload.feedback).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("sample not found"),
        Err(e) => {
            eprintln!("rate signal db error: {}", e);
            HttpResponse::InternalServerError().body("failed to save the rating")
        }
    }
}

// Skip. A reply nobody can form a verdict on is not a verdict withheld — it is a
// reply with no signal in it, and leaving it pending would put it back in front of
// the next person to open the panel.
#[delete("/{id}")]
async fn skip_sample(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::signal::delete(pool.get_ref(), path.into_inner()).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("sample not found"),
        Err(e) => {
            eprintln!("skip signal db error: {}", e);
            HttpResponse::InternalServerError().body("failed to skip the sample")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // "pending" before "{id}", or the literal would be captured by the parameter.
    cfg.service(list_pending);
    cfg.service(get_signals);
    cfg.service(rate_sample);
    cfg.service(skip_sample);
}
