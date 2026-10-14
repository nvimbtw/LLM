//! Private project metadata and paginated chat membership owned by the authenticated user.

use crate::{conversation as c, extractors::AuthUser, upstream::Answer};
use actix_web::{delete, get, patch, post, web, HttpResponse};
use db::project::{self as queries, Project};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct Page {
    #[serde(default = "default_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}

/// Supply the default page size for omitted pagination parameters.
fn default_limit() -> i64 {
    50
}

impl Page {
    /// Validate an effective page budget of 1–200 rows and a nonnegative signed 32-bit offset.
    fn validate(&self) -> Result<(), service::Error> {
        if !(1..=200).contains(&self.limit) || !(0..=i64::from(i32::MAX)).contains(&self.offset) {
            return Err(c::failure(400, "invalid project pagination"));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fields {
    name: Option<String>,
    description: Option<String>,
    notes: Option<String>,
}

impl Fields {
    /// Check project field lengths and require a nonblank name on creation or whenever supplied.
    fn validate(&self, creating: bool) -> Result<(), service::Error> {
        if creating && self.name.is_none() {
            return Err(c::failure(400, "project name is required"));
        }
        if self
            .name
            .as_ref()
            .is_some_and(|name| name.trim().is_empty() || name.chars().count() > 120)
        {
            return Err(c::failure(
                400,
                "project name must contain 1 to 120 characters",
            ));
        }
        for (value, max, detail) in [
            (&self.description, 2000, "project description is too long"),
            (&self.notes, 8000, "project notes are too long"),
        ] {
            if value
                .as_ref()
                .is_some_and(|value| value.chars().count() > max)
            {
                return Err(c::failure(400, detail));
            }
        }
        Ok(())
    }
}

/// Project stored metadata into the public contract without exposing internal ownership.
fn metadata(row: &Project) -> Value {
    json!({"project_id":row.id,"name":row.name,"description":row.description,"notes":row.notes,
        "thread_count":row.thread_count,"created_at":row.created_at,"updated_at":row.updated_at})
}

/// Preserve safe project failure messages while logging internal database causes.
fn database(error: sqlx::Error) -> service::Error {
    eprintln!("project storage: {error}");
    match error {
        sqlx::Error::RowNotFound => c::failure(404, "project not found"),
        _ => c::failure(500, "project storage failed"),
    }
}

/// Return owned project metadata or the same 404 used for an unknown ID.
async fn owned(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<Project, service::Error> {
    queries::get(pool, owner, id)
        .await
        .map_err(database)?
        .ok_or_else(|| c::failure(404, "project not found"))
}

/// List a bounded page of only the authenticated user's projects and their total count.
#[get("")]
async fn list(user: AuthUser, page: web::Query<Page>, pool: web::Data<PgPool>) -> Answer {
    page.validate()?;
    let (rows, total) = queries::list(&pool, user.claims.sub, page.limit, page.offset)
        .await
        .map_err(database)?;
    Ok(HttpResponse::Ok().json(
        json!({"projects":rows.iter().map(metadata).collect::<Vec<_>>(),
        "total":total,"limit":page.limit,"offset":page.offset}),
    ))
}

/// Create a private project from validated metadata and acknowledge committed storage with 201.
#[post("")]
async fn create(user: AuthUser, fields: web::Json<Fields>, pool: web::Data<PgPool>) -> Answer {
    fields.validate(true)?;
    let row = queries::create(
        &pool,
        user.claims.sub,
        fields.name.as_deref().unwrap_or_default(),
        fields.description.as_deref().unwrap_or_default(),
        fields.notes.as_deref().unwrap_or_default(),
    )
    .await
    .map_err(database)?;
    Ok(HttpResponse::Created().json(metadata(&row)))
}

/// Return owned project metadata and a bounded chat page from one consistent database snapshot.
#[get("/{project_id}")]
async fn get_one(
    user: AuthUser,
    id: web::Path<Uuid>,
    page: web::Query<Page>,
    pool: web::Data<PgPool>,
) -> Answer {
    page.validate()?;
    let detail = queries::detail(&pool, user.claims.sub, *id, page.limit, page.offset)
        .await
        .map_err(database)?
        .ok_or_else(|| c::failure(404, "project not found"))?;
    let mut result = metadata(&detail.project);
    result["threads"] = json!(detail.threads.iter().map(c::chat).collect::<Vec<_>>());
    result["threads_total"] = json!(detail.project.thread_count);
    result["threads_limit"] = json!(page.limit);
    result["threads_offset"] = json!(page.offset);
    Ok(HttpResponse::Ok().json(result))
}

/// Apply validated partial metadata only to an owned project and return its current projection.
#[patch("/{project_id}")]
async fn update(
    user: AuthUser,
    id: web::Path<Uuid>,
    fields: web::Json<Fields>,
    pool: web::Data<PgPool>,
) -> Answer {
    fields.validate(false)?;
    if !queries::update(
        &pool,
        user.claims.sub,
        *id,
        fields.name.as_deref(),
        fields.description.as_deref(),
        fields.notes.as_deref(),
    )
    .await
    .map_err(database)?
    {
        return Err(c::failure(404, "project not found").into());
    }
    Ok(HttpResponse::Ok().json(metadata(&owned(&pool, user.claims.sub, *id).await?)))
}

/// Delete an owned project after detaching its chats in the same database action; messages remain stored.
#[delete("/{project_id}")]
async fn remove(user: AuthUser, id: web::Path<Uuid>, pool: web::Data<PgPool>) -> Answer {
    if !queries::remove(&pool, user.claims.sub, *id)
        .await
        .map_err(database)?
    {
        return Err(c::failure(404, "project not found").into());
    }
    Ok(HttpResponse::NoContent().finish())
}

/// Register authenticated private-project CRUD routes on their owning scope.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list)
        .service(create)
        .service(get_one)
        .service(update)
        .service(remove);
}
