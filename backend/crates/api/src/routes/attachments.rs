//! Owner-scoped uploads stored with conversations in the backend database.
use crate::conversation::{attachment, database, failure};
use actix_multipart::form::{tempfile::TempFile, MultipartForm, MultipartFormConfig};
use actix_multipart::MultipartError;
use actix_web::{
    delete, error::InternalError, get, http::header, post, web, HttpRequest, HttpResponse,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::extractors::AuthUser;
use crate::upstream::Answer;
use service::Error as Upstreamed;

/// What a file is when the browser said nothing useful about it.
const UNKNOWN_MEDIA_TYPE: &str = "application/octet-stream";

/// Keep the multipart literal and database size constraint aligned.
const MAX_MB: usize = 50;

/// Enforced before loading the spooled file into memory.
const MAX_BYTES: usize = MAX_MB * 1024 * 1024;

/// The whole form, which actix limits separately and by default to the same
/// 50 MiB a single field is now allowed.
///
/// Left at the default, a file at the ceiling would be refused by the *form*
/// limit a byte before the field limit could refuse it — same status, but the
/// message would be about a form the person never saw. The headroom is what
/// makes the field limit the one that always speaks.
const FORM_TOTAL_BYTES: usize = MAX_BYTES + 1024 * 1024;

#[derive(MultipartForm)]
struct Upload {
    // A literal because the macro takes one; `MAX_MB` is the same number and
    // the two move together.
    #[multipart(rename = "file", limit = "50MiB")]
    file: TempFile,
}

#[post("")]
async fn create(
    user: AuthUser,
    MultipartForm(form): MultipartForm<Upload>,
    pool: web::Data<PgPool>,
) -> Answer {
    let part = read_part(form.file)?;
    let created = db::attachment::create(
        &pool,
        user.claims.sub,
        &part.filename,
        &part.media_type,
        &part.bytes,
    )
    .await
    .map_err(database)?;
    Ok(HttpResponse::Created().json(attachment(&created)))
}

#[get("/{attachment_id}")]
async fn show(user: AuthUser, id: web::Path<Uuid>, pool: web::Data<PgPool>) -> Answer {
    let (metadata, bytes) = db::attachment::get(&pool, user.claims.sub, *id)
        .await
        .map_err(database)?
        .ok_or_else(|| failure(404, "attachment not found"))?;
    Ok(HttpResponse::Ok()
        .content_type(metadata.media_type)
        .insert_header(header::ContentDisposition {
            disposition: header::DispositionType::Inline,
            parameters: vec![header::DispositionParam::FilenameExt(
                header::ExtendedValue {
                    charset: header::Charset::Ext("UTF-8".into()),
                    language_tag: None,
                    value: metadata.filename.into_bytes(),
                },
            )],
        })
        .body(bytes))
}

#[delete("/{attachment_id}")]
async fn destroy(user: AuthUser, id: web::Path<Uuid>, pool: web::Data<PgPool>) -> Answer {
    if !db::attachment::discard(&pool, user.claims.sub, *id)
        .await
        .map_err(database)?
    {
        return Err(failure(404, "unsent attachment not found").into());
    }
    Ok(HttpResponse::NoContent().finish())
}

fn clean_filename(name: &str) -> String {
    let label: String = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("file")
        .chars()
        .filter(|c| !c.is_control())
        .take(255)
        .collect();
    if label.is_empty() {
        "file".into()
    } else {
        label
    }
}

/// Read the spooled upload after enforcing its size limit.
///
/// Actix spools a part to a temporary file, so this is where it comes back into
/// memory — and therefore where the ceiling is enforced a second time. The
/// `limit` on the field already refuses a larger one; this covers the case of
/// the two numbers drifting apart, which would otherwise be a silent OOM.
struct FilePart {
    filename: String,
    media_type: String,
    bytes: Vec<u8>,
}

fn read_part(file: TempFile) -> Result<FilePart, Upstreamed> {
    let size = file.size;
    if size == 0 {
        return Err(over(400, "an empty file is not an attachment"));
    }
    if size > MAX_BYTES {
        return Err(over(413, &oversized()));
    }

    let bytes = std::fs::read(file.file.path())
        .map_err(|e| Upstreamed::Transport(format!("reading the upload: {e}")))?;

    let media_type = file
        .content_type
        .map(|mime| mime.to_string())
        .unwrap_or_else(|| UNKNOWN_MEDIA_TYPE.to_string());
    if media_type.len() > 160 {
        return Err(over(400, "attachment media type is too long"));
    }
    Ok(FilePart {
        filename: clean_filename(file.file_name.as_deref().unwrap_or("file")),
        media_type,
        bytes,
    })
}

/// Built rather than written down, so the sentence cannot disagree with the
/// constant it describes — which is the failure mode a duplicated message has.
fn oversized() -> String {
    format!("an attachment may not exceed {MAX_MB} MB")
}

/// A refusal this side reached on its own, spelled as an upstream status so it
/// reaches the client through the same path a service's would.
fn over(code: u16, detail: &str) -> Upstreamed {
    Upstreamed::Status {
        code,
        detail: detail.to_string(),
    }
}

/// What the browser is told when a part is refused before this crate sees it.
///
/// Actix enforces the field limit while it is still reading the body, so an
/// oversized upload never reaches `read_part` and would otherwise come back as
/// a bare 400 "Payload error" — which says nothing a person can act on. This
/// returns the attachment limit with a useful status and message.
fn refusal(error: MultipartError, _: &HttpRequest) -> actix_web::Error {
    // `Payload(Overflow)`, not `Field` — actix raises the limit from inside the
    // reader (`form::Limits::try_consume_limits`), so the field it belongs to is
    // no longer in the error by the time it surfaces.
    let over_limit = matches!(
        error,
        MultipartError::Payload(actix_web::error::PayloadError::Overflow)
    );
    let response = if over_limit {
        HttpResponse::PayloadTooLarge().json(serde_json::json!({ "error": oversized() }))
    } else {
        HttpResponse::BadRequest().json(serde_json::json!({ "error": error.to_string() }))
    };
    InternalError::from_response(error, response).into()
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.app_data(
        MultipartFormConfig::default()
            .total_limit(FORM_TOTAL_BYTES)
            .error_handler(refusal),
    );
    cfg.service(create);
    cfg.service(show);
    cfg.service(destroy);
}
