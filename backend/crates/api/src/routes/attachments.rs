//! Files a person attached to a turn. The engine owns them, for the same reason
//! it owns the messages they hang off; this hop proves who is asking and carries
//! bytes rather than JSON.
//!
//! The one route pair in this crate that is not a `Value` in and a `Value` out.
//! An upload arrives as multipart from the browser and leaves as multipart to
//! the service; a download comes back as bytes with the headers the service set,
//! because only the service knows what the file is.

use actix_multipart::MultipartError;
use actix_multipart::form::{MultipartForm, MultipartFormConfig, tempfile::TempFile};
use actix_web::{HttpRequest, HttpResponse, delete, error::InternalError, get, http::header, post, web};
use engine::{FilePart, Proxy};
use uuid::Uuid;

use crate::extractors::AuthUser;
use crate::upstream::{Answer, caller};
use service::Error as Upstreamed;

/// The multipart field the engine reads an upload from.
const FILE_FIELD: &str = "file";

/// What a file is when the browser said nothing useful about it.
const UNKNOWN_MEDIA_TYPE: &str = "application/octet-stream";

/// The ceiling, in the unit the refusal is written in. Kept in step with
/// `MAX_BYTES` in `engine/api/routes/attachments.py` — and with the `limit` on
/// the field below, which a macro forces to be a literal.
const MAX_MB: usize = 50;

/// Refused here as well as in the engine, so an oversized body is dropped at the
/// edge instead of being read into memory twice on its way to being refused.
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
    engine: web::Data<Proxy>,
) -> Answer {
    let part = read_part(form.file)?;
    let created = engine.upload(&caller(&user), "/attachments", part).await?;
    Ok(HttpResponse::Created().json(created))
}

#[get("/{attachment_id}")]
async fn show(user: AuthUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    let blob = engine
        .fetch(&caller(&user), &format!("/attachments/{id}"))
        .await?;

    let mut response = HttpResponse::Ok();
    response.content_type(blob.media_type);
    if let Some(disposition) = blob.disposition {
        response.insert_header((header::CONTENT_DISPOSITION, disposition));
    }
    Ok(response.body(blob.bytes))
}

#[delete("/{attachment_id}")]
async fn destroy(user: AuthUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    engine
        .delete(&caller(&user), &format!("/attachments/{id}"))
        .await?;
    Ok(HttpResponse::NoContent().finish())
}

/// The uploaded file as something that can be forwarded.
///
/// Actix spools a part to a temporary file, so this is where it comes back into
/// memory — and therefore where the ceiling is enforced a second time. The
/// `limit` on the field already refuses a larger one; this covers the case of
/// the two numbers drifting apart, which would otherwise be a silent OOM.
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

    Ok(FilePart {
        field: FILE_FIELD.to_string(),
        // The engine sanitizes this into a label; nothing on this side opens it.
        filename: file.file_name.unwrap_or_else(|| "file".to_string()),
        media_type: file
            .content_type
            .map(|mime| mime.to_string())
            .unwrap_or_else(|| UNKNOWN_MEDIA_TYPE.to_string()),
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
/// turns the one refusal that actually happens into the status and the sentence
/// the engine would have given it.
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
