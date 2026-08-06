use std::path::Path;

use actix_multipart::Multipart;
use actix_web::{HttpResponse, Responder, post, web};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::extractors::AdminUser;
use crate::routes::validate;
use storage;

// Formats the ingestion step can process. Mirrors the frontend's accept list;
// validated again here since the extractor is the only real trust boundary.
const ALLOWED_EXTENSIONS: [&str; 3] = ["txt", "pdf", "docx"];

// Accepts one or more files as multipart/form-data (any field name) and writes
// each to INPUT_DIR. The `AdminUser` extractor gates the whole handler: a
// non-admin never reaches this body. Names are discarded — files are throwaway
// inputs — so each gets a random UUID name (collision-proof when two uploads
// share a filename); only the extension is kept so the pipeline knows the type.
#[post("")]
async fn upload_files(_admin: AdminUser, mut payload: Multipart) -> impl Responder {
    // Raw uploads land here for the ingestion pipeline to pick up later.
    // Created lazily so a fresh checkout / custom DATA_DIR still works.
    let uploads_dir = storage::uploads_dir();
    if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
        eprintln!("failed to create uploads dir: {}", e);
        return HttpResponse::InternalServerError().body("failed to store files");
    }

    let mut saved = 0u32;

    while let Some(field) = payload.next().await {
        let mut field = match field {
            Ok(field) => field,
            Err(e) => {
                eprintln!("multipart read error: {}", e);
                return HttpResponse::BadRequest().body("malformed upload");
            }
        };

        // Pull the extension from the client-supplied filename. A missing or
        // unsupported type is a client error → 400.
        let ext = field
            .content_disposition()
            .and_then(|cd| cd.get_filename())
            .and_then(|name| Path::new(name).extension())
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let ext = match ext {
            Some(ext) if ALLOWED_EXTENSIONS.contains(&ext.as_str()) => ext,
            _ => return HttpResponse::BadRequest().body("unsupported or missing file type"),
        };

        let path = uploads_dir.join(format!("{}.{}", Uuid::new_v4(), ext));

        let mut file = match tokio::fs::File::create(&path).await {
            Ok(file) => file,
            Err(e) => {
                eprintln!("failed to create {}: {}", path.display(), e);
                return HttpResponse::InternalServerError().body("failed to store files");
            }
        };

        // Stream chunks straight to disk instead of buffering the whole file.
        // On any failure mid-write, drop the partial file so the pipeline never
        // ingests a truncated input.
        let mut written: u64 = 0;
        while let Some(chunk) = field.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    eprintln!("multipart chunk error: {}", e);
                    let _ = tokio::fs::remove_file(&path).await;
                    return HttpResponse::BadRequest().body("malformed upload");
                }
            };
            // Cap per-file size so a client can't fill the disk; drop the partial
            // file and reject with 413 the moment the limit is crossed.
            written += chunk.len() as u64;
            if written > validate::MAX_UPLOAD_BYTES {
                let _ = tokio::fs::remove_file(&path).await;
                return HttpResponse::PayloadTooLarge().body("file exceeds maximum size");
            }
            if let Err(e) = file.write_all(&chunk).await {
                eprintln!("failed to write {}: {}", path.display(), e);
                let _ = tokio::fs::remove_file(&path).await;
                return HttpResponse::InternalServerError().body("failed to store files");
            }
        }

        saved += 1;
    }

    if saved == 0 {
        return HttpResponse::BadRequest().body("no files provided");
    }

    HttpResponse::Created().body(format!("{} file(s) uploaded", saved))
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(upload_files);
}
