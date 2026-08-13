use serde::{Deserialize, Serialize};

/// What a user is allowed to reach.
///
/// Lives here rather than in `db` because it is an authorization concept that
/// happens to be persisted, not a storage one. With it in `db`, token claims
/// carrying it forced the whole JWT module into the HTTP layer to avoid a
/// cycle — `db` needs this crate for password hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type, Serialize, Deserialize)]
#[sqlx(type_name = "access_level", rename_all = "lowercase")]
pub enum AccessLevel {
    User,
    Admin,
}

impl AccessLevel {
    pub fn is_admin(self) -> bool {
        matches!(self, AccessLevel::Admin)
    }
}
