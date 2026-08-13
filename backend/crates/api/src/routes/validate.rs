//! Length limits, enforced before a value reaches argon2 or the database.
//! Hashing a megabyte-long password is work an unauthenticated caller must not
//! be able to ask for.
//!
//! Counted in characters rather than bytes, so a limit means the same thing to
//! the person typing regardless of what they type.

use std::ops::RangeInclusive;

const USERNAME: RangeInclusive<usize> = 1..=64;
const PASSWORD: RangeInclusive<usize> = 8..=256;

pub fn username(value: &str) -> Result<(), String> {
    within("username", value, USERNAME)
}

pub fn password(value: &str) -> Result<(), String> {
    within("password", value, PASSWORD)
}

/// Upper bounds only, and one verdict for both: login must not reveal *which*
/// field was wrong.
pub fn plausible_credentials(username: &str, password: &str) -> bool {
    username.chars().count() <= *USERNAME.end() && password.chars().count() <= *PASSWORD.end()
}

fn within(field: &str, value: &str, bounds: RangeInclusive<usize>) -> Result<(), String> {
    let len = value.chars().count();
    if len < *bounds.start() {
        Err(format!("{field} must be at least {} character(s)", bounds.start()))
    } else if len > *bounds.end() {
        Err(format!("{field} must be at most {} character(s)", bounds.end()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_measured_in_characters_not_bytes() {
        assert!(username(&"é".repeat(64)).is_ok());
        assert!(username(&"é".repeat(65)).is_err());
    }

    #[test]
    fn the_message_names_the_field_and_the_bound() {
        let message = password("short").unwrap_err();
        assert!(message.contains("password") && message.contains("8"));
    }

    #[test]
    fn login_judges_both_fields_together() {
        assert!(plausible_credentials("someone", "x"));
        assert!(!plausible_credentials("someone", &"x".repeat(257)));
    }
}
