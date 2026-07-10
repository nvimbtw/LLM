---
id: ISSUE-0021
title: Missing input validation on user-facing route parameters (R10)
scope: backend
priority: P1
status: todo
rule: R10
location: backend/src/routes/user.rs:40-73 backend/src/routes/files.rs:22-93 backend/src/routes/chat.rs:191-225,476-493 backend/src/routes/model.rs:91-118 backend/src/routes/auth.rs:14-66
blocked-by: []
blocks: []
---

## Description
Several HTTP route handlers accept user input without validating size/format
boundaries: `insert_user`/`update_user` accept arbitrary-length usernames/passwords;
`upload_files` has no max file size; `handle_message`/`create_chat` have no max
content length; `update_model` has no name/description length limit; `login` has no
username/password length check. Malicious or oversized inputs can exhaust resources
or cause unexpected DB behavior. Validate at the boundary before passing to
DB/crypto layers.

## Comments
_No comments yet._
