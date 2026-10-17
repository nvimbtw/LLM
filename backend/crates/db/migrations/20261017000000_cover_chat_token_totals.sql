CREATE INDEX messages_chat_tokens ON messages(chat_id) INCLUDE (tokens);
DROP INDEX messages_chat;
