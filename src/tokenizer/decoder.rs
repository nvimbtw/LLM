use std::collections::HashMap;

pub fn decode(tokens: &[u32], vocab: &[(u32, u32)]) -> String {
    let mut cache: HashMap<u32, Vec<u8>> = HashMap::new();
    let mut all_bytes = Vec::new();

    for &token_id in tokens {
        all_bytes.extend(get_token_bytes(token_id, vocab, &mut cache));
    }

    String::from_utf8_lossy(&all_bytes).into_owned()
}

fn get_token_bytes(
    token_id: u32,
    vocab: &[(u32, u32)],
    cache: &mut HashMap<u32, Vec<u8>>,
) -> Vec<u8> {
    if let Some(bytes) = cache.get(&token_id) {
        return bytes.clone();
    }

    let bytes = if token_id < 256 {
        vec![token_id as u8]
    } else {
        let vocab_index = (token_id - 256) as usize;
        if vocab_index < vocab.len() {
            let (left, right) = vocab[vocab_index];
            let mut b = get_token_bytes(left, vocab, cache);
            b.extend(get_token_bytes(right, vocab, cache));
            b
        } else {
            // This case should ideally not happen with a correct vocab
            format!("[TOKEN:{}]", token_id).as_bytes().to_vec()
        }
    };

    cache.insert(token_id, bytes.clone());
    bytes
}
