use std::io::Write;

use crate::io::*;

pub fn encoder() {
    let vocab_path = "data/pairs.bin";
    let vocab = read_vocab(vocab_path).expect("Failed to read tokenizer binary file");
    let text = read_text_file("inputs/input.txt");
    let tokens = encode(text, vocab);
    write_tokens(&tokens, "data/tokens.bin").expect("Failed to write tokens to file");
}

pub fn encode(text: String, vocab: Vec<(u32, u32)>) -> Vec<u32> {
    let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
    println!("Input length before encoding: {}", tokens.len());

    for (merge_index, &(left, right)) in vocab.iter().enumerate() {
        std::io::stdout().flush().unwrap();

        let new_token_id = (256 + merge_index) as u32;
        let mut i = 0;
        while i < tokens.len() - 1 {
            if tokens[i] == left && tokens[i + 1] == right {
                tokens[i] = new_token_id;
                tokens.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    println!("Input length after encoding:  {}", tokens.len());

    tokens
}
