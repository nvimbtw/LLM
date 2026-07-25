use crate::bpe::*;
#[test]
fn test_overlap() {
    let pre_tokenized = vec![vec!['a', 'a', 'a']];
    let (vocab, corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();
    println!("Merges: {:?}", merges);
    let encoded = encode("aaa", &merges);
    println!("Encoded: {:?}", encoded);
    println!("Corpus: {:?}", corpus);
    // corpus should match encoded exactly!
    assert_eq!(encoded, corpus[0]);
}
