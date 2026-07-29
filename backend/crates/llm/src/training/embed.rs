use rand::prelude::*;

pub fn embed(dimensions: i64, vocab: i64) -> Vec<Vec<f32>> {
    let limit: f32 = (6.0 / (dimensions as f32 + vocab as f32)).sqrt();
    let mut rng = rand::thread_rng();

    let mut embedding_table: Vec<Vec<f32>> = vec![vec![0.0; dimensions as usize]; vocab as usize];

    for i in 0..vocab as usize {
        for j in 0..dimensions as usize {
            embedding_table[i][j] = rng.gen_range(-limit..limit);
        }
    }

    embedding_table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_dimensions() {
        let table = embed(10, 20);
        assert_eq!(table.len(), 20);
        assert_eq!(table[0].len(), 10);
    }

    #[test]
    fn test_embed_values_in_range() {
        let dims = 64;
        let vocab = 100;
        let limit: f32 = (6.0 / (dims as f32 + vocab as f32)).sqrt();
        let table = embed(dims, vocab);
        for (i, row) in table.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val >= -limit && val < limit,
                    "table[{}][{}] = {} out of range [-{:.6}, {:.6})",
                    i,
                    j,
                    val,
                    limit,
                    limit
                );
            }
        }
    }

    #[test]
    fn test_embed_not_all_zeros() {
        let table = embed(32, 50);
        let has_nonzero = table.iter().flatten().any(|&v| v != 0.0);
        assert!(has_nonzero, "embedding table should have non-zero values");
    }

    #[test]
    fn test_embed_range_at_various_sizes() {
        for (d, v) in &[(10, 20), (64, 100), (128, 500), (4, 10)] {
            let limit: f32 = (6.0 / (*d as f32 + *v as f32)).sqrt();
            let table = embed(*d, *v);
            assert_eq!(table.len(), *v as usize);
            assert_eq!(table[0].len(), *d as usize);
            for row in &table {
                for &val in row {
                    assert!(
                        val >= -limit && val < limit,
                        "val {} out of range for d={} v={}",
                        val,
                        d,
                        v
                    );
                }
            }
        }
    }
}
