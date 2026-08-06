//! Model persistence: the versioned `GORK` binary format (shared by the
//! file-backed `save`/`load` and in-memory `to_bytes`/`from_bytes` paths via a
//! generic reader/writer) plus the `LayerRow` inspection format used by
//! `to_layer_rows`/`from_layer_rows`.

use std::io::Read;

use super::Model;

pub struct LayerRow {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

#[derive(serde::Serialize, Clone)]
pub struct ModelHeader {
    pub dimensions: i64,
    pub vocab_size: i64,
    pub max_seq_len: i64,
    pub heads: i64,
    pub ff_multiplier: i64,
    pub layers: i64,
}

/// Read just the fixed-size header of a saved model (`model.bin` in `dir`)

pub fn read_model_header(dir: &str) -> Result<ModelHeader, std::io::Error> {
    let path = format!("{}/model.bin", dir);
    let mut f = std::fs::File::open(&path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GORK" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let _version = u32::from_le_bytes(buf);
    let read_i64 = |f: &mut std::fs::File| -> Result<i64, std::io::Error> {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    };
    let dimensions = read_i64(&mut f)?;
    let vocab_size = read_i64(&mut f)?;
    let max_seq_len = read_i64(&mut f)?;
    let heads = read_i64(&mut f)?;
    let ff_multiplier = read_i64(&mut f)?;
    let layers = read_i64(&mut f)?;
    validate_header_fields(dimensions, vocab_size, max_seq_len, heads, ff_multiplier, layers)?;
    Ok(ModelHeader { dimensions, vocab_size, max_seq_len, heads, ff_multiplier, layers })
}

// A model header comes from a file (external input), so its fields are validated
// at the boundary rather than trusted. A non-positive field would produce a
// negative-sized allocation, and heads not dividing dimensions (or being zero)
// would make `dimensions / heads` panic or compute a wrong head_dim.
fn validate_header_fields(
    dimensions: i64,
    vocab: i64,
    ctx: i64,
    heads: i64,
    ff_multiplier: i64,
    layers: i64,
) -> Result<(), std::io::Error> {
    let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    for (name, value) in [
        ("dimensions", dimensions),
        ("vocab_size", vocab),
        ("max_seq_len", ctx),
        ("heads", heads),
        ("ff_multiplier", ff_multiplier),
        ("layers", layers),
    ] {
        if value <= 0 {
            return Err(invalid(format!(
                "model header field `{}` must be positive, got {}",
                name, value
            )));
        }
    }
    if dimensions % heads != 0 {
        return Err(invalid(format!(
            "model header: dimensions ({}) must be divisible by heads ({})",
            dimensions, heads
        )));
    }
    Ok(())
}

/// A decoder-only transformer held as plain nested `Vec`s on the host. This is
/// the canonical CPU-side copy that gets (de)serialized and used for inference;
/// during training the weights are mirrored to GPU buffers in `Workspace`.

const MODEL_MAGIC: &[u8; 4] = b"GORK";
/// v4 dropped the learned absolute positional table (RoPE replaced it). v3 files
/// still parse — `read_model` reads the table body and discards it — but they load
/// with a warning; see the note there for why they cannot generate sensibly.
const MODEL_VERSION: u32 = 5;
// First version with RoPE instead of a learned absolute positional table. Below
// it the file still carries that table's body, which has to be read and dropped
// or every weight after it is misaligned.
const ROPE_VERSION: u32 = 4;
// First version carrying the final layer-norm's gamma/beta (before `w_output`).
const FINAL_LN_VERSION: u32 = 5;

fn write_2d<W: std::io::Write>(w: &mut W, m: &[Vec<f32>]) -> Result<(), std::io::Error> {
    w.write_all(&(m.len() as u64).to_le_bytes())?;
    w.write_all(&(m[0].len() as u64).to_le_bytes())?;
    for row in m {
        for &v in row {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_1d<W: std::io::Write>(w: &mut W, v: &[f32]) -> Result<(), std::io::Error> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn write_model<W: std::io::Write>(t: &Model, w: &mut W) -> Result<(), std::io::Error> {
    w.write_all(MODEL_MAGIC)?;
    w.write_all(&MODEL_VERSION.to_le_bytes())?;
    let d = t.wo[0].len() as i64;
    let vocab = t.w_output[0].len() as i64;
    let ctx = t.max_seq_len;
    let ff = t.w1[0][0].len() as i64;
    for field in [d, vocab, ctx, t.heads, ff, t.layers] {
        w.write_all(&field.to_le_bytes())?;
    }
    write_2d(w, &t.embedding_table)?;
    for layer in 0..t.layers as usize {
        for head in 0..t.heads as usize {
            write_2d(w, &t.wq[layer][head])?;
            write_2d(w, &t.wk[layer][head])?;
            write_2d(w, &t.wv[layer][head])?;
        }
    }
    for mat in &t.wo { write_2d(w, mat)?; }
    for mat in &t.w1 { write_2d(w, mat)?; }
    for mat in &t.w2 { write_2d(w, mat)?; }
    for v in &t.b1 { write_1d(w, v)?; }
    for v in &t.b2 { write_1d(w, v)?; }
    for v in &t.ln_gamma { write_1d(w, v)?; }
    for v in &t.ln_beta { write_1d(w, v)?; }
    // Empty when the model was loaded from a pre-v5 file: the file is stamped v5
    // (this is the v5 layout) but records that it carries no final norm, so a
    // fine-tune of an old base keeps saving as the model it actually is.
    write_1d(w, &t.ln_final_gamma)?;
    write_1d(w, &t.ln_final_beta)?;
    write_2d(w, &t.w_output)?;
    Ok(())
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32, std::io::Error> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i64<R: std::io::Read>(r: &mut R) -> Result<i64, std::io::Error> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_u64<R: std::io::Read>(r: &mut R) -> Result<u64, std::io::Error> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_2d<R: std::io::Read>(r: &mut R) -> Result<Vec<Vec<f32>>, std::io::Error> {
    let rows = read_u64(r)? as usize;
    let cols = read_u64(r)? as usize;
    let mut buf = vec![0u8; rows * cols * 4];
    r.read_exact(&mut buf)?;
    // chunks_exact(4) yields exactly-4-byte slices, so try_into can't fail here.
    let data: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(data.chunks(cols).map(|c| c.to_vec()).collect())
}

fn read_1d<R: std::io::Read>(r: &mut R) -> Result<Vec<f32>, std::io::Error> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len * 4];
    r.read_exact(&mut buf)?;
    // chunks_exact(4) yields exactly-4-byte slices, so try_into can't fail here.
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

struct ModelFileHeader {
    version: u32,
    heads: i64,
    head_dim: i64,
    layers: i64,
    max_seq_len: i64,
}

// Parse and validate the fixed-size header: magic, version, and the six i64
// dimension fields. `layers` was added in v2; v1 files are assumed to be 6-layer.
fn read_header<R: std::io::Read>(r: &mut R) -> Result<ModelFileHeader, std::io::Error> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MODEL_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = read_u32(r)?;
    let dimensions = read_i64(r)?;
    let vocab = read_i64(r)?;
    let ctx = read_i64(r)?;
    let heads = read_i64(r)?;
    let ff_multiplier = read_i64(r)?;
    let layers = if version >= 2 { read_i64(r)? } else { 6 };
    validate_header_fields(dimensions, vocab, ctx, heads, ff_multiplier, layers)?;
    Ok(ModelFileHeader {
        version,
        heads,
        head_dim: dimensions / heads,
        layers,
        max_seq_len: ctx,
    })
}

// Per-head Q/K/V weights. v3 stores them per layer ([layers][heads][d][hd]);
// legacy v1/v2 stored a single layer's weights, broadcast to every layer.
#[allow(clippy::type_complexity)]
fn read_qkv_weights<R: std::io::Read>(
    r: &mut R,
    version: u32,
    layers: usize,
    heads: usize,
) -> Result<(Vec<Vec<Vec<Vec<f32>>>>, Vec<Vec<Vec<Vec<f32>>>>, Vec<Vec<Vec<Vec<f32>>>>), std::io::Error>
{
    let read_layer = |r: &mut R| -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>), std::io::Error> {
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        for _ in 0..heads {
            wq.push(read_2d(r)?);
            wk.push(read_2d(r)?);
            wv.push(read_2d(r)?);
        }
        Ok((wq, wk, wv))
    };

    let mut wq = Vec::new();
    let mut wk = Vec::new();
    let mut wv = Vec::new();
    if version >= 3 {
        for _ in 0..layers {
            let (lq, lk, lv) = read_layer(r)?;
            wq.push(lq);
            wk.push(lk);
            wv.push(lv);
        }
    } else {
        let (sq, sk, sv) = read_layer(r)?;
        for _ in 0..layers {
            wq.push(sq.clone());
            wk.push(sk.clone());
            wv.push(sv.clone());
        }
    }
    Ok((wq, wk, wv))
}

struct LayerBlockWeights {
    wo: Vec<Vec<Vec<f32>>>,
    w1: Vec<Vec<Vec<f32>>>,
    w2: Vec<Vec<Vec<f32>>>,
    b1: Vec<Vec<f32>>,
    b2: Vec<Vec<f32>>,
    ln_gamma: Vec<Vec<f32>>,
    ln_beta: Vec<Vec<f32>>,
}

// The per-layer block tensors (wo, w1, w2, biases, layer-norm scale/shift). Same
// v3-per-layer vs legacy-broadcast rule as the QKV weights.
fn read_block_weights<R: std::io::Read>(
    r: &mut R,
    version: u32,
    layers: usize,
) -> Result<LayerBlockWeights, std::io::Error> {
    let mut b = LayerBlockWeights {
        wo: Vec::new(), w1: Vec::new(), w2: Vec::new(),
        b1: Vec::new(), b2: Vec::new(), ln_gamma: Vec::new(), ln_beta: Vec::new(),
    };
    if version >= 3 {
        for _ in 0..layers { b.wo.push(read_2d(r)?); }
        for _ in 0..layers { b.w1.push(read_2d(r)?); }
        for _ in 0..layers { b.w2.push(read_2d(r)?); }
        for _ in 0..layers { b.b1.push(read_1d(r)?); }
        for _ in 0..layers { b.b2.push(read_1d(r)?); }
        for _ in 0..layers { b.ln_gamma.push(read_1d(r)?); }
        for _ in 0..layers { b.ln_beta.push(read_1d(r)?); }
    } else {
        let (wo, w1, w2) = (read_2d(r)?, read_2d(r)?, read_2d(r)?);
        let (b1, b2) = (read_1d(r)?, read_1d(r)?);
        let (ln_gamma, ln_beta) = (read_1d(r)?, read_1d(r)?);
        for _ in 0..layers {
            b.wo.push(wo.clone());
            b.w1.push(w1.clone());
            b.w2.push(w2.clone());
            b.b1.push(b1.clone());
            b.b2.push(b2.clone());
            b.ln_gamma.push(ln_gamma.clone());
            b.ln_beta.push(ln_beta.clone());
        }
    }
    Ok(b)
}

#[cfg(test)]
mod final_ln_roundtrip_tests {
    use super::*;
    use crate::training::transformer::Model;

    fn roundtrip(t: &Model) -> Model {
        let mut buf = Vec::new();
        write_model(t, &mut buf).expect("write");
        read_model(&mut buf.as_slice()).expect("read")
    }

    #[test]
    fn a_fresh_model_keeps_its_final_layer_norm() {
        let t = Model::init(16, 32, 8, 2, 2, 2);
        let back = roundtrip(&t);
        assert!(back.has_final_ln());
        assert_eq!(back.ln_final_gamma, t.ln_final_gamma);
        assert_eq!(back.ln_final_beta, t.ln_final_beta);
    }

    // The case that matters for the models already on disk: a v4 base loads with
    // no final norm, and saving it (as a fine-tune does) must not invent one.
    // Writing the length is what makes "absent" survive the v5 layout.
    #[test]
    fn a_model_without_a_final_layer_norm_stays_without_one() {
        let mut t = Model::init(16, 32, 8, 2, 2, 2);
        t.ln_final_gamma = Vec::new();
        t.ln_final_beta = Vec::new();

        let back = roundtrip(&t);
        assert!(!back.has_final_ln());
        assert!(back.ln_final_beta.is_empty());
    }

    #[test]
    fn a_wrongly_sized_final_layer_norm_is_rejected() {
        let mut t = Model::init(16, 32, 8, 2, 2, 2);
        t.ln_final_gamma = vec![1.0; 15];
        let mut buf = Vec::new();
        write_model(&t, &mut buf).expect("write");
        assert!(read_model(&mut buf.as_slice()).is_err());
    }
}

fn read_model<R: std::io::Read>(r: &mut R) -> Result<Model, std::io::Error> {
    let header = read_header(r)?;
    let embedding_table = read_2d(r)?;
    // v3 and earlier stored a learned absolute positional table here. RoPE
    // replaced it, so the body is read and thrown away — the weights after it sit
    // at a fixed offset, and skipping the read would misalign every one of them.
    //
    // The load is allowed (the model browser and metadata paths must keep working
    // on old runs) but it cannot produce sensible output: such a model was trained
    // under absolute positions this code no longer applies, and — since the
    // attention backward was broken for the entire period these files were
    // written — with a K projection still at its random initialization. The
    // warning is what stops that being a silent wrong answer.
    if header.version < ROPE_VERSION {
        let _ = read_2d(r)?;
        eprintln!(
            "warning: v{} model (current is v{MODEL_VERSION}) — trained with absolute \
             positional embeddings and an untrained K projection. Its output will be \
             meaningless; re-pretrain.",
            header.version,
        );
    }
    let (wq, wk, wv) = read_qkv_weights(r, header.version, header.layers as usize, header.heads as usize)?;
    let blocks = read_block_weights(r, header.version, header.layers as usize)?;
    // Absent before v5, and legitimately empty in a v5 file written from a model
    // that was itself loaded from v4 — the length in the file is what says which,
    // so an empty pair round-trips as "no final norm" rather than being inferred.
    let (ln_final_gamma, ln_final_beta) = if header.version >= FINAL_LN_VERSION {
        (read_1d(r)?, read_1d(r)?)
    } else {
        (Vec::new(), Vec::new())
    };
    let d = blocks.ln_gamma.first().map_or(0, Vec::len);
    if !ln_final_gamma.is_empty() && (ln_final_gamma.len() != d || ln_final_beta.len() != d) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "final layer-norm is {}/{} wide, expected {d}",
                ln_final_gamma.len(),
                ln_final_beta.len(),
            ),
        ));
    }
    let w_output = read_2d(r)?;
    Ok(Model {
        embedding_table,
        max_seq_len: header.max_seq_len,
        wq, wk, wv,
        wo: blocks.wo,
        heads: header.heads,
        head_dim: header.head_dim,
        layers: header.layers,
        w1: blocks.w1,
        w2: blocks.w2,
        b1: blocks.b1,
        b2: blocks.b2,
        ln_gamma: blocks.ln_gamma,
        ln_beta: blocks.ln_beta,
        ln_final_gamma,
        ln_final_beta,
        w_output,
    })
}


impl Model {
    /// Serialize the model to `path` in the versioned `GORK` binary format.
    pub fn save_to_path(&self, path: &str) -> Result<(), std::io::Error> {
        let mut f = std::fs::File::create(path)?;
        write_model(self, &mut f)
    }

    /// Serialize the model to `model.bin` inside `dir`.
    pub fn save(&self, dir: &str) -> Result<(), std::io::Error> {
        self.save_to_path(&format!("{}/model.bin", dir))
    }

    /// Load a model from `model.bin` inside `dir`, accepting any supported format
    /// version. Errors on bad magic, truncation, or an invalid header.
    pub fn load(dir: &str) -> Result<Self, std::io::Error> {
        let mut f = std::fs::File::open(format!("{}/model.bin", dir))?;
        read_model(&mut f)
    }
}

impl Model {
    /// Serialize the model to an in-memory buffer (same format as `save`).
    pub fn to_bytes(&self) -> Result<Vec<u8>, std::io::Error> {
        let mut buf = Vec::new();
        write_model(self, &mut buf)?;
        Ok(buf)
    }

    /// Deserialize a model from a byte buffer produced by `to_bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, std::io::Error> {
        read_model(&mut std::io::Cursor::new(bytes))
    }

    pub fn to_layer_rows(&self) -> Vec<LayerRow> {
        let mut rows = Vec::new();
        let push_2d = |rows: &mut Vec<LayerRow>, name: &str, m: &[Vec<f32>]| {
            let shape = vec![m.len() as i64, m[0].len() as i64];
            let mut data = Vec::with_capacity(m.len() * m[0].len() * 4);
            for row in m {
                for &v in row {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
            rows.push(LayerRow { name: name.to_string(), shape, data });
        };
        let push_1d = |rows: &mut Vec<LayerRow>, name: &str, v: &[f32]| {
            let shape = vec![v.len() as i64];
            let mut data = Vec::with_capacity(v.len() * 4);
            for &x in v {
                data.extend_from_slice(&x.to_le_bytes());
            }
            rows.push(LayerRow { name: name.to_string(), shape, data });
        };
        push_2d(&mut rows, "embedding", &self.embedding_table);
        let h = self.heads as usize;
        let d = self.wo[0].len();
        let hd = self.head_dim as usize;
        for l in 0..self.layers as usize {
            let mut flat_wq = Vec::with_capacity(h * d * hd * 4);
            let mut flat_wk = Vec::with_capacity(h * d * hd * 4);
            let mut flat_wv = Vec::with_capacity(h * d * hd * 4);
            for head in 0..h {
                for i in 0..d {
                    for j in 0..hd {
                        flat_wq.extend_from_slice(&self.wq[l][head][i][j].to_le_bytes());
                        flat_wk.extend_from_slice(&self.wk[l][head][i][j].to_le_bytes());
                        flat_wv.extend_from_slice(&self.wv[l][head][i][j].to_le_bytes());
                    }
                }
            }
            rows.push(LayerRow {
                name: format!("h.{}.attn.wq", l),
                shape: vec![h as i64, d as i64, hd as i64],
                data: flat_wq,
            });
            rows.push(LayerRow {
                name: format!("h.{}.attn.wk", l),
                shape: vec![h as i64, d as i64, hd as i64],
                data: flat_wk,
            });
            rows.push(LayerRow {
                name: format!("h.{}.attn.wv", l),
                shape: vec![h as i64, d as i64, hd as i64],
                data: flat_wv,
            });
            push_2d(&mut rows, &format!("h.{}.attn.wo", l), &self.wo[l]);
            push_2d(&mut rows, &format!("h.{}.ffn.w1", l), &self.w1[l]);
            push_2d(&mut rows, &format!("h.{}.ffn.w2", l), &self.w2[l]);
            push_1d(&mut rows, &format!("h.{}.ffn.b1", l), &self.b1[l]);
            push_1d(&mut rows, &format!("h.{}.ffn.b2", l), &self.b2[l]);
            push_1d(&mut rows, &format!("h.{}.ln.gamma", l), &self.ln_gamma[l]);
            push_1d(&mut rows, &format!("h.{}.ln.beta", l), &self.ln_beta[l]);
        }
        push_2d(&mut rows, "unembed", &self.w_output);
        rows
    }

    pub fn from_layer_rows(header: &ModelHeader, rows: &[LayerRow]) -> Result<Self, std::io::Error> {
        use std::collections::HashMap;
        let d = header.dimensions as usize;
        let _vocab = header.vocab_size as usize;
        let h = header.heads as usize;
        let hd = d / h;
        let _ff = d * header.ff_multiplier as usize;
        let l = header.layers as usize;

        let map: HashMap<&str, &LayerRow> = rows.iter().map(|r| (r.name.as_str(), r)).collect();

        let get_2d = |map: &HashMap<&str, &LayerRow>, name: &str| -> Result<Vec<Vec<f32>>, std::io::Error> {
            let row = map.get(name).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing layer: {}", name))
            })?;
            let r = row.shape[0] as usize;
            let c = row.shape[1] as usize;
            let flat: Vec<f32> = row.data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            if flat.len() != r * c {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("layer {} data size mismatch", name)));
            }
            Ok(flat.chunks(c).map(|c| c.to_vec()).collect())
        };

        let get_1d = |map: &HashMap<&str, &LayerRow>, name: &str| -> Result<Vec<f32>, std::io::Error> {
            let row = map.get(name).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing layer: {}", name))
            })?;
            Ok(row.data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };

        let _get_3d_heads = |map: &HashMap<&str, &LayerRow>, name_prefix: &str| -> Result<Vec<Vec<Vec<f32>>>, std::io::Error> {
            let mut result = Vec::new();
            for hi in 0..h {
                let key = format!("{}.{}", name_prefix, hi);
                let row = map.get(key.as_str()).unwrap_or_else(|| {
                    // If per-head rows don't exist, the data is stored as a single 3D blob
                    map.get(name_prefix).unwrap()
                });
                // Single 3D blob case: all heads in one row
                if result.is_empty() && h > 1 && row.shape.len() == 3 {
                    let data: Vec<f32> = row.data.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let heads = row.shape[0] as usize;
                    let rows_d = row.shape[1] as usize;
                    let cols = row.shape[2] as usize;
                    for hi in 0..heads {
                        let mut mat = vec![vec![0.0f32; cols]; rows_d];
                        let off = hi * rows_d * cols;
                        for i in 0..rows_d {
                            for j in 0..cols {
                                mat[i][j] = data[off + i * cols + j];
                            }
                        }
                        result.push(mat);
                    }
                    return Ok(result);
                }
            }
            // Single head or per-head rows
            for hi in 0..h {
                let key = format!("{}.{}", name_prefix, hi);
                if let Ok(mat) = get_2d(map, &key) {
                    result.push(mat);
                }
            }
            if result.is_empty() {
                // Fallback: try the 3D blob approach
                if let Some(row) = map.get(name_prefix) {
                    if row.shape.len() == 3 {
                        let data: Vec<f32> = row.data.chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        let heads = row.shape[0] as usize;
                        let rows_d = row.shape[1] as usize;
                        let cols = row.shape[2] as usize;
                        for hi in 0..heads {
                            let mut mat = vec![vec![0.0f32; cols]; rows_d];
                            let off = hi * rows_d * cols;
                            for i in 0..rows_d {
                                for j in 0..cols {
                                    mat[i][j] = data[off + i * cols + j];
                                }
                            }
                            result.push(mat);
                        }
                    }
                }
            }
            Ok(result)
        };

        let embedding_table = get_2d(&map, "embedding")?;

        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        let mut w1 = Vec::new();
        let mut w2 = Vec::new();
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        let mut ln_gamma = Vec::new();
        let mut ln_beta = Vec::new();

        for li in 0..l {
            // wq: single 3D blob [heads][d][hd] or per-head 2D blobs
            let prefix = format!("h.{}.attn", li);

            // Read QKV as flat 3D blob from the combined name
            let read_qkv = |map: &HashMap<&str, &LayerRow>, name: &str| -> Result<Vec<Vec<Vec<f32>>>, std::io::Error> {
                let row = map.get(name).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing layer: {}", name))
                })?;
                if row.shape.len() != 3 {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("expected 3D shape for {}", name)));
                }
                let heads_d = row.shape[0] as usize;
                let rows_d = row.shape[1] as usize;
                let cols = row.shape[2] as usize;
                let data: Vec<f32> = row.data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let mut result = Vec::new();
                for hi in 0..heads_d {
                    let mut mat = vec![vec![0.0f32; cols]; rows_d];
                    let off = hi * rows_d * cols;
                    for i in 0..rows_d {
                        for j in 0..cols {
                            mat[i][j] = data[off + i * cols + j];
                        }
                    }
                    result.push(mat);
                }
                Ok(result)
            };

            wq.push(read_qkv(&map, &format!("{}.wq", prefix))?);
            wk.push(read_qkv(&map, &format!("{}.wk", prefix))?);
            wv.push(read_qkv(&map, &format!("{}.wv", prefix))?);
            wo.push(get_2d(&map, &format!("{}.wo", prefix))?);
            w1.push(get_2d(&map, &format!("h.{}.ffn.w1", li))?);
            w2.push(get_2d(&map, &format!("h.{}.ffn.w2", li))?);
            b1.push(get_1d(&map, &format!("h.{}.ffn.b1", li))?);
            b2.push(get_1d(&map, &format!("h.{}.ffn.b2", li))?);
            ln_gamma.push(get_1d(&map, &format!("h.{}.ln.gamma", li))?);
            ln_beta.push(get_1d(&map, &format!("h.{}.ln.beta", li))?);
        }

        let w_output = get_2d(&map, "unembed")?;

        Ok(Model {
            embedding_table,
            max_seq_len: header.max_seq_len,
            wq, wk, wv, wo,
            heads: header.heads,
            head_dim: hd as i64,
            layers: header.layers,
            w1, w2, b1, b2, ln_gamma, ln_beta,
            // An imported checkpoint carries no final norm in this layout, and
            // guessing one would change what it predicts. Left empty, which is
            // the same "no final norm" the pre-v5 native path reports.
            ln_final_gamma: Vec::new(),
            ln_final_beta: Vec::new(),
            w_output,
        })
    }
}
