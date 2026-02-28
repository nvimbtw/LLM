use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub fn read_text_file(file_path: &str) -> String {
    let contents = match fs::read_to_string(file_path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("Error reading file: {}", error);
            panic!("Failed to read file");
        }
    };
    contents
}

pub fn write_vocab(pairs: &Vec<String>, path: &str) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for pair_str in pairs {
        let parts: Vec<&str> = pair_str.split(",").collect();
        let num1: u32 = parts[0].parse().unwrap();
        let num2: u32 = parts[1].parse().unwrap();
        file.write_all(&num1.to_le_bytes())?;
        file.write_all(&num2.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

pub fn read_vocab(bpe_tokens_path: &str) -> std::io::Result<Vec<(u32, u32)>> {
    let mut file = BufReader::new(File::open(bpe_tokens_path)?);
    let mut pairs = Vec::new();

    loop {
        let mut num1_bytes = [0u8; 4];
        let mut num2_bytes = [0u8; 4];

        match file.read_exact(&mut num1_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        file.read_exact(&mut num2_bytes)?;

        let num1 = u32::from_le_bytes(num1_bytes);
        let num2 = u32::from_le_bytes(num2_bytes);

        pairs.push((num1, num2));
    }

    Ok(pairs)
}

pub fn write_tokens(tokens: &Vec<u32>, path: &str) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    for token in tokens {
        file.write_all(&token.to_le_bytes())?;
    }

    file.flush()?;
    Ok(())
}

pub fn read_tokens(path: &str) -> std::io::Result<Vec<u32>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut tokens = Vec::new();

    loop {
        let mut bytes = [0u8; 4];
        match file.read_exact(&mut bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        tokens.push(u32::from_le_bytes(bytes));
    }

    Ok(tokens)
}
#[allow(dead_code)]
pub fn save_matrix(matrix: &Vec<Vec<f32>>, path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    let rows = matrix.len() as u32;
    let cols = matrix[0].len() as u32;
    file.write_all(&rows.to_le_bytes())?;
    file.write_all(&cols.to_le_bytes())?;
    for row in matrix {
        for val in row {
            file.write_all(&val.to_le_bytes())?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn load_matrix(path: &str) -> std::io::Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let rows = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut matrix = vec![vec![0.0f32; cols]; rows];
    let mut offset = 8;
    for i in 0..rows {
        for j in 0..cols {
            let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            matrix[i][j] = val;
            offset += 4;
        }
    }
    Ok(matrix)
}
