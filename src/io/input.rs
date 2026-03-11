use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub fn prepare_input_data() -> io::Result<()> {
    let input_dir = "inputs";
    let output_file = "data/input.txt";

    println!("Combining text files from {}/ into {}...", input_dir, output_file);

    let mut combined_content = String::new();
    let paths = fs::read_dir(input_dir)?;

    for path in paths {
        let path = path?.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            // Skip the output file if it happens to be in the same dir (not the case here)
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                if filename == "input.txt" {
                    // Decide if we skip the original input.txt or not. 
                    // Usually, if we're combining, we include everything.
                }
            }
            
            println!("Reading: {:?}", path);
            let content = fs::read_to_string(&path)?;
            combined_content.push_str(&content);
            combined_content.push('\n'); // Add a newline between files
        }
    }

    if !Path::new("data").exists() {
        fs::create_dir_all("data")?;
    }

    let mut file = fs::File::create(output_file)?;
    file.write_all(combined_content.as_bytes())?;

    println!("Successfully combined files into {}.", output_file);
    Ok(())
}
