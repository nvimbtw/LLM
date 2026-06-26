# New model pipeline
This document is the pipeline written out, to ease the dev process 

## Pre-tokenization
Pre tokenization ensures that we append Ġ before each word, so the model can differentiate between what should be a new word and what is not a new word - we give it a new "white space" that it can orient on

### Pre-tokenization process
#### [DONE] input.txt
Get the input.txt file from frontend

#### [DONE] Raw text string
Load the input file from /api/data/input/

#### [DONE] Normalize whitespace → preserve structure tokens
Replace \n with Ċ and \t with ĉ before splitting, so structure
isn't lost. Double newlines (Ċ Ċ) will naturally merge in BPE
as paragraph boundaries.

#### [DONE] Split on whitespace → raw words
Split on whitespace to get an array of words

#### [DONE] Split punctuation off each word → clean tokens
Split the punctuation from words

#### [DONE] Prepend Ġ to each token → space-aware tokens
Prepend the Ġ to each word (excluding punctuation and structure
tokens like Ċ) so we get space aware tokens

#### [TODO] Convert each token to char/byte sequence → BPE input units
"ĠHello" → ['Ġ', 'H', 'e', 'l', 'l', 'o']
"Ġworld" → ['Ġ', 'w', 'o', 'r', 'l', 'd']
"Ċ"      → ['Ċ']

#### [TODO] Save to pre_tokenization.bin
Save the file to /api/data/pre_tokenization.bin

## Byte pair encoding
