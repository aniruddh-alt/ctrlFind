use std::env;
use std::fs;
use std::error::Error;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use ndarray::Array1;
use std::path::Path;
use syn::{parse_file, File};
use syn::visit::Visit;
use rustpython_parser::{parser, ast};
use syn::{ItemFn, ItemStruct, ItemEnum, Ident, Attribute, Meta, MetaList};
use quote::ToTokens;
use syn::visit::{self};
use std::collections::HashMap;

use trial::{SemanticSearch,cosine_similarity};


fn main() {
    use std::io::{stdin,stdout,Write};
    
    let dir_path = env::current_dir().unwrap();
    println!("The current directory is {}", dir_path.display());
    
    let mut query=String::new();
    print!("Please enter some text: ");
    let _=stdout().flush();
    stdin().read_line(&mut query).expect("Did not enter a correct string");
    if let Some('\n')=query.chars().next_back() {
        query.pop();
    }
    if let Some('\r')=query.chars().next_back() {
        query.pop();
    }
    
    let mut search = SemanticSearch::new().unwrap();
    search.index_directory(&dir_path);
    let results = search.search(&query, 5).unwrap();
    
    for (element, score) in results{
        println!("{}","=".repeat(20));
        println!("\nScore: {:.4}\n Name: {}\n Content: {}\n Code Type{}\n", score, element.name, element.content, element.code_type);
    }
    
}

