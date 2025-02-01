use ndarray::Array1;
use syn::{parse_file, File};
use std::error::Error;
use std::fs;
use std::path::Path;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use syn::visit::Visit;
use rustpython_parser::{parser, ast};
use syn::{ItemFn, ItemStruct, ItemEnum, Ident, Attribute, Meta, MetaList};
use quote::ToTokens;
use syn::visit::{self};

#[derive(Debug)]
pub struct CodeElement{
    pub name: String,
    pub code_type: String,
    pub content: String, 
    pub path: String,
    pub language: String,
    pub docs: String,
    pub attributes: Vec<String>,
    pub parameters: Option<Vec<(String, String)>>,
    pub return_type: Option<String>,
    pub context: Vec<String>,
    pub embedding: Option<Array1<f32>>,
}

struct CodeVisitor{
    elements: Vec<CodeElement>,
    current_file: String,
    context_stack: Vec<String>,
}

impl CodeVisitor {
    fn new(file_path: String) -> Self {
        CodeVisitor {
            elements: Vec::new(),
            current_file: file_path,
            context_stack: Vec::new()
        }
    }

    fn extract_docs(attrs: &[Attribute]) -> String {
        attrs.iter()
            .filter_map(|attr| {
                if attr.path().is_ident("doc") {
                    attr.meta.require_name_value().ok().and_then(|meta| {
                        meta.value.clone().into_token_stream().to_string()
                            .trim_matches('"')
                            .to_string()
                            .into()
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    /// Extract all attributes as strings
    fn extract_attributes(attrs: &[Attribute]) -> Vec<String> {
        attrs.iter()
            .map(|attr| attr.to_token_stream().to_string())
            .collect()
    }

    fn process_function_sig(sig: &syn::Signature) -> (Vec<(String, String)>,Option<String>){
        let params = sig.inputs.iter().filter_map(|params|{
            if let syn::FnArg::Typed(pat_type) = params{
                Some((
                    pat_type.pat.to_token_stream().to_string(),
                    pat_type.ty.to_token_stream().to_string(),
                ))
            } else {
                None
            }
        }).collect();

        let return_type = match &sig.output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => Some(ty.to_token_stream().to_string()),
        };
        (params, return_type)
    }
}

impl<'ast> Visit<'ast> for CodeVisitor {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        // Extract function details
        let fn_name = node.sig.ident.to_string();
        let (params, return_type) = Self::process_function_sig(&node.sig);

        let body = node.block.to_token_stream().to_string();

        let fn_content = format!(
            "fn {}({}) {} {{\n{}\n}}",
            fn_name,
            params.iter().map(|(name, typ)| format!("{} {}", name, typ)).collect::<Vec<String>>().join(", "),
            return_type.as_deref().unwrap_or(""),
            body
        );

        self.elements.push(CodeElement {
            name: fn_name,
            code_type: "function".to_string(),
            content: fn_content,
            path: self.current_file.clone(),
            embedding: None,
            language: "rust".to_string(),
            docs: Self::extract_docs(&node.attrs),
            attributes: Self::extract_attributes(&node.attrs),
            parameters: Some(params),
            return_type: return_type,
            context: self.context_stack.clone(),
        });

        visit::visit_item_fn(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        // Extract struct details
        let struct_name = node.ident.to_string();
        let struct_content = format!("struct {} {{...}}", struct_name);

        self.elements.push(CodeElement {
            name: struct_name,
            code_type: "struct".to_string(),
            content: struct_content,
            path: self.current_file.clone(),
            embedding: None,
            language: "rust".to_string(),
            docs: Self::extract_docs(&node.attrs),
            attributes: Self::extract_attributes(&node.attrs),
            parameters: None,
            return_type: None,
            context: self.context_stack.clone(),
        });
        visit::visit_item_struct(self, node)
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let mod_name = node.ident.to_string();
        self.context_stack.push(format!("mod {}", mod_name));
        visit::visit_item_mod(self, node);
        self.context_stack.pop();
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let type_name = node.self_ty.to_token_stream().to_string();
        self.context_stack.push(format!("impl {}", type_name));
    
        // Visit all items in the impl block
        for item in &node.items {
            self.visit_impl_item(item); // Handle all impl item types
        }
    
        visit::visit_item_impl(self, node);
        self.context_stack.pop();
    }
    
    fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
        match node {
            syn::ImplItem::Fn(method) => {
                // Handle methods
                let method_name = method.sig.ident.to_string();
                let (params, return_type) = Self::process_function_sig(&method.sig);
                let method_content = format!(
                    "fn {}({}) {} {{\n{}\n}}",
                    method_name,
                    params.iter().map(|(name, typ)| format!("{} {}", name, typ)).collect::<Vec<String>>().join(", "),
                    return_type.as_deref().unwrap_or(""),
                    method.block.to_token_stream().to_string()
                );

                
                self.elements.push(CodeElement {
                    name: method_name,
                    code_type: "method".into(),
                    content: method_content.to_string(),
                    language: "rust".into(),
                    context: self.context_stack.clone(),
                    parameters: Some(params),
                    return_type: return_type,
                    docs: Self::extract_docs(&method.attrs),
                    attributes: Self::extract_attributes(&method.attrs),
                    path: self.current_file.clone(),
                    embedding: None,
                });
                
                visit::visit_impl_item_fn(self, method);
            }
            syn::ImplItem::Const(cnst) => {
                // Handle constants
                visit::visit_impl_item_const(self, cnst);
            }
            syn::ImplItem::Type(typ) => {
                // Handle associated types
                visit::visit_impl_item_type(self, typ);
            }
            _ => {}
        }
    }
}

pub struct SemanticSearch {
    model: rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel,
    code_elements: Vec<CodeElement>,
}

impl SemanticSearch{
    pub fn new() -> Result<Self, Box<dyn Error>>{
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .create_model()?;
        Ok(SemanticSearch {
            model,
            code_elements: Vec::new(),
        })
    }

    fn generate_embeddings(&mut self, content: &str) -> Result<Array1<f32>, Box<dyn Error>> {
        let embeddings = self.model.encode(&[content])?;
        Ok(Array1::from(embeddings.into_iter().next().unwrap()))
    }

    pub fn index_directory(&mut self, dir: &Path){
        let results = process_directory(&dir).unwrap();

        for (path, file) in results{
            let mut visitor = CodeVisitor::new(path);
            visitor.visit_file(&file);

            for mut element in visitor.elements {
                // Create rich context for embedding
                let context = format!(
                    "Name: {}\nType: {}\nContext: {}\nDocs: {}\nParameters: {}\nReturn Type: {}\nAttributes: {}\nContent: {}",
                    element.name,
                    element.code_type,
                    element.context.join(" -> "),
                    element.docs,
                    element.parameters.as_ref().map_or(String::new(), |params| {
                        params.iter()
                            .map(|(name, ty)| format!("{}: {}", name, ty))
                            .collect::<Vec<_>>()
                            .join(", ")
                    }),
                    element.return_type.as_deref().unwrap_or(""),
                    element.attributes.join("\n"),
                    element.content
                );
                println!("{:?}",&context);
                
                let embedding = self.generate_embeddings(&context).unwrap();
                element.embedding = Some(embedding);
                self.code_elements.push(element);
            }           
        }
    }

    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<(&CodeElement, f32)>, Box<dyn Error>> {
        let query_embed = self.generate_embeddings(query)?;
       let mut results: Vec<(&CodeElement, f32)> = self.code_elements
        .iter()
        .filter_map(|element| {
            element.embedding.as_ref().map(|emb| {
                let similarity = cosine_similarity(&emb, &query_embed);
                (element, similarity)
            })
        }).collect();
        // Sort by similarity score
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        Ok(results)
    }
}
pub fn index_code(code: &str) -> Result<Array1<f32>, Box<dyn Error>> {
    let model = SentenceEmbeddingsBuilder::remote
    (SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;
    let embeddings = model.encode(&[code])?;
    Ok(Array1::from(embeddings.into_iter().next().unwrap()))
}

pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_prod = a.dot(b);
    let magnitude = (a.dot(a).sqrt() * b.dot(b).sqrt()).max(1e-6);
    dot_prod / magnitude
}

fn parse_code(path: &str) -> Result<File, Box<dyn std::error::Error>>{
    let source_code = std::fs::read_to_string(&path).expect("failed to read the file");
    let syntax_tree = parse_file(&source_code).expect("failed to parse the code");
    Ok(syntax_tree)
}

fn process_directory(dir: &Path) -> Result<Vec<(String, File)>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            if path.file_name() != Some(std::ffi::OsStr::new("target")) {
                results.extend(process_directory(&path)?);
            }
        } else if path.extension().map_or(false, |ext| ext == "rs") {
            let path = path.to_str().unwrap().to_string();
            let analyzer = parse_code(&path)?;
            results.push((path, analyzer));
        } 
    }
    Ok(results)
}
