# **Semantic Codebase Search**

A **semantic code search tool** for Rust projects that enables users to find relevant code elements using **natural language queries**. This tool leverages **sentence embeddings** to generate vector representations of code elements and **performs semantic search** to retrieve the most relevant code snippets.

## **Features**
- 🔍 **Semantic Search**: Find code elements using natural language queries.
- 📌 **Code Indexing**: Automatically indexes the codebase to generate embeddings for code elements.
- 📈 **Similarity Calculation**: Uses **cosine similarity** to rank search results based on relevance.

## **Project Structure**
- **CodeElement**: Represents a code element (e.g., function, struct, module) with attributes like name, type, content, and path.
- **CodeVisitor**: Extracts and processes code elements to gather documentation and metadata.
- **SemanticSearch**: Manages the indexing and searching of code elements using sentence embeddings.

## **Getting Started**
### **Prerequisites**
Ensure you have the following installed:
- [Rust (latest stable version)](https://www.rust-lang.org/)
- [Cargo (Rust package manager)](https://doc.rust-lang.org/cargo/)

### **Installation**
Clone the repository:
```sh
git clone https://github.com/your-username/semantic-codebase-search.git
cd semantic-codebase-search
```
Build the project:
```sh
cargo build --release
```

### **Usage**
Run the project:
```sh
cargo run
```
Follow the on-screen instructions to enter your search query. The tool will **index** the current directory and perform a **semantic search** based on your input.

## **Example**
```plaintext
Enter your search query: "Find function to read a file"
Results:
1. read_file(path: &str) -> String  (src/utils.rs)
2. load_config(file: &str) -> Config  (src/config.rs)
```

## **Key Functions**
- **`index_code`**: Generates embeddings for code snippets.
- **`cosine_similarity`**: Computes similarity between query and indexed code embeddings.
- **`parse_code`**: Parses source files and constructs a syntax tree.
- **`process_directory`**: Iterates through all files, extracting and indexing code elements.
- **`search`**: Executes the semantic search and ranks results.

## **Contributing**
Contributions are welcome! Feel free to:
- Open an [issue](https://github.com/your-username/semantic-codebase-search/issues) for bug reports or feature requests.
- Submit a pull request with improvements.

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

