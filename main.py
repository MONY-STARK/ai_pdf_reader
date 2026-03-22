import os
import argparse
import chromadb.utils.embedding_functions as embedding_functions

from chromadb import EmbeddingFunction, Documents, Embeddings
from preprocesser import TextPreprocessor
from chunker import Chunker
from dataloader import DataLoader
from vectordatabase import VectorDatabase
from highlighter import Highlighter
from llama_cpp import Llama

# ── Qwen3 Local Embedding Model ───────────────────────────────────────────────
_llm_model = Llama(
    model_path="/home/stark/Embedding_model_Qwen/Qwen3-Embedding-4B-Q5_K_M.gguf",
    embedding=True,
    n_ctx=512,
    verbose=False
)

class QwenEmbedding(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [_llm_model.embed(text) for text in input]

def build_pipeline():
    """Instantiate and wire all components."""
    preprocessor = TextPreprocessor()
    chunker      = Chunker(preprocessor)
    loader       = DataLoader()
    highlighter  = Highlighter()
  
    vector_db = VectorDatabase(QwenEmbedding())

    return loader, preprocessor, chunker, vector_db, highlighter


def ingest(pdf_path: str, collection_name: str, loader, preprocessor, chunker, vector_db, highlighter):
    """Load → preprocess → chunk → embed."""
    print(f"[1/3] Loading PDF: {pdf_path}")
    pages_data = loader.load_data(pdf_path)

    print("[2/3] Preprocessing text...")
    pages_data = preprocessor.preprocess(pages_data)

    # Share pages_data with highlighter so it can resolve word coords later
    highlighter.set_pages_data(pages_data)

    print("[3/3] Chunking and embedding...")
    chunks = chunker.chunk(pages_data)

    if not chunks:
        raise ValueError("No chunks were produced. Check if the PDF has extractable text.")

    vector_db.create_collection(collection_name)

    # Skip first chunk only if it looks like a header (very short)
    insert_chunks = chunks[1:] if len(chunks[0]["text"].split()) < 10 else chunks
    vector_db.insert(insert_chunks)

    print(f"Ingested {len(insert_chunks)} chunks from {len(pages_data)} pages.\n")
    return pages_data


def query_and_highlight(query: str, pdf_path: str, output_path: str,
                        vector_db, highlighter, n_results=3):
    """Query vector DB → highlight matching chunks in PDF."""
    print(f"Query : {query!r}")
    results = vector_db.query(query, n_results=n_results)

    matched = results.get("metadatas", [[]])[0]
    if not matched:
        print("No relevant chunks found for your query.")
        return

    print(f"Found {len(matched)} matching chunk(s).")
    highlighter.highlight(pdf_path, results, output_path)
    print(f"Highlighted PDF saved to: {output_path}\n")


def interactive_mode(pdf_path: str, output_dir: str,
                     loader, preprocessor, chunker, vector_db, highlighter):
    """REPL — ingest once, query multiple times."""
    collection_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ingest(pdf_path, collection_name, loader, preprocessor, chunker, vector_db, highlighter)

    print("Ready! Type your question (or 'exit' to quit).\n")
    idx = 1
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if not query:
            continue

        output_path = os.path.join(output_dir, f"result_{idx}.pdf")
        query_and_highlight(query, pdf_path, output_path, vector_db, highlighter)
        idx += 1


def main():
    parser = argparse.ArgumentParser(description="Smart PDF Highlighter")
    parser.add_argument("pdf",           help="Path to the input PDF")
    parser.add_argument("--query", "-q", help="Question to answer (omit for interactive mode)")
    parser.add_argument("--output", "-o", default="output.pdf",
                        help="Output PDF path (used only with --query)")
    parser.add_argument("--n_results", "-n", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"Error: file not found — {args.pdf}")
        return

    loader, preprocessor, chunker, vector_db, highlighter = build_pipeline()

    if args.query:
        # Single-shot mode
        collection_name = os.path.splitext(os.path.basename(args.pdf))[0]
        ingest(args.pdf, collection_name, loader, preprocessor, chunker, vector_db, highlighter)
        query_and_highlight(args.query, args.pdf, args.output,
                            vector_db, highlighter, n_results=args.n_results)
    else:
        # Interactive REPL mode
        output_dir = os.path.dirname(args.output) or "."
        os.makedirs(output_dir, exist_ok=True)
        interactive_mode(args.pdf, output_dir,
                         loader, preprocessor, chunker, vector_db, highlighter)


if __name__ == "__main__":
    main()