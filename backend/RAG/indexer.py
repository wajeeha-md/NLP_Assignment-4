import os
import glob
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer
import asyncio

# Configuration
DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Token chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return text

def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

async def load_documents(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    
    # Process PDFs and TXTs
    loop = asyncio.get_running_loop()
    
    # Gather all file paths
    filepaths = []
    for ext in ["*.pdf", "*.txt"]:
        for file in glob.glob(str(data_dir / ext)):
            filepaths.append(Path(file))
            
    async def process_file(path):
        print(f"Loading {path.name}...")
        if path.suffix.lower() == ".pdf":
            # PDF extraction is synchronous and potentially CPU-bound, run in thread
            text = await loop.run_in_executor(None, extract_text_from_pdf, path)
        else:
            # File I/O run in thread
            text = await loop.run_in_executor(None, extract_text_from_txt, path)
            
        if text.strip():
            return {"id": path.stem, "text": text, "source": path.name}
        return None

    tasks = [process_file(p) for p in filepaths]
    results = await asyncio.gather(*tasks)
    
    for r in results:
        if r:
            documents.append(r)
            
    return documents

def chunk_text(text, tokenizer, chunk_size, chunk_overlap):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    if len(tokens) == 0:
        return chunks
        
    step = chunk_size - chunk_overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
    return chunks

async def process_document(doc, tokenizer, model, loop):
    print(f"Chunking {doc['source']}...")
    chunks = chunk_text(doc["text"], tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
    
    if not chunks:
        return []
        
    print(f" -> Generated {len(chunks)} chunks for {doc['source']}. Computing embeddings...")
    # SentenceTransformer encoding is CPU-heavy, run in thread
    def _encode():
        return model.encode(chunks, show_progress_bar=False)
    embeddings = await loop.run_in_executor(None, _encode)
    
    doc_results = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc['id']}_chunk_{i}"
        doc_results.append({
            "chunk": chunk,
            "metadata": {"source": doc["source"], "chunk_index": i},
            "id": chunk_id,
            "embedding": emb.tolist()
        })
    return doc_results

async def build_index():
    print("Initializing embedding model and tokenizer...")
    loop = asyncio.get_running_loop()
    
    # Load model in thread to avoid blocking loop
    model = await loop.run_in_executor(None, SentenceTransformer, MODEL_NAME)
    tokenizer = await loop.run_in_executor(None, AutoTokenizer.from_pretrained, MODEL_NAME)
    
    print(f"Loading documents from {DATA_DIR}...")
    docs = await load_documents(DATA_DIR)
    
    if not docs:
        print(f"No documents found in {DATA_DIR}. Please add PDFs or TXT files and run again.")
        return

    print(f"Initializing ChromaDB at {INDEX_DIR}...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # ChromaDB persistent client is synchronous
    chroma_client = await loop.run_in_executor(None, chromadb.PersistentClient, str(INDEX_DIR))
    
    # Recreate collection to run from scratch each time
    try:
        await loop.run_in_executor(None, chroma_client.delete_collection, "real_estate_docs")
    except Exception:
        pass
        
    def _create_collection():
        return chroma_client.create_collection(
            name="real_estate_docs", 
            metadata={"hnsw:space": "cosine"}
        )
        
    collection = await loop.run_in_executor(None, _create_collection)
    
    add_docs = []
    add_metadatas = []
    add_ids = []
    add_embeddings = []
    
    # Process all documents concurrently
    tasks = [process_document(doc, tokenizer, model, loop) for doc in docs]
    results = await asyncio.gather(*tasks)
    
    for doc_results in results:
        for res in doc_results:
            add_docs.append(res["chunk"])
            add_metadatas.append(res["metadata"])
            add_ids.append(res["id"])
            add_embeddings.append(res["embedding"])
            
    if add_docs:
        print(f"Inserting {len(add_docs)} chunks into ChromaDB...")
        batch_size = 5000
        for i in range(0, len(add_docs), batch_size):
            await loop.run_in_executor(
                None,
                lambda: collection.add(
                    documents=add_docs[i:i+batch_size],
                    embeddings=add_embeddings[i:i+batch_size],
                    metadatas=add_metadatas[i:i+batch_size],
                    ids=add_ids[i:i+batch_size]
                )
            )
        print("Indexing pipeline complete!")
    else:
        print("No valid chunks generated.")

def test_retrieval(query="house for sale"):
    print(f"\n--- Testing Retrieval for query: '{query}' ---")
    try:
        model = SentenceTransformer(MODEL_NAME)
        query_emb = model.encode([query])[0].tolist()
        
        chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
        collection = chroma_client.get_collection(name="real_estate_docs")
            
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i]
                print(f"\nResult {i+1} [Distance: {dist:.4f}] from {meta['source']}:")
                snippet = doc[:200].replace('\n', ' ')
                print(f"{snippet}...")
        else:
            print("No matching results found in index.")
    except Exception as e:
        print(f"Test retrieval failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline RAG Indexer")
    parser.add_argument("--test", action="store_true", help="Run a test query instead of indexing")
    parser.add_argument("--query", type=str, default="buy a house", help="Test query to run")
    args = parser.parse_args()
    
    if args.test:
        test_retrieval(args.query)
    else:
        asyncio.run(build_index())
