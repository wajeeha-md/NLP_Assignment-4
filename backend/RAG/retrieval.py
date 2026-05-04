import os
from typing import Optional, List
import difflib
import asyncio
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cache the model, DB client, and embeddings
_model = None
_client = None
_embedding_cache = {}
_init_lock = asyncio.Lock()

async def get_model():
    """Shared singleton for the embedding model."""
    global _model
    if _model is None:
        loop = asyncio.get_running_loop()
        _model = await loop.run_in_executor(None, SentenceTransformer, MODEL_NAME)
    return _model

async def get_client():
    """Shared singleton for the ChromaDB client."""
    global _client
    if _client is None:
        if not INDEX_DIR.exists():
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        _client = await loop.run_in_executor(None, chromadb.PersistentClient, str(INDEX_DIR))
    return _client

async def get_collection(name: str):
    """Fetch a specific collection by name with cosine similarity."""
    client = await get_client()
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, 
            lambda: client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        )
    except Exception as e:
        raise RuntimeError(f"Could not load collection '{name}'. Error: {e}")

async def get_embedding(text: str):
    """Retrieve an embedding with local LRU-style caching."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    
    model = await get_model()
    loop = asyncio.get_running_loop()
    
    def _encode():
        return model.encode([text], show_progress_bar=False)[0].tolist()
        
    emb = await loop.run_in_executor(None, _encode)
    _embedding_cache[text] = emb
    return emb

async def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieve top-k relevant text chunks for a given query asynchronously."""
    try:
        async with _init_lock:
            collection = await get_collection("real_estate_docs")
        
        query_emb = await get_embedding(query)
        
        loop = asyncio.get_running_loop()
        # Query ChromaDB collection in a thread
        def _query():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=k
            )
            
        results = await loop.run_in_executor(None, _query)
        
        chunks = []
        if results.get('documents') and results['documents'][0]:
            for doc in results['documents'][0]:
                chunks.append(doc)
                
        return chunks
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []


async def semantic_match(query: str, options: List[str], threshold: float = 0.5) -> Optional[str]:
    """Find the best match for a query among a list of options.
    Uses a hybrid of Substring Matching, Fuzzy Matching, and Embeddings.
    """
    if not options:
        return None
    
    lower_query = query.lower()
    
    # 1. Exact Substring Matching (Fastest)
    for opt in options:
        if opt.lower() in lower_query:
            return opt

    # 2. Word-level Fuzzy Matching (Excellent for typos like 'mlra' -> 'marla')
    words = lower_query.split()
    for opt in options:
        # Check if any word in the query is close to the option
        if difflib.get_close_matches(opt.lower(), words, n=1, cutoff=0.5):
            return opt

    # 3. Embedding Matching (For semantic synonyms like 'spending limit' -> 'budget')
    query_emb = await get_embedding(query)
    best_option = None
    max_sim = -1.0
    
    for opt in options:
        opt_emb = await get_embedding(opt)
        sim = sum(a * b for a, b in zip(query_emb, opt_emb))
        if sim > max_sim:
            max_sim = sim
            best_option = opt
            
    if max_sim >= threshold:
        return best_option
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test RAG retrieval manually.")
    parser.add_argument("--query", type=str, default="buy a house in lahore", help="Query to run")
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to return")
    args = parser.parse_args()
    
    async def main():
        print(f"--- Retrieving top {args.k} chunks for: '{args.query}' ---")
        retrieved_docs = await retrieve(args.query, k=args.k)
        
        if retrieved_docs:
            for idx, doc in enumerate(retrieved_docs, 1):
                snippet = doc[:250].replace('\n', ' ')
                print(f"\n[Result {idx}]:\n{snippet}...")
        else:
            print("No documents found.")
            
    asyncio.run(main())
