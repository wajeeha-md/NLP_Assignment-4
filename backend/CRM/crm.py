import sqlite3
import json
import asyncio
from pathlib import Path
import numpy as np
from RAG.retrieval import get_collection, get_embedding

# Setup database path
CRM_DIR = Path(__file__).parent
CRM_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CRM_DIR / "crm.db"

def _init_db():
    """Initialize the SQLite database with the users table."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                data TEXT
            )
        ''')
        conn.commit()

# Ensure the database exists on import
_init_db()

SIMILARITY_THRESHOLD = 0.45

async def _find_semantic_field(user_id: str, field: str) -> str:
    """Find the semantically closest field for a user in the user_memory collection."""
    try:
        collection = await get_collection("user_memory")
        query_emb = await get_embedding(field)
        
        loop = asyncio.get_running_loop()
        def _search():
            return collection.query(
                query_embeddings=[query_emb],
                where={"user_id": user_id},
                n_results=1
            )
            
        results = await loop.run_in_executor(None, _search)
        
        if results and results['distances'] and results['distances'][0]:
            distance = results['distances'][0][0]
            similarity = 1.0 - distance
            found_field = results['metadatas'][0][0]['field']
            print(f"[DEBUG] Semantic match: '{field}' vs '{found_field}' | similarity: {similarity:.4f}")
            
            if similarity >= SIMILARITY_THRESHOLD:
                return found_field
                
    except Exception as e:
        print(f"Semantic search error: {e}")
        
    return field # Fallback to exact match

async def _sync_memory_entry(user_id: str, field: str):
    """Ensure a field name is indexed in the user_memory collection."""
    try:
        collection = await get_collection("user_memory")
        emb = await get_embedding(field)
        
        loop = asyncio.get_running_loop()
        def _upsert():
            # ID is unique per user+field
            mem_id = f"{user_id}_{field}"
            collection.upsert(
                ids=[mem_id],
                embeddings=[emb],
                metadatas=[{"user_id": user_id, "field": field}],
                documents=[field]
            )
            
        await loop.run_in_executor(None, _upsert)
    except Exception as e:
        print(f"Memory sync error: {e}")

def _create_user_sync(user_id: str, data: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO users (user_id, data) VALUES (?, ?)',
            (user_id, json.dumps(data))
        )
        conn.commit()
    return True

def _get_user_info_sync(user_id: str) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return {}

def _update_user_info_sync(user_id: str, field: str, value: any):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            data = json.loads(row[0])
        else:
            data = {}
            
        data[field] = value
        
        cursor.execute(
            'INSERT OR REPLACE INTO users (user_id, data) VALUES (?, ?)',
            (user_id, json.dumps(data))
        )
        conn.commit()
    return data

async def create_user(user_id: str, data: dict):
    """Asynchronously create or overwrite a user and sync their keys to memory."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _create_user_sync, user_id, data)
    # Index all keys
    for field in data.keys():
        await _sync_memory_entry(user_id, field)
    return True

async def get_user_info(user_id: str) -> dict:
    """Asynchronously retrieve the user data dictionary."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _get_user_info_sync, user_id)

async def update_user_info(user_id: str, field: str, value: any) -> dict:
    """Asynchronously update a field using semantic matching for keys."""
    # Find the real key name semantically (typo tolerance)
    canonical_field = await _find_semantic_field(user_id, field)
    
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _update_user_info_sync, user_id, canonical_field, value)
    
    # Ensure the key is indexed
    await _sync_memory_entry(user_id, canonical_field)
    return data

if __name__ == "__main__":
    async def test_crm():
        print("--- Testing Semantic CRM Module ---")
        user_id = "test_user_sem"
        
        print(f"1. Creating user '{user_id}' with 'marla' field...")
        await create_user(user_id, {"marla": 5})
        
        print("2. Updating 'mlra' (typo) to 10...")
        await update_user_info(user_id, "mlra", 10)
        
        print("3. Fetching user info...")
        info = await get_user_info(user_id)
        print(f"   -> Info (expecting 'marla': 10): {info}")
        
        print("4. Adding new field 'budget'...")
        await update_user_info(user_id, "budget", "50 Lac")
        
        print("5. Updating 'spending limit' (paraphrase) to '1 Crore'...")
        await update_user_info(user_id, "spending limit", "1 Crore")
        
        final_info = await get_user_info(user_id)
        print(f"   -> Final Info (expecting 'budget': '1 Crore'): {final_info}")
        
        print("--- CRM Test Complete ---")

    asyncio.run(test_crm())
