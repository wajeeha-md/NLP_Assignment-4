import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict

# Setup database path
CALENDAR_DIR = Path(__file__).parent
CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CALENDAR_DIR / "calendar.db"

def _init_db():
    """Initialize the SQLite database with the events table."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                description TEXT NOT NULL
            )
        ''')
        # Create an index on date for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON events(date)')
        conn.commit()

# Ensure the database exists on import
_init_db()

def _add_event_sync(date: str, description: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO events (date, description) VALUES (?, ?)',
            (date, description)
        )
        conn.commit()
    return True

def _get_events_sync(date: str = None) -> List[Dict[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if date:
            cursor.execute('SELECT id, date, description FROM events WHERE date = ? ORDER BY id ASC', (date,))
        else:
            # If no date is provided, return all events
            cursor.execute('SELECT id, date, description FROM events ORDER BY date ASC, id ASC')
            
        rows = cursor.fetchall()
        return [{"id": r[0], "date": r[1], "description": r[2]} for r in rows]

async def add_event(date: str, description: str = None, **kwargs) -> str:
    """
    Asynchronously add an event to the calendar.
    
    Args:
        date: The date of the event (e.g., '2026-05-02')
        description: Primary description of the event
        **kwargs: Additional details (notes, title, location, etc.)
    """
    # If description is missing, try to build it from common hallucinated fields
    if not description:
        parts = []
        for key in ['title', 'event_title', 'notes', 'location', 'property', 'event_type']:
            if key in kwargs:
                parts.append(f"{key.capitalize()}: {kwargs[key]}")
        
        if parts:
            description = " | ".join(parts)
        else:
            description = "No description provided."
    
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _add_event_sync, date, description)
    return f"Successfully added event: {description} on {date}."

async def get_events(date: str = None) -> List[Dict[str, str]]:
    """
    Asynchronously retrieve events from the calendar.
    
    Args:
        date: Optional specific date to filter by (e.g., '2026-05-02')
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _get_events_sync, date)

if __name__ == "__main__":
    async def test_calendar():
        print("--- Testing Calendar Module ---")
        
        print("\n1. Adding events...")
        print(await add_event("2026-05-02", "Meeting with client for Bahria Town plot"))
        print(await add_event("2026-05-02", "Property viewing at DHA Phase 6"))
        print(await add_event("2026-05-03", "Follow-up calls"))
        
        print("\n2. Fetching events for '2026-05-02'...")
        events_today = await get_events("2026-05-02")
        for e in events_today:
            print(f"   -> {e}")
            
        print("\n3. Fetching all events...")
        all_events = await get_events()
        for e in all_events:
            print(f"   -> {e}")
            
        print("\n--- Calendar Test Complete ---")

    asyncio.run(test_calendar())
