import asyncio
import sys
from pathlib import Path

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from Conversation.conversation import Session, _advance_stage_on_user

async def test_semantic_state():
    print("--- Testing Semantic State Tracking ---")
    session = Session(session_id="test_state_123")
    
    # 1. Test category with typo
    print("User: I want a huse")
    await _advance_stage_on_user(session, "I want a huse")
    print(f"   -> Stage: {session.stage} | Category: {session.selected_category}")
    
    # 2. Test subtype with typo
    print("User: give me 5 mlra one")
    await _advance_stage_on_user(session, "give me 5 mlra one")
    print(f"   -> Stage: {session.stage} | Subtype: {session.selected_subtype} | Price: {session.selected_price}")
    
    if session.selected_subtype == "5 Marla House":
        print("SUCCESS: Semantic state tracking works!")
    else:
        print("FAILURE: Semantic state tracking failed.")

if __name__ == "__main__":
    asyncio.run(test_semantic_state())
