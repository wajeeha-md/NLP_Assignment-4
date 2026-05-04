"""
backend/Conversation/conversation.py

Conversation manager for Ali — a Pakistani real estate assistant chatbot.
Handles session management, context window trimming, stage tracking,
off-topic policy enforcement, and streaming Ollama integration.

Context strategy
----------------
Small models (2B) cannot reliably re-infer what a user chose several turns
ago from raw chat history alone.  Instead, every turn we inject an explicit
CONVERSATION STATE block into the system prompt that names the stage, the
chosen category, and the chosen subtype as ground truth.  The model never
has to guess — we tell it exactly what has already been decided.

The old "pin the first greeting turn" approach is intentionally removed.
Re-inserting Ali's opening "What category would you like?" into a late-turn
context caused the model to think it still needed to ask that question.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator

import ollama
from RAG.retrieval import retrieve, semantic_match
from CRM.crm import get_user_info, update_user_info, create_user
from Tools.orchestrator import orchestrator
from Tools.calculator import calculate
from Tools.weather import get_weather
from Tools.calendar import add_event, get_events

# Register tools with descriptive instructions
orchestrator.register("get_user_info", get_user_info, "Retrieves profile details (budget, preferences) for a user ID.")
orchestrator.register("update_user_info", update_user_info, "Updates a specific profile field for a user ID. Handles semantic field matching.")
orchestrator.register("create_user", create_user, "Creates a new user profile record.")
orchestrator.register("calculate", calculate, "Evaluates mathematical expressions safely.")
orchestrator.register("get_weather", get_weather, "Fetches weather for a city. Required: 'city'.")
orchestrator.register("add_event", add_event, "Schedules a property visit or meeting. Required: 'date' (YYYY-MM-DD), 'description' (What the event is).")
orchestrator.register("get_events", get_events, "Lists calendar events. Optional: 'date' (YYYY-MM-DD) to filter.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "ali-realestate"
SESSION_TTL_SECONDS = 30 * 60   # 30-minute inactivity timeout
MAX_HISTORY_TURNS = 10           # sliding window: last N user+assistant pairs

# ── Inventory ───────────────────────────────────────────────────────────────
# Single source of truth.  Used in prompts AND in subtype extraction logic.

INVENTORY: dict[str, list[tuple[str, str]]] = {
    "Shops": [
        ("5 Marla Shop",   "PKR 1.2 Crore"),
        ("8 Marla Shop",   "PKR 2.1 Crore"),
        ("1 Kanal Shop",   "PKR 3.8 Crore"),
    ],
    "Houses/Villas": [
        ("5 Marla House",  "PKR 1.8 Crore"),
        ("7 Marla House",  "PKR 2.6 Crore"),
        ("10 Marla House", "PKR 4.2 Crore"),
        ("1 Kanal Villa",  "PKR 8.5 Crore"),
    ],
    "Apartments": [
        ("1 Bedroom Apt",  "PKR 55 Lac"),
        ("2 Bedroom Apt",  "PKR 95 Lac"),
        ("3 Bedroom Apt",  "PKR 1.5 Crore"),
    ],
}

def _inventory_block() -> str:
    """Render the full inventory as a formatted string for system prompts."""
    lines: list[str] = ["AUTHORISED INVENTORY — THE ONLY PROPERTIES THAT EXIST"]
    lines.append("=" * 55)
    for category, items in INVENTORY.items():
        lines.append(category.upper())
        for name, price in items:
            lines.append(f"  - {name:<20}: {price}")
        lines.append("")
    lines.append(
        "DO NOT invent locations, addresses, square footage, or prices.\n"
        "DO NOT modify or estimate any listed price.\n"
        "If asked for a size not listed, say it is unavailable and show what IS listed."
    )
    return "\n".join(lines)

CORE_IDENTITY = (
    "You are Ali, a highly capable and friendly AI real estate agent for a property agency in Pakistan.\n\n"
    "ROUTING & CAPABILITY POLICY:\n"
    "1. REAL ESTATE: If the user asks about properties, prices, or project details, use the AUTHORISED INVENTORY and the 'Context' provided from our documents (RAG).\n"
    "2. TOOLS: You have access to tools for CALCULATIONS, WEATHER, CALENDAR, and CRM (User Memory). If a query can be handled by a tool, invoke it immediately using the JSON format.\n"
    "3. CASUAL CONVERSATION: You are allowed to engage in brief, friendly conversation (greetings, 'how are you', 'thank you').\n"
    "4. REJECTION: Only politely decline a request if it is completely outside your capabilities AND no tool can handle it (e.g., medical advice, writing complex code).\n\n"
    + _inventory_block()
)

# ── Stage goal hints ─────────────────────────────────────────────────────────

STAGE_HINTS: dict[str, str] = {
    "greeting": (
        "CURRENT GOAL: Greet the customer warmly. "
        "Ask which category they want — Shops, Houses/Villas, or Apartments. "
        "Do NOT list prices yet."
    ),
    "category_selection": (
        "CURRENT GOAL: The customer has chosen a category (see CONVERSATION STATE). "
        "List ONLY the subtypes and exact PKR prices for that category from the "
        "AUTHORISED INVENTORY. Do NOT show subtypes from other categories."
    ),
    "subtype_selection": (
        "CURRENT GOAL: The customer has selected a specific subtype (see CONVERSATION STATE). "
        "State its exact price. Briefly describe it (great for families / good investment). "
        "Then ask: would they like to schedule a visit or speak to an agent? "
        "Do NOT offer other subtypes or re-list the category menu."
    ),
    "closing": (
        "CURRENT GOAL: Arrange a property visit or agent call for the chosen property "
        "(see CONVERSATION STATE). Be warm, confirm which property they selected, "
        "and offer clear next steps."
    ),
}

# ---------------------------------------------------------------------------
# Stage-transition keyword tables  (checked on USER messages only)
# ---------------------------------------------------------------------------

# keyword → canonical category name in INVENTORY
_CATEGORY_MAP: list[tuple[str, str]] = [
    ("shop",      "Shops"),
    ("house",     "Houses/Villas"),
    ("villa",     "Houses/Villas"),
    ("apartment", "Apartments"),
    ("flat",      "Apartments"),
]

# (size_keyword, category) → canonical subtype name
# category=None means the keyword is unambiguous across all categories
_SUBTYPE_MAP: list[tuple[str, Optional[str], str]] = [
    # Shops
    ("5 marla",   "Shops",         "5 Marla Shop"),
    ("8 marla",   "Shops",         "8 Marla Shop"),
    ("1 kanal",   "Shops",         "1 Kanal Shop"),
    # Houses / Villas
    ("5 marla",   "Houses/Villas", "5 Marla House"),
    ("7 marla",   "Houses/Villas", "7 Marla House"),
    ("10 marla",  "Houses/Villas", "10 Marla House"),
    ("1 kanal",   "Houses/Villas", "1 Kanal Villa"),
    # Apartments (bedroom count is unambiguous — no category check needed)
    ("1 bedroom", None,            "1 Bedroom Apt"),
    ("2 bedroom", None,            "2 Bedroom Apt"),
    ("3 bedroom", None,            "3 Bedroom Apt"),
    ("1bed",      None,            "1 Bedroom Apt"),
    ("2bed",      None,            "2 Bedroom Apt"),
    ("3bed",      None,            "3 Bedroom Apt"),
]

# User explicitly requests a booking/visit → subtype_selection → closing
_CLOSING_KW: list[str] = [
    "schedule", "book a visit", "i'd like to visit", "i want to visit",
    "arrange a visit", "speak to an agent", "contact agent", "book agent",
    "i'd like to schedule",
]

# Broad set for off-topic detection
_REALESTATE_KW: list[str] = [
    "shop", "house", "villa", "apartment", "flat", "property", "properties",
    "marla", "kanal", "bedroom", "price", "pkr", "crore", "lac", "lakh",
    "buy", "purchase", "rent", "visit", "agent", "booking", "schedule",
    "real estate", "plot", "area", "size", "category",
    "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye",
    "yes", "no", "okay", "ok", "sure", "please", "show", "tell",
    "more", "info", "interested", "looking",
    "weather", "calculate", "math", "update", "budget", "crm", "event", "calendar", "add"
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """All mutable state for one user conversation.

    Attributes
    ----------
    session_id        : Unique UUID for this session.
    history           : Ordered list of {role, content} message dicts.
    stage             : Current position in the conversation flow.
    selected_category : Canonical category name chosen by the user, or None.
    selected_subtype  : Canonical subtype name chosen by the user, or None.
    selected_price    : Price string for the chosen subtype, or None.
    last_active       : Unix timestamp of the last activity (for TTL).
    """

    session_id:        str
    history:           list[dict]     = field(default_factory=list)
    stage:             str            = "greeting"
    selected_category: Optional[str]  = None
    selected_subtype:  Optional[str]  = None
    selected_price:    Optional[str]  = None
    user_info:         dict           = field(default_factory=dict)
    last_active:       float          = field(default_factory=time.time)

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def create_session() -> str:
    """Create a new session, store it, and return its UUID string."""
    sid = str(uuid.uuid4())
    _sessions[sid] = Session(session_id=sid)
    return sid


def get_session(session_id: str) -> Optional[Session]:
    """Return the Session for session_id, or None if expired / not found.

    Triggers a purge of all expired sessions as a side-effect.
    """
    _purge_expired_sessions()
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    """Remove a session from the store immediately."""
    _sessions.pop(session_id, None)

def get_session_info(session_id: str) -> Optional[dict]:
    """Return a JSON-safe summary of session state (no full history)."""
    session = get_session(session_id)
    if session is None:
        return None
    return {
        "session_id":        session.session_id,
        "stage":             session.stage,
        "selected_category": session.selected_category,
        "selected_subtype":  session.selected_subtype,
        "selected_price":    session.selected_price,
        "turn_count":        len(session.history) // 2,
    }

def _purge_expired_sessions() -> None:
    """Delete all sessions inactive longer than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s.last_active > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]

# Off-topic detection removed in favor of LLM-based agentic routing policy
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage tracking + state extraction  (USER messages only)
# ---------------------------------------------------------------------------

async def _advance_stage_on_user(session: Session, user_message: str) -> None:
    """Advance session stage and extract chosen category / subtype from the
    USER's message.  The assistant's own text NEVER drives state changes.

    Side-effects
    ------------
    - session.stage             may be advanced one step
    - session.selected_category may be set when the user picks a category
    - session.selected_subtype  may be set when the user picks a size
    - session.selected_price    may be set alongside selected_subtype

    Stages flow strictly one way:
        greeting → category_selection → subtype_selection → closing
    """
    lower = user_message.lower()

    # ── greeting → category_selection ───────────────────────────────────────
    if session.stage == "greeting":
        matched_cat = None
        for kw, canonical in _CATEGORY_MAP:
            if kw in lower:
                matched_cat = canonical
                break
        
        # Hybrid semantic fallback
        if not matched_cat:
            cat_options = [kw for kw, _ in _CATEGORY_MAP]
            best_kw = await semantic_match(lower, cat_options, threshold=0.45)
            if best_kw:
                matched_cat = dict(_CATEGORY_MAP).get(best_kw)
        
        if matched_cat:
            session.selected_category = matched_cat
            session.stage = "category_selection"

    # ── category_selection → subtype_selection ───────────────────────────────
    elif session.stage == "category_selection":
        matched_subtype = None
        
        # Try exact match first
        for size_kw, cat_filter, subtype_name in _SUBTYPE_MAP:
            if size_kw in lower:
                if cat_filter is None or session.selected_category == cat_filter:
                    matched_subtype = (subtype_name, cat_filter)
                    break
        
        # Hybrid semantic fallback
        if not matched_subtype:
            relevant_subtypes = [
                (skw, cf, sn) for skw, cf, sn in _SUBTYPE_MAP 
                if cf is None or session.selected_category == cf
            ]
            size_options = [s[0] for s in relevant_subtypes]
            best_size_kw = await semantic_match(lower, size_options, threshold=0.45)
            
            if best_size_kw:
                for skw, cf, sn in relevant_subtypes:
                    if skw == best_size_kw:
                        matched_subtype = (sn, cf)
                        break
        
        if matched_subtype:
            subtype_name, cat_filter = matched_subtype
            session.selected_subtype = subtype_name
            category = cat_filter or session.selected_category or ""
            for name, price in INVENTORY.get(category, []):
                if name == subtype_name:
                    session.selected_price = price
                    break
            session.stage = "subtype_selection"

    # ── subtype_selection → closing ──────────────────────────────────────────
    elif session.stage == "subtype_selection":
        if any(kw in lower for kw in _CLOSING_KW):
            session.stage = "closing"

    # "closing" is terminal

# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------

def _trimmed_history(session: Session) -> list[dict]:
    """Return at most MAX_HISTORY_TURNS user+assistant pairs from history.

    No greeting pinning is performed here.  Context is preserved through
    the explicit CONVERSATION STATE block in the system prompt instead,
    which is a far more reliable mechanism for small models.
    """
    max_entries = MAX_HISTORY_TURNS * 2   # each turn = one dict
    if len(session.history) > max_entries:
        return list(session.history[-max_entries:])
    return list(session.history)

# ---------------------------------------------------------------------------
# Prompt orchestration
# ---------------------------------------------------------------------------

def _build_conversation_state(session: Session) -> str:
    """Render the CONVERSATION STATE block injected into every system prompt.

    This gives the model explicit, authoritative ground truth about what has
    already been decided so it never needs to infer it from raw history.
    """
    lines = [
        "CONVERSATION STATE  (tracked by the system — treat as ground truth)",
        "-" * 60,
        f"Stage             : {session.stage}",
        f"Category chosen   : {session.selected_category or 'not yet chosen'}",
        f"Subtype chosen    : {session.selected_subtype  or 'not yet chosen'}",
        f"Price confirmed   : {session.selected_price    or 'not yet confirmed'}",
        f"CRM User Info     : {session.user_info}",
        "-" * 60,
        "IMPORTANT: Do NOT ask the customer again about choices already made above.",
        "           Focus only on the CURRENT GOAL for the current stage.",
    ]
    return "\n".join(lines)


def _build_system_prompt(session: Session) -> dict:
    """Compose the dynamic system prompt turn-by-turn."""
    tool_instructions = orchestrator.get_system_instructions()
    
    parts = [
        CORE_IDENTITY,
        tool_instructions,
        _build_conversation_state(session),
        STAGE_HINTS.get(session.stage, ""),
    ]
    return {"role": "system", "content": "\n\n".join(parts)}

# ---------------------------------------------------------------------------
# Ollama streaming integration
# ---------------------------------------------------------------------------

async def stream_response(
    session_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Async generator that drives one complete conversational turn.

    Pipeline
    --------
    1.  Validate session.
    2.  Detect off-topic content.
    3.  Advance stage + extract state from USER message.
    4.  Append user turn to history.
    5.  Build: [dynamic_system_prompt] + [trimmed_history].
    6.  Stream Ollama; yield tokens as they arrive.
    7.  Append complete assistant turn to history.

    Yields
    ------
    str
        Individual content tokens, or a single [ERROR] string on failure.
    """
    session = get_session(session_id)
    if session is None:
        yield "[ERROR] Session not found or expired. Please start a new session."
        return

    session.last_active = time.time()

    if not session.history:
        session.user_info = await get_user_info("test_user_123")

    # ── 1. Stage + state extraction (USER only) ──────────────────────────────
    await _advance_stage_on_user(session, user_message)

    # ── 2. Append user turn before building the context window ───────────────
    session.history.append({"role": "user", "content": user_message})

    # ── 3. Build final message list ──────────────────────────────────────────
    system_msg = _build_system_prompt(session)
    
    # Grounding fallback rule
    system_msg["content"] += "\n\n[INSTRUCTION] Answer the user's question primarily using the 'Context' chunks provided in their prompt. If the Context does not contain the answer, you may use your INVENTORY or say you don't know."
    
    messages = [system_msg] + _trimmed_history(session)
    
    # ── RAG Injection ────────────────────────────────────────────────────────
    rag_start = time.perf_counter()
    if messages and messages[-1].get("role") == "user":
        try:
            chunks = await retrieve(user_message, k=3)
            if chunks:
                context_str = "\n\n".join(chunks)
                # Overwrite the latest user message strictly for the ollama inference payload
                messages[-1]["content"] = f"Context:\n{context_str}\n\nQuestion:\n{user_message}"
        except Exception as e:
            # Fallback smoothly if retrieval errors out
            pass
    rag_end = time.perf_counter()
    print(f"[PERF] RAG retrieval took: {rag_end - rag_start:.3f}s")

    # ── 5. Stream ─────────────────────────────────────────────────────────────
    client = ollama.AsyncClient()
    
    max_tool_iterations = 3
    for _ in range(max_tool_iterations):
        full_response: list[str] = []

        try:
            async for chunk in await client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                think=False,
            ):
                token: str = chunk.message.content or ""
                if token:
                    full_response.append(token)
                    yield token

        except ollama.ResponseError as exc:
            yield f"\n[ERROR] Ollama ResponseError: {exc.error}"
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()
            return

        except Exception as exc:  # noqa: BLE001
            yield f"\n[ERROR] Could not reach Ollama: {exc}"
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()
            return
            
        response_text = "".join(full_response)
        
        # ── Check for tool calls ──────────────────────────────────────────────
        calls = orchestrator.parse_tool_calls(response_text)
        if not calls:
            # No tools called, save assistant turn and break
            assistant_turn = {"role": "assistant", "content": response_text}
            session.history.append(assistant_turn)
            session.last_active = time.time()
            break
            
        # Execute tools
        yield "\n\n[System: Executing tools...]\n"
        tool_start = time.perf_counter()
        results = await orchestrator.execute_all(response_text)
        tool_end = time.perf_counter()
        print(f"[PERF] Tool execution took: {tool_end - tool_start:.3f}s")
        
        # Append assistant's response that contained the tool call
        messages.append({"role": "assistant", "content": response_text})
        
        # Append the tool results as a user message
        tool_feedback = "Tool Results:\n"
        for r in results:
            cached_flag = " (CACHED)" if r['execution'].get('cached') else ""
            tool_feedback += f"Result for {r['tool_call']['tool_name']}{cached_flag}: {r['execution']}\n"
        tool_feedback += "\nPlease provide the final answer or next steps based on these results."
        
        messages.append({"role": "user", "content": tool_feedback})
        yield "\n[System: Tools executed. Generating final answer...]\n\n"

# ---------------------------------------------------------------------------
# Multi-turn smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    TEST_CONVERSATION = [
        "What is the weather in Lahore right now?",
        "Can you calculate (150 * 5) / 2 for me?",
        "Update my budget to 2 Crore in the CRM, and let me know my current info.",
    ]

    async def run_test() -> None:
        """Simulate the 6-turn test dialogue on a single session."""
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        sid = create_session()
        print("\n=== Ali Real Estate Chatbot — Smoke Test ===")
        print(f"Session ID: {sid}\n")

        for i, user_msg in enumerate(TEST_CONVERSATION, 1):
            label = f"Turn {i}"
            session = get_session(sid)
            if session:
                state_str = (
                    f"stage={session.stage} | "
                    f"category={session.selected_category} | "
                    f"subtype={session.selected_subtype} | "
                    f"price={session.selected_price}"
                )
            else:
                state_str = "session not found"

            print(f"[{label}] User  : {user_msg}")
            print(f"          State : {state_str}")
            print(f"          Ali   : ", end="", flush=True)

            async for token in stream_response(sid, user_msg):
                print(token, end="", flush=True)

            session = get_session(sid)
            if session:
                state_after = (
                    f"stage={session.stage} | "
                    f"category={session.selected_category} | "
                    f"subtype={session.selected_subtype} | "
                    f"price={session.selected_price}"
                )
            else:
                state_after = "session not found"

            print(f"\n          → {state_after}\n")

        print("=== Test Complete ===\n")

    asyncio.run(run_test())