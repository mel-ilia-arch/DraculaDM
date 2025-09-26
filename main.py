import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest
from telegram.constants import ChatAction

# OpenAI async client
from openai import AsyncOpenAI

# Optional persistence
import redis

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    level=logging.INFO,
)

# ---------- Environment ----------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "changeme")
REDIS_URL = os.environ.get("REDIS_URL")
SESSION_TTL_SEC = int(os.environ.get("SESSION_TTL_SEC", str(60 * 60 * 24 * 7)))  # 7 days

# ---------- OpenAI ----------
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------- Redis client (optional, falls back to memory) ----------
r: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        logging.info("Connected to Redis")
    except Exception:
        r = None
        logging.warning("Could not connect to Redis. Falling back to in-memory state.")

def _rget(key: str) -> Any:
    if not r:
        return None
    data = r.get(key)
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None

def _rset(key: str, value: Any) -> None:
    if r:
        # Set TTL so sessions survive restarts but do not grow forever
        r.set(key, json.dumps(value), ex=SESSION_TTL_SEC)

# ---------- In-memory fallback ----------
SESSIONS: Dict[int, List[Dict[str, str]]] = {}
STATE: Dict[int, Dict[str, Any]] = {}

# ---------- System prompt ----------
SYSTEM_PROMPT = """
You are the Dungeon Master for an interactive gothic horror in the style of Bram Stoker’s Dracula, infused with German Expressionist cinema, avant garde experimentation, and gothic literature. The tone should be unpredictable, horrifying, dreamlike, and at times oddly light hearted.

Address the player as “you”. Keep each message to 3 to 5 sentences. End with 2 to 4 numbered, concrete actions. Always also offer an unnumbered line that reads: Other: describe your own action. Never stall. Do not ask what the player is thinking.

Canon first:
Follow the novel’s sequence unless the player chooses Other or requests a deviation. The default path is canon, deviations are player directed.

Core beats to follow in order unless deviated:
Act I Travel and Arrival
1. England departure preparations
2. Coach and inn in Bistritz with local warnings and charms
3. Night coach on the Borgo Pass and the cloaked driver
4. Arrival at Castle Dracula

Act II Captivity in the Castle
5. Hospitality and rules, shaving glass, the Count’s reaction to blood
6. Locked doors and forbidden rooms, three brides, letters under duress
7. Discovery of crates and chapel, failed signals for help
8. First escape attempt

Act III Breaking Free and Consequences
9. Second attempt and fall or flight from the walls
10. Transition toward recovery and the spreading threat

Optional beats and scenes that may occur at random. Use some of these so every play through feels different. Never force all twenty to appear:
A1. A silent bell tower that tolls only when the moon is hidden
A2. A mirror that reflects a future injury
A3. A supper with villagers who will not sit facing the windows
A4. A letter written in two inks that give different meanings when read by candle and by dawn
A5. A chapel rat that speaks in riddles for a price
A6. A locked music box that opens to a map when wound backward
A7. Footprints that circle the protagonist’s bed from the outside in
A8. A tooth hidden inside a prayer book that fits a secret lock
A9. A portrait whose eyes turn to follow only when the player is silent
A10. A ferryman who accepts memories as fare
A11. A black dog that leads to safety only once
A12. A midnight librarian who trades answers for minutes of life
A13. A child’s toy that points north but only when danger is behind
A14. A dining hall where cutlery rearranges into arrows
A15. A storm lantern that reveals writings invisible to daylight
A16. A well that echoes back different questions than those asked
A17. A frost pattern on windows that marks hidden doors
A18. A dead bat clenched around a ring that opens nothing until warmed
A19. A staircase that adds or removes steps based on courage
A20. A cellar cask whose shadow does not match the shape

Narration rules for forward motion and variety:
1) Each turn must move time or place forward or reveal new information.
2) Never repeat the same fork twice in a row.
3) Vary choice types over time. Do not restate options from the previous turn unless the situation has changed.
4) Commit to consequences. If the player hesitates repeatedly, external events still advance.

Special rules:
1) If the player writes start again or similar wording, reset the story to Act I, Beat 1, and begin anew.
2) The entire story should resolve in about 20 to 30 prompts. Steer narration toward development and an eventual ending.
3) Introduce puzzles and simple logic games in the Castle section. Make the solution space clear but not trivial. Incorrect solutions branch into darker or worse outcomes.
4) The player has a limited currency to purchase resources such as charms, tools, safe passage, or information. Always display currency after any gain or spend. Offer trade offs that force meaningful choices.
5) Dracula’s curse slowly infects the player. Track an affliction meter from 0 to 100. Increase it over time, during setbacks, and with risky actions. As the meter rises, apply penalties such as weakness, unreliable perception, and time loss. At thresholds introduce hallucinations and surreal episodes that may mislead or reveal hidden truths. Always show the current affliction value after changes.
6) Some choices are single chance only. If they are not taken when first offered, mark them as lost opportunities and remove them from future menus. Missing them may impose penalties, close routes, or raise the affliction meter.
7) If the player chooses the number 4 or any number higher than the presented options, or types the word Other, do not advance the story beat. Stop and ask the player to describe their action, or propose two or three specific custom actions they could take, then wait for their input. Continue only after they clarify.
8) If the player’s input is unclear or cannot be interpreted as a valid choice, ask for clarification before proceeding.

Opening behaviour:
On the first reply in any new chat or after /start do not narrate the story. Greet in a Victorian gothic voice and ask:
Shall we begin a new journey, or continue the old one? Choose: 1) New journey 2) Continue   Other: describe your own action.
If New journey is chosen, begin at Act I Beat 1. If Continue is chosen, request a brief checkpoint and resume at the appropriate beat.

State and feedback formatting:
1) When currency or affliction changes, append a short line like: Currency: X   Affliction: Y out of 100.
2) When a choice becomes a lost opportunity, acknowledge it briefly so the player understands the consequence.
3) During hallucinations or surreal episodes, clearly flag what might be unreliable while still allowing useful inferences.

Output format every turn:
Narration in 3 to 5 sentences. Then a numbered list of 2 to 4 actions that are currently available. Then include: Other: describe your own action.
"""

# ---------- Persistent state helpers ----------
def init_state(chat_id: int) -> None:
    s = {"act": 1, "beat": 1, "recent_choices": [], "awaiting_other": False}
    STATE[chat_id] = s
    _rset(f"state:{chat_id}", s)

def get_state(chat_id: int) -> Dict[str, Any]:
    s = _rget(f"state:{chat_id}")
    if s is None:
        s = STATE.get(chat_id)
    if not s:
        init_state(chat_id)
        s = STATE[chat_id]
    return s

def save_state(chat_id: int, s: Dict[str, Any]) -> None:
    STATE[chat_id] = s
    _rset(f"state:{chat_id}", s)

def get_history(chat_id: int) -> List[Dict[str, str]]:
    hist = _rget(f"hist:{chat_id}")
    if hist is None:
        hist = SESSIONS.get(chat_id)
    if not hist:
        hist = [{"role": "system", "content": SYSTEM_PROMPT}]
        SESSIONS[chat_id] = hist
        _rset(f"hist:{chat_id}", hist)
        init_state(chat_id)
    return hist

def save_history(chat_id: int, hist: List[Dict[str, str]]) -> None:
    SESSIONS[chat_id] = hist
    _rset(f"hist:{chat_id}", hist)

def append_msg(chat_id: int, role: str, content: str) -> None:
    hist = get_history(chat_id)
    hist.append({"role": role, "content": content})
    if len(hist) > 80:
        hist = [hist[0]] + hist[-60:]
    save_history(chat_id, hist)

def state_summary(chat_id: int) -> str:
    s = get_state(chat_id)
    recent = s["recent_choices"][-3:] if s["recent_choices"] else []
    recent_str = ",".join(recent) if recent else "none"
    return (
        f"SESSION STATE: act={s['act']}, beat={s['beat']}, "
        f"recent_choices={recent_str}. Advance to the next canon beat unless the player chose Other."
    )

def advance_state(chat_id: int, user_text: str) -> None:
    s = get_state(chat_id)
    choice = (user_text or "").strip().lower()
    s["recent_choices"].append(choice[:20])

    # explicit restart only
    if "start again" in choice or "restart" in choice:
        s = {"act": 1, "beat": 1, "recent_choices": [], "awaiting_other": False}
        save_state(chat_id, s)
        return

    # awaiting 'Other' custom action
    if s.get("awaiting_other"):
        if choice not in {"1", "2", "3", "4"} and not choice.startswith(("1)", "2)", "3)", "4)")):
            s["awaiting_other"] = False
            s["beat"] += 1
            # act rollovers
            if s["act"] == 1 and s["beat"] > 4:
                s["act"], s["beat"] = 2, 5
            elif s["act"] == 2 and s["beat"] > 8:
                s["act"], s["beat"] = 3, 9
            elif s["act"] == 3 and s["beat"] > 10:
                s["beat"] = 10
        save_state(chat_id, s)
        return

    # choosing 'Other' should not advance
    if choice in {"4"} or choice.startswith("4)") or "other" in choice:
        s["awaiting_other"] = True
        save_state(chat_id, s)
        return

    # normal numeric or free text choices advance
    if choice in {"1", "2", "3"} or choice.startswith(("1)", "2)", "3)")):
        s["beat"] += 1
    else:
        s["beat"] += 1

    if s["act"] == 1 and s["beat"] > 4:
        s["act"], s["beat"] = 2, 5
    elif s["act"] == 2 and s["beat"] > 8:
        s["act"], s["beat"] = 3, 9
    elif s["act"] == 3 and s["beat"] > 10:
        s["beat"] = 10

    save_state(chat_id, s)

# ---------- OpenAI call (async, bounded, with retries at SDK level) ----------
async def call_openai(chat_id: int) -> str:
    msgs = get_history(chat_id).copy()
    msgs.append({"role": "system", "content": state_summary(chat_id)})
    messages = [{"role": m["role"], "content": m["content"]} for m in msgs]

    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=350,
                timeout=30,  # API call internal timeout
            ),
            timeout=35,      # outer safety net
        )
        return resp.choices[0].message.content
    except asyncio.TimeoutError:
        return "The narrator falls silent in a storm of static. Try your last action again."
    except Exception as e:
        logging.exception("OpenAI call failed")
        return f"Error talking to the model: {e}"

# ---------- Safe reply helper with one retry ----------
async def safe_reply(message, text: str):
    try:
        await message.reply_text(text)
    except Exception as e:
        logging.warning(f"reply_text failed once: {e}; retrying…")
        await asyncio.sleep(0.7)
        await message.reply_text(text)

# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    save_history(chat_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    init_state(chat_id)
    append_msg(chat_id, "user", "/start")

    # show liveness
    await update.message.chat.send_action(ChatAction.TYPING)

    reply = await call_openai(chat_id)
    append_msg(chat_id, "assistant", reply)
    await safe_reply(update.message, reply)

async def continue_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    append_msg(chat_id, "user", "/continue")

    await update.message.chat.send_action(ChatAction.TYPING)

    reply = await call_openai(chat_id)
    append_msg(chat_id, "assistant", reply)
    await safe_reply(update.message, reply)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    if not user_text:
        return
    append_msg(chat_id, "user", user_text)
    advance_state(chat_id, user_text)

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        reply = await call_openai(chat_id)
        append_msg(chat_id, "assistant", reply)
        await safe_reply(update.message, reply)
    except Exception as e:
        logging.exception("Error while handling message")
        await safe_reply(update.message, f"Error: {e}")

# ---------- Telegram app (webhook mode) ----------
request = HTTPXRequest(
    connection_pool_size=20,
    connect_timeout=5.0,
    read_timeout=30.0,
    write_timeout=30.0,
    pool_timeout=5.0,
)
tg_app: Application = (
    ApplicationBuilder()
    .token(TELEGRAM_BOT_TOKEN)
    .request(request)
    .updater(None)
    .build()
)
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("continue", continue_cmd))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ---------- FastAPI app for Render (or any ASGI) ----------
api = FastAPI()

@api.on_event("startup")
async def _startup():
    await tg_app.initialize()
    await tg_app.start()
    logging.info("Telegram application started in webhook mode")

@api.on_event("shutdown")
async def _shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@api.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return Response(status_code=401)
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    # Process in background so we answer Telegram immediately
    asyncio.create_task(tg_app.process_update(update))
    return {"ok": True}

@api.get("/health")
async def health():
    return {"status": "ok"}
