import os
import logging
import asyncio
from typing import Dict, List

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
from openai import OpenAI

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    level=logging.INFO,
)

# --- Environment (Render provides these) ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "changeme")  # set a random string in Render

# --- OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Simple in-memory session stores ---
SESSIONS: Dict[int, List[Dict[str, str]]] = {}
STATE: Dict[int, Dict[str, object]] = {}

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


# --- State helpers ---
def init_state(chat_id: int):
    STATE[chat_id] = {"act": 1, "beat": 1, "recent_choices": [], "awaiting_other": False}

def state_summary(chat_id: int) -> str:
    s = STATE.get(chat_id)
    if not s:
        init_state(chat_id)
        s = STATE[chat_id]
    recent = s["recent_choices"][-3:] if s["recent_choices"] else []
    recent_str = ",".join(recent) if recent else "none"
    return (
        f"SESSION STATE: act={s['act']}, beat={s['beat']}, "
        f"recent_choices={recent_str}. Advance to the next canon beat unless the player chose Other."
    )

def advance_state(chat_id: int, user_text: str):
    s = STATE.get(chat_id)
    if not s:
        init_state(chat_id)
        s = STATE[chat_id]

    choice = (user_text or "").strip().lower()
    s["recent_choices"].append(choice[:20])

    # Reset to start
    if "start again" in choice or "restart" in choice:
        s["act"], s["beat"], s["recent_choices"], s["awaiting_other"] = 1, 1, [], False
        return

    # If we are waiting for a custom action from a previous "Other"
    if s.get("awaiting_other"):
        # If the player now gives any non-numeric action, accept it and advance
        if choice not in {"1", "2", "3", "4"} and not choice.startswith(("1)", "2)", "3)", "4)")):
            s["awaiting_other"] = False
            s["beat"] += 1
            # Act rollovers
            if s["act"] == 1 and s["beat"] > 4:
                s["act"], s["beat"] = 2, 5
            elif s["act"] == 2 and s["beat"] > 8:
                s["act"], s["beat"] = 3, 9
            elif s["act"] == 3 and s["beat"] > 10:
                s["beat"] = 10
        # If they typed 1–4 again while we are waiting, do not advance yet
        return

    # New input, not currently waiting for "Other"
    # Option 4 or explicit "other" sets the waiting flag and does not advance
    if choice in {"4"} or choice.startswith("4)") or "other" in choice:
        s["awaiting_other"] = True
        return

    # Normal numeric choices 1–3 advance
    if choice in {"1", "2", "3"} or choice.startswith(("1)", "2)", "3)")):
        s["beat"] += 1
    else:
        # Free text without choosing 4: treat as a concrete action and advance
        s["beat"] += 1

    # Act rollovers
    if s["act"] == 1 and s["beat"] > 4:
        s["act"], s["beat"] = 2, 5
    elif s["act"] == 2 and s["beat"] > 8:
        s["act"], s["beat"] = 3, 9
    elif s["act"] == 3 and s["beat"] > 10:
        s["beat"] = 10



# --- History helpers ---
def get_history(chat_id: int) -> List[Dict[str, str]]:
    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        init_state(chat_id)
    return SESSIONS[chat_id]

def append_msg(chat_id: int, role: str, content: str):
    hist = get_history(chat_id)
    hist.append({"role": role, "content": content})
    if len(hist) > 40:
        SESSIONS[chat_id] = [hist[0]] + hist[-30:]

# --- OpenAI call (works on all current SDKs) ---
async def call_openai(chat_id: int) -> str:
    msgs = get_history(chat_id).copy()
    msgs.append({"role": "system", "content": state_summary(chat_id)})

    # Format for chat.completions
    messages = [{"role": m["role"], "content": m["content"]} for m in msgs]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=350,
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Last resort text if something unusual happens
        logging.exception("OpenAI call failed")
        return f"Error talking to the model: {e}"


# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SESSIONS[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    init_state(chat_id)
    append_msg(chat_id, "user", "/start")
    reply = await call_openai(chat_id)
    append_msg(chat_id, "assistant", reply)
    await update.message.reply_text(reply)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    if not user_text:
        return
    append_msg(chat_id, "user", user_text)
    advance_state(chat_id, user_text)
    try:
        reply = await call_openai(chat_id)
        append_msg(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
    except Exception as e:
        logging.exception("Error while handling message")
        await update.message.reply_text(f"Error: {e}")

# --- Telegram app in webhook mode ---
tg_app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).updater(None).build()
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# --- FastAPI app for Render ---
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

    # Process the update asynchronously so we return fast to Telegram
    asyncio.create_task(tg_app.process_update(update))

    return {"ok": True}


@api.get("/health")
async def health():
    return {"status": "ok"}
