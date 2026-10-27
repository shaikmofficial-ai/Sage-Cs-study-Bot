"""Multi-turn LLM chatbot (Flask + Google Gemini) with context window management."""
import logging
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import errors, types

load_dotenv()  # reads .env (which is git-ignored)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_MESSAGES = 10        # recent messages sent verbatim (5 user/model turns)
MAX_INPUT_CHARS = 2000   # reject absurdly long user messages
TIMEOUT_MS = 30_000      # 30s API timeout

SYSTEM_PROMPT = (
    "You are Sage, a friendly and patient study buddy for computer science students. "
    "Explain concepts step by step with small, concrete examples, and keep answers "
    "concise (under ~150 words unless asked for more). If a question is ambiguous, ask "
    "one clarifying question. Stay on topic: programming, CS fundamentals, AI/ML and "
    "study techniques. Politely steer unrelated requests back to learning. If you are "
    "unsure about something, say so instead of guessing."
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatbot")

app = Flask(__name__)
client = (
    genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    if API_KEY
    else None
)

# In-memory store: conversation_id -> {"messages": [{"role", "text"}], "summary": str}
# (Resets when the server restarts. Fine for this task; use a DB for production.)
conversations: dict[str, dict] = {}


class ChatError(Exception):
    """An error with a user-friendly message and an HTTP status."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message, self.status = message, status


def to_contents(messages):
    return [types.Content(role=m["role"], parts=[types.Part(text=m["text"])]) for m in messages]


def call_llm(messages, summary=""):
    """Send history to Gemini and return reply text. Raises ChatError on failure."""
    if client is None:
        raise ChatError("Server is missing GEMINI_API_KEY. Add it to your .env file.", 500)

    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nSummary of the earlier part of this conversation:\n{summary}"

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=to_contents(messages),
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.7),
        )
        text = (resp.text or "").strip()
        if not text:
            raise ChatError("The model returned an empty response. Please try rephrasing.", 502)
        return text
    except ChatError:
        raise
    except errors.APIError as e:
        log.warning("Gemini API error %s: %s", e.code, e)
        if e.code == 429:
            raise ChatError("Rate limit reached. Please wait a moment and try again.", 429)
        if e.code in (401, 403):
            raise ChatError("The API key was rejected. Check GEMINI_API_KEY.", 500)
        raise ChatError("The AI service had a problem. Please try again.", 502)
    except Exception as e:  # network failure, timeout, etc.
        log.warning("LLM call failed: %r", e)
        raise ChatError("Couldn't reach the AI service (network issue or timeout). Try again.", 504)


def summarise(old_summary, overflow):
    """Fold messages that fell out of the window into a running summary."""
    transcript = "\n".join(f"{m['role']}: {m['text']}" for m in overflow)
    prompt = (
        "Update the running summary of a conversation. Keep key facts, names, user goals "
        "and decisions in under 120 words. Output only the summary.\n\n"
        f"Current summary:\n{old_summary or '(none)'}\n\nNew messages:\n{transcript}"
    )
    try:
        return call_llm([{"role": "user", "text": prompt}])
    except ChatError:
        log.warning("Summarisation failed; falling back to plain truncation.")
        return old_summary  # degrade gracefully: older messages are simply dropped


def manage_context(conv):
    """Keep the last MAX_MESSAGES verbatim; summarise everything older."""
    msgs = conv["messages"]
    if len(msgs) <= MAX_MESSAGES:
        return
    cut = len(msgs) - MAX_MESSAGES
    while cut < len(msgs) and msgs[cut]["role"] != "user":  # window must start on a user turn
        cut += 1
    overflow, conv["messages"] = msgs[:cut], msgs[cut:]
    if overflow:
        conv["summary"] = summarise(conv["summary"], overflow)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="Please type a message."), 400
    if len(message) > MAX_INPUT_CHARS:
        return jsonify(error=f"Message too long (max {MAX_INPUT_CHARS} characters)."), 400

    cid = data.get("conversation_id") or str(uuid.uuid4())
    conv = conversations.setdefault(cid, {"messages": [], "summary": ""})

    # Only commit the user message to history if the API call succeeds, so retries stay clean.
    pending = conv["messages"] + [{"role": "user", "text": message}]
    try:
        reply = call_llm(pending, conv["summary"])
    except ChatError as e:
        return jsonify(error=e.message), e.status

    conv["messages"] = pending + [{"role": "model", "text": reply}]
    manage_context(conv)
    return jsonify(conversation_id=cid, reply=reply)


@app.post("/api/reset")
def reset():
    conversations.pop((request.get_json(silent=True) or {}).get("conversation_id"), None)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
