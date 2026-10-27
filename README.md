# Sage – Multi-turn LLM Chatbot (Flask + Gemini)

A CS study-buddy chatbot with per-conversation memory and context window management.

## Features
- `POST /api/chat` endpoint backed by the Gemini API
- Multi-turn history kept per `conversation_id`
- Context management: last 10 messages sent verbatim; older messages are folded into a running summary that is added to the system prompt (falls back to plain truncation if summarising fails)
- Custom system prompt (persona: "Sage")
- Error handling: timeouts, rate limits (429), bad key, empty replies, network failures, with clear UI messages
- API key via environment variable (`.env` is git-ignored)

## Setup (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your key into .env
python app.py
```
Open http://127.0.0.1:5000. On macOS/Linux usSe `source venv/bin/activate` and `cp`.

## Try the memory
1. "My name is Abdulla and I'm learning binary search."
2. "Explain it simply."
3. "Give me an example in Java."
4. "What's the time complexity?"
5. "What was my name again, and what topic am I learning?"
