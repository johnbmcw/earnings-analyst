"""AI Earnings Call Analyst: paste a transcript, get a structured analysis from Claude."""

import json
import os
import re
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

MODEL = "claude-opus-5-5"
ENV_FILE = Path(__file__).parent / ".env"

# JSON schema the API forces Claude's answer to follow (structured outputs).
# Counts like "2-3 quotes" can't be enforced by the schema, so the prompt asks for them.
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "sentiment": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        "sentiment_justification": {"type": "string"},
        "key_quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["speaker", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "key_risks", "sentiment", "sentiment_justification", "key_quotes"],
    "additionalProperties": False,
}

PROMPT = """You are an equity research analyst. Analyze the earnings call transcript below.

Return:
- summary: 2-3 sentences.
- key_risks: the main risks management or analysts raised (short phrases).
- sentiment: bullish, neutral, or bearish, from an investor's point of view.
- sentiment_justification: exactly one sentence.
- key_quotes: 2-3 quotes from management, copied word for word from the transcript, with the speaker's name.

Use only information in the transcript. Do not invent figures.

<transcript>
{transcript}
</transcript>"""


def load_credentials():
    """Optional .env support. With no key, the SDK falls back to your `ant auth login` profile."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
    # An empty ANTHROPIC_API_KEY= line would take priority over the login profile, so drop it.
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        os.environ.pop("ANTHROPIC_API_KEY", None)


AUTH_HELP = (
    "Claude API authentication failed. Either run `ant auth login` in a terminal "
    "(sign-ins expire; `ant auth status` shows yours), or put `ANTHROPIC_API_KEY=your-key` "
    "in a `.env` file in the project folder. Then restart the app."
)


def normalize(text):
    """Lowercase, straighten curly quotes, and collapse whitespace so line breaks don't break the match."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def analyze(transcript):
    """Send the transcript to Claude and return the parsed JSON analysis."""
    # No api_key argument: the SDK uses ANTHROPIC_API_KEY if set, otherwise the `ant auth login` profile.
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": PROMPT.format(transcript=transcript)}],
        output_config={
            "effort": "low",  # extraction task; low = faster and cheaper
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to analyze this transcript (safety refusal). Try a different transcript.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The response was cut off before it finished. Try a shorter transcript.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# ---------------- UI ----------------
st.title("AI Earnings Call Analyst")

load_credentials()

transcript = st.text_area("Paste an earnings call transcript", height=300)

if st.button("Analyze"):
    if not transcript.strip():
        st.error("Paste a transcript first.")
        st.stop()

    with st.spinner("Asking Claude..."):
        try:
            result = analyze(transcript)
        except anthropic.AuthenticationError:  # credentials found but rejected (401)
            st.warning(AUTH_HELP)
            st.stop()
        except TypeError as e:  # SDK raises TypeError when it finds no credentials at all
            if "authentication method" not in str(e):
                raise  # a real bug, not a sign-in problem: show the full traceback
            st.warning(AUTH_HELP)
            st.stop()
        except anthropic.APIStatusError as e:  # e.g. no credits, rate limit, server error
            st.error(f"API error ({e.status_code}): {e.message}")
            st.stop()
        except anthropic.APIConnectionError:
            st.error("Could not reach the Anthropic API. Check your internet connection.")
            st.stop()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Summary")
    st.write(result["summary"])

    st.subheader("Sentiment")
    st.write(f"**{result['sentiment'].capitalize()}**: {result['sentiment_justification']}")

    st.subheader("Key risks")
    for risk in result["key_risks"]:
        st.markdown(f"- {risk}")

    # Quote check: flag any quote that doesn't appear word for word in the transcript.
    st.subheader("Key management quotes")
    for q in result["key_quotes"]:
        found = normalize(q["quote"]) in normalize(transcript)
        label = "" if found else " ⚠️ *not found verbatim*"
        st.markdown(f"> \"{q['quote']}\"\n>\n> **{q['speaker']}**{label}")
