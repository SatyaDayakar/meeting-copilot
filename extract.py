"""
extract.py — Multi-agent Gen AI pipeline for meeting intelligence.

Agents:
  1. Summarizer        → executive summary + key decisions
  2. ActionExtractor   → action items with owner, task, deadline, priority
  3. RiskDetector      → blockers, open questions, risks flagged in the meeting
  4. ContentGenerator  → follow-up email, Slack message, JIRA-style ticket drafts
  5. SentimentAnalyzer → speaker sentiment + meeting health score

Each agent is a focused LLM call with strict JSON output — composable and testable.
"""

import os
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class ActionItem:
    person: str
    task: str
    deadline: str
    priority: str          # HIGH / MEDIUM / LOW
    category: str          # e.g. Engineering, Design, Marketing


@dataclass
class Risk:
    type: str              # BLOCKER / OPEN_QUESTION / DEPENDENCY / RISK
    description: str
    owner: str             # person mentioned or "Unassigned"
    severity: str          # HIGH / MEDIUM / LOW


@dataclass
class SpeakerSentiment:
    speaker: str
    sentiment: str         # POSITIVE / NEUTRAL / NEGATIVE
    engagement: str        # HIGH / MEDIUM / LOW
    notable_quote: str


@dataclass
class MeetingInsights:
    # Summarizer
    executive_summary: str
    key_decisions: list[str]
    meeting_type: str      # e.g. "Sprint Planning", "Retrospective", "Stakeholder Review"
    topics: list[str]

    # Action extractor
    action_items: list[ActionItem]

    # Risk detector
    risks: list[Risk]
    open_questions: list[str]

    # Sentiment
    speaker_sentiments: list[SpeakerSentiment]
    health_score: int      # 1-10

    # Content generator
    email: str
    slack_message: str
    ticket_drafts: list[dict]   # [{title, description, labels, assignee}]

    # Meta
    processing_time_ms: int = 0


# ─── LLM helper ──────────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_content: str, retries: int = 2) -> dict:
    """Call LLM and parse JSON response. Retries on parse failure."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
            )
            raw = resp.choices[0].message.content
            # Strip markdown fences
            cleaned = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
            # Fix lone backslashes
            cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
            return json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            if attempt == retries:
                raise RuntimeError(f"LLM parse failed after {retries+1} tries: {e}\nRaw: {raw[:300]}")
            time.sleep(1)


# ─── Agent 1: Summarizer ─────────────────────────────────────────────────────

SUMMARIZER_SYSTEM = """You are an expert meeting analyst. Analyze the meeting transcript and return ONLY valid JSON.
No explanation, no markdown fences, no preamble. Keep all strings on single lines.

Return this exact structure:
{
  "executive_summary": "3-4 sentence executive summary",
  "key_decisions": ["decision1", "decision2"],
  "meeting_type": "e.g. Sprint Planning / Standup / Design Review / Stakeholder Update",
  "topics": ["topic1", "topic2", "topic3"]
}"""

def run_summarizer(transcript: str) -> dict:
    return _call_llm(SUMMARIZER_SYSTEM, f"Transcript:\n{transcript}")


# ─── Agent 2: Action Extractor ───────────────────────────────────────────────

ACTION_SYSTEM = """You are a project manager extracting action items from a meeting.
Return ONLY valid JSON. No markdown, no preamble. Single-line strings only.

Return this exact structure:
{
  "action_items": [
    {
      "person": "Name or 'Team'",
      "task": "concrete task description",
      "deadline": "date or 'End of Sprint' or 'TBD'",
      "priority": "HIGH or MEDIUM or LOW",
      "category": "Engineering / Design / Marketing / Operations / Research / Other"
    }
  ]
}
If no action items found, return {"action_items": []}."""

def run_action_extractor(transcript: str) -> dict:
    return _call_llm(ACTION_SYSTEM, f"Transcript:\n{transcript}")


# ─── Agent 3: Risk Detector ──────────────────────────────────────────────────

RISK_SYSTEM = """You are a risk analyst reviewing a meeting transcript.
Return ONLY valid JSON. No markdown, no preamble. Single-line strings only.

Type definitions — use EXACTLY one:
- BLOCKER: something that will stop progress entirely (external approval needed, system down, etc.)
- DEPENDENCY: relies on a third party outside this team (vendor, another team, client sign-off)
- OPEN_QUESTION: a decision or detail that was raised but NOT resolved in the meeting
- RISK: a potential future problem flagged by attendees (timeline slip, resource gap, technical uncertainty)

Do NOT tag self-assigned action items or internal deadlines as risks. Only flag genuine blockers, external dependencies, unresolved questions, or explicitly stated concerns.

Return this exact structure:
{
  "risks": [
    {
      "type": "BLOCKER or OPEN_QUESTION or DEPENDENCY or RISK",
      "description": "concise description",
      "owner": "person responsible or 'Unassigned'",
      "severity": "HIGH or MEDIUM or LOW"
    }
  ],
  "open_questions": ["question1", "question2"]
}
If nothing found, return {"risks": [], "open_questions": []}."""

def run_risk_detector(transcript: str) -> dict:
    return _call_llm(RISK_SYSTEM, f"Transcript:\n{transcript}")


# ─── Agent 4: Sentiment Analyzer ─────────────────────────────────────────────

SENTIMENT_SYSTEM = """You are an organizational psychologist analyzing meeting dynamics.
Return ONLY valid JSON. No markdown, no preamble. Single-line strings only.

Return this exact structure:
{
  "speaker_sentiments": [
    {
      "speaker": "Speaker A / name",
      "sentiment": "POSITIVE or NEUTRAL or NEGATIVE",
      "engagement": "HIGH or MEDIUM or LOW",
      "notable_quote": "a short quote that captures their stance (max 15 words)"
    }
  ],
  "health_score": 7
}
health_score is 1-10 (10 = highly productive, aligned, energized meeting).
IMPORTANT: Even if there is only one speaker or the recording is a monologue, always analyze that speaker's tone, energy, and language and return at least one entry in speaker_sentiments. Never return an empty list."""

def run_sentiment_analyzer(transcript: str) -> dict:
    return _call_llm(SENTIMENT_SYSTEM, f"Transcript:\n{transcript}")


# ─── Agent 5: Content Generator ──────────────────────────────────────────────

CONTENT_SYSTEM = """You are a professional communications assistant. Given meeting context, generate follow-up content.
Return ONLY valid JSON. No markdown fences in the JSON. Single-line strings only (use \\n for newlines inside values).

Return this exact structure:
{
  "email": "Full professional follow-up email with subject line embedded as first line starting 'Subject: ...'",
  "slack_message": "Concise Slack summary under 3 lines using emoji sparingly",
  "ticket_drafts": [
    {
      "title": "short ticket title",
      "description": "acceptance criteria or description",
      "labels": ["label1", "label2"],
      "assignee": "person or 'Unassigned'"
    }
  ]
}
Create one ticket per distinct engineering/design task from action items. Max 5 tickets."""

def run_content_generator(summary: str, action_items: list, risks: list) -> dict:
    context = f"""
Meeting Summary: {summary}

Action Items:
{json.dumps(action_items, indent=2)}

Risks & Blockers:
{json.dumps(risks, indent=2)}
"""
    return _call_llm(CONTENT_SYSTEM, context)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def extract_insights(
    transcript: str,
    progress_callback=None,
) -> MeetingInsights:
    """
    Run the full multi-agent pipeline on a transcript.

    Args:
        transcript:         Plain text or speaker-labeled transcript
        progress_callback:  Optional callable(step: str, pct: int) for UI updates

    Returns:
        MeetingInsights dataclass with all extracted data
    """
    t0 = time.time()

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("Summarizing meeting...", 10)
    summary_data = run_summarizer(transcript)

    _progress("Extracting action items...", 30)
    action_data = run_action_extractor(transcript)

    _progress("Detecting risks & blockers...", 50)
    risk_data = run_risk_detector(transcript)

    _progress("Analyzing meeting sentiment...", 65)
    sentiment_data = run_sentiment_analyzer(transcript)

    _progress("Generating follow-up content...", 80)
    content_data = run_content_generator(
        summary_data.get("executive_summary", ""),
        action_data.get("action_items", []),
        risk_data.get("risks", []),
    )

    _progress("Finalizing insights...", 95)

    action_items = [
        ActionItem(**{k: v for k, v in item.items() if k in ActionItem.__dataclass_fields__})
        for item in action_data.get("action_items", [])
    ]

    risks = [
        Risk(**{k: v for k, v in r.items() if k in Risk.__dataclass_fields__})
        for r in risk_data.get("risks", [])
    ]

    sentiments = [
        SpeakerSentiment(**{k: v for k, v in s.items() if k in SpeakerSentiment.__dataclass_fields__})
        for s in sentiment_data.get("speaker_sentiments", [])
    ]

    elapsed_ms = int((time.time() - t0) * 1000)

    return MeetingInsights(
        executive_summary=summary_data.get("executive_summary", ""),
        key_decisions=summary_data.get("key_decisions", []),
        meeting_type=summary_data.get("meeting_type", "General Meeting"),
        topics=summary_data.get("topics", []),
        action_items=action_items,
        risks=risks,
        open_questions=risk_data.get("open_questions", []),
        speaker_sentiments=sentiments,
        health_score=sentiment_data.get("health_score", 5),
        email=content_data.get("email", ""),
        slack_message=content_data.get("slack_message", ""),
        ticket_drafts=content_data.get("ticket_drafts", []),
        processing_time_ms=elapsed_ms,
    )


def insights_to_dict(insights: MeetingInsights) -> dict:
    """Convert MeetingInsights to a plain dict for serialization / display."""
    d = asdict(insights)
    return d