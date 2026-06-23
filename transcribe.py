"""
transcribe.py — Audio transcription via Groq Whisper API.
No local Whisper or torch required — runs on Streamlit Cloud.
"""

import os
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str
    confidence: float


@dataclass
class TranscriptResult:
    full_text: str
    segments: list
    language: str
    duration: float
    speaker_count: int


def transcribe_audio(file_path: str, model_size: str = "small", language=None) -> TranscriptResult:
    """
    Transcribe audio using Groq's Whisper API.
    No local model download — fast and cloud-friendly.
    """
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file),
            model="whisper-large-v3",
            response_format="verbose_json",
            language=language,
        )

    raw_segments = getattr(response, "segments", []) or []
    duration = raw_segments[-1]["end"] if raw_segments else 0.0

    segments = []
    for seg in raw_segments:
        segments.append(TranscriptSegment(
            start=round(seg["start"], 2),
            end=round(seg["end"], 2),
            speaker="Speaker A",
            text=seg["text"].strip(),
            confidence=round(min(1.0, max(0.0, __import__("math").exp(seg.get("avg_logprob", -0.5)))), 3),
        ))

    # Gap-based speaker detection
    if len(segments) > 1:
        speaker_idx = 0
        current_speaker = "Speaker A"
        for i in range(1, len(segments)):
            gap = segments[i].start - segments[i - 1].end
            if gap > 2.0:
                speaker_idx = (speaker_idx + 1) % 4
                current_speaker = f"Speaker {chr(65 + speaker_idx)}"
            segments[i].speaker = current_speaker

    unique_speakers = {s.speaker for s in segments}

    return TranscriptResult(
        full_text=response.text.strip(),
        segments=segments,
        language=getattr(response, "language", "en"),
        duration=duration,
        speaker_count=len(unique_speakers),
    )


def format_transcript_for_llm(result: TranscriptResult) -> str:
    lines = []
    for seg in result.segments:
        mins = int(seg.start // 60)
        secs = int(seg.start % 60)
        lines.append(f"{seg.speaker} [{mins:02d}:{secs:02d}]: {seg.text}")
    return "\n".join(lines)