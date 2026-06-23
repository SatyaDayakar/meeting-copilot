"""
transcribe.py — Enhanced transcription with speaker diarization and timestamps.

Uses OpenAI Whisper for STT. Optionally uses pyannote.audio for speaker diarization
when a HuggingFace token is available (set HF_TOKEN env var).
"""

import os
import whisper
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptSegment:
    start: float          # seconds
    end: float            # seconds
    speaker: str          # e.g. "Speaker A"
    text: str
    confidence: float     # avg log-prob converted to 0-1


@dataclass
class TranscriptResult:
    full_text: str
    segments: list[TranscriptSegment]
    language: str
    duration: float       # seconds
    speaker_count: int


def _load_whisper(model_size: str = "base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_size, device=device)


def _try_diarize(audio_path: str, num_segments: int) -> Optional[dict]:
    """
    Attempt speaker diarization via pyannote.audio.
    Returns dict mapping (start, end) → speaker label, or None if unavailable.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        mapping = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            mapping[(round(turn.start, 2), round(turn.end, 2))] = speaker
        return mapping
    except Exception:
        return None


def _assign_speaker(start: float, end: float, diarization: Optional[dict]) -> str:
    """Find best-matching speaker for a whisper segment."""
    if not diarization:
        return "Speaker"
    best_overlap = 0.0
    best_speaker = "Speaker"
    for (d_start, d_end), speaker in diarization.items():
        overlap = max(0, min(end, d_end) - max(start, d_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    # Remap to friendly names
    label_map = {}
    counter = [0]
    def friendly(label):
        if label not in label_map:
            letter = chr(65 + counter[0] % 26)
            label_map[label] = f"Speaker {letter}"
            counter[0] += 1
        return label_map[label]
    return friendly(best_speaker)


def _logprob_to_confidence(avg_logprob: float) -> float:
    """Convert whisper's avg_logprob to a 0–1 confidence score."""
    import math
    return round(min(1.0, max(0.0, math.exp(avg_logprob))), 3)


def transcribe_audio(
    file_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> TranscriptResult:
    """
    Transcribe audio and return a rich TranscriptResult with:
    - Full transcript text
    - Per-segment timestamps, speaker labels, and confidence
    - Auto-detected language and total duration
    """
    model = _load_whisper(model_size)

    options = {"word_timestamps": False, "verbose": False}
    if language:
        options["language"] = language

    raw = model.transcribe(file_path, **options)

    whisper_segments = raw.get("segments", [])
    duration = whisper_segments[-1]["end"] if whisper_segments else 0.0

    # Attempt diarization (graceful fallback)
    diarization = _try_diarize(file_path, len(whisper_segments))

    # Track unique speakers for count
    seen_speakers: set[str] = set()
    segments: list[TranscriptSegment] = []

    for seg in whisper_segments:
        speaker = _assign_speaker(seg["start"], seg["end"], diarization)
        seen_speakers.add(speaker)
        segments.append(TranscriptSegment(
            start=round(seg["start"], 2),
            end=round(seg["end"], 2),
            speaker=speaker,
            text=seg["text"].strip(),
            confidence=_logprob_to_confidence(seg.get("avg_logprob", -0.5)),
        ))

   

    return TranscriptResult(
        full_text=raw["text"].strip(),
        segments=segments,
        language=raw.get("language", "en"),
        duration=duration,
        speaker_count=len({s.speaker for s in segments})
    )


def format_transcript_for_llm(result: TranscriptResult) -> str:
    """
    Format transcript segments as 'Speaker X [00:00]: text' for LLM consumption.
    """
    lines = []
    for seg in result.segments:
        mins = int(seg.start // 60)
        secs = int(seg.start % 60)
        lines.append(f"{seg.speaker} [{mins:02d}:{secs:02d}]: {seg.text}")
    return "\n".join(lines)