"""
Rule-based music selection for optional background audio.

This module reads a small local metadata file and picks one track using
simple tag overlap against the user's prompt.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
LIBRARY_PATH = ASSETS_DIR / "music_library.json"

TOKEN_RE = re.compile(r"[a-z0-9]+")

KEYWORD_TAGS = {
    "sunset": ["sunset", "warm", "dreamy", "cinematic"],
    "night": ["night", "dark", "ambient", "moody"],
    "snow": ["snow", "calm", "soft", "winter"],
    "winter": ["winter", "calm", "soft"],
    "beach": ["beach", "summer", "bright", "chill"],
    "ocean": ["ocean", "calm", "ambient", "nature"],
    "sky": ["airy", "ambient", "cinematic"],
    "flower": ["gentle", "bright", "nature"],
    "forest": ["nature", "calm", "organic"],
    "mountain": ["cinematic", "wide", "nature"],
    "city": ["urban", "modern", "energetic"],
    "car": ["urban", "driving", "modern"],
    "neon": ["synth", "night", "urban"],
    "blue": ["cool", "calm", "clean"],
    "red": ["bold", "dramatic", "intense"],
    "gold": ["warm", "uplifting", "cinematic"],
    "dark": ["dark", "moody", "ambient"],
    "bright": ["bright", "uplifting", "light"],
    "happy": ["happy", "uplifting", "playful"],
    "playful": ["playful", "light", "fun"],
    "calm": ["calm", "soft", "ambient"],
    "peaceful": ["calm", "soft", "gentle"],
    "dreamy": ["dreamy", "ambient", "soft"],
    "epic": ["epic", "cinematic", "dramatic"],
    "cinematic": ["cinematic", "dramatic", "wide"],
    "dramatic": ["dramatic", "intense", "cinematic"],
    "sad": ["sad", "piano", "slow"],
    "romantic": ["romantic", "soft", "warm"],
    "magic": ["dreamy", "cinematic", "soft"],
}


def _load_library() -> List[Dict]:
    if not LIBRARY_PATH.exists():
        logger.info("Music library metadata not found at %s", LIBRARY_PATH)
        return []

    try:
        with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read music library metadata")
        return []

    if not isinstance(data, list):
        logger.warning("music_library.json should contain a list of tracks")
        return []

    tracks: List[Dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        file_name = entry.get("file")
        if not file_name:
            continue
        file_path = MUSIC_DIR / file_name
        if not file_path.exists():
            logger.warning("Skipping missing music file: %s", file_path)
            continue
        tracks.append(entry)
    return tracks


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _derive_prompt_tags(prompt: str) -> List[str]:
    tokens = _tokenize(prompt)
    tags = set(tokens)
    for token in tokens:
        tags.update(KEYWORD_TAGS.get(token, []))
    return sorted(tags)


def _track_tags(track: Dict) -> set[str]:
    values: List[str] = []
    for key in ("title", "genre"):
        value = track.get(key)
        if isinstance(value, str):
            values.append(value)

    for key in ("mood", "tags"):
        value = track.get(key, [])
        if isinstance(value, list):
            values.extend(str(item) for item in value)

    tags = set()
    for value in values:
        tags.update(_tokenize(value))
    return tags


def get_library_count() -> int:
    return len(_load_library())


def select_track(prompt: str) -> Optional[Dict]:
    tracks = _load_library()
    if not tracks:
        return None

    prompt_tags = set(_derive_prompt_tags(prompt))
    best_track: Optional[Dict] = None
    best_score = -1
    best_overlap: List[str] = []

    for idx, track in enumerate(tracks):
        tags = _track_tags(track)
        overlap = sorted(prompt_tags & tags)
        score = len(overlap)

        # Prefer slightly shorter tracks when there is a tie.
        duration = int(track.get("duration_sec", 999))
        tie_breaker = 1 if duration <= 20 else 0
        final_score = (score * 10) + tie_breaker

        if final_score > best_score:
            best_score = final_score
            best_track = track
            best_overlap = overlap
        elif final_score == best_score and best_track is None:
            best_track = track
            best_overlap = overlap

    if best_track is None:
        return None

    reason = (
        f"Matched tags: {', '.join(best_overlap[:4])}"
        if best_overlap
        else "Fallback pick from curated instrumental library"
    )

    return {
        "id": best_track.get("id", best_track.get("file")),
        "title": best_track.get("title", best_track.get("file")),
        "file": best_track.get("file"),
        "file_url": f"/assets/music/{best_track.get('file')}",
        "duration_sec": best_track.get("duration_sec"),
        "mood": best_track.get("mood", []),
        "genre": best_track.get("genre"),
        "tags": best_track.get("tags", []),
        "reason": reason,
    }
