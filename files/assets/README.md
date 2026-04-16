# Music Assets

Put your local instrumental tracks in:

`assets/music/`

Then list them in:

`assets/music_library.json`

Example entry:

```json
[
  {
    "id": "track_01",
    "title": "Warm Sunset",
    "file": "warm_sunset.mp3",
    "duration_sec": 20,
    "genre": "ambient",
    "mood": ["calm", "dreamy"],
    "tags": ["sunset", "warm", "soft", "cinematic"]
  }
]
```

Notes:
- Use local files only.
- Keep tracks instrumental if you want no lyrics.
- `file` must exactly match the filename inside `assets/music/`.
- The rule-based selector uses `mood`, `genre`, and `tags`.
