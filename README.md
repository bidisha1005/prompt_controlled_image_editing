# RAG Image Editor
**Prompt-Decomposed Retrieval-Augmented Image Editing using InstructPix2Pix**

> Semester project implementation — covers RAG, multimodal LLMs, prompt engineering, and diffusion-based generation.

---

## Architecture

```
User Image + Prompt
        │
        ▼
Prompt Decomposition          ← Novelty 1 (prompt_decomposer.py)
        │
        ▼
CLIP Embedding (per sub-task)
        │
        ▼
FAISS Vector Retrieval        ← MagicBrush RAG index
        │
        ▼
Prompt Construction + State Memory Injection  ← Novelty 2
        │
        ▼
InstructPix2Pix (timbrooks/instruct-pix2pix)
        │
        ▼
Edit State Update + Metrics
        │
        ▼
Final Edited Image
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you want Groq-based prompt reconstruction, set:

```bash
export GROQ_API_KEY=your_key_here
export GROQ_MODEL=llama-3.1-8b-instant
```

> GPU strongly recommended. On CPU, inference takes ~3–5 min per image.

### 2. Build RAG index (one-time, ~3 min)

```bash
python -m rag_pipeline.build_index
```

This downloads the MagicBrush dataset from Hugging Face and builds a FAISS index (~1200 samples).

### 3. Run

```bash
python main.py
```

This starts the FastAPI backend on **http://localhost:8000** and opens the UI automatically.

Alternatively open `frontend/index.html` directly in your browser after the server is running.

---

## API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | /health     | Server health check                  |
| POST   | /decompose  | Preview prompt decomposition         |
| POST   | /edit       | Full RAG-guided edit pipeline        |
| POST   | /reset      | Reset edit state memory              |
| GET    | /state      | Current edit state                   |

### /edit request body

```json
{
  "image_b64": "<base64 image>",
  "instruction": "Remove the car and make the sky sunset",
  "image_guidance_scale": 1.5,
  "text_guidance_scale": 7.5,
  "num_steps": 30,
  "use_rag": true
}
```

---

## Evaluation

```python
from utils.metrics import CLIPEvaluator, evaluate_edit, run_ablation
from PIL import Image

ev = CLIPEvaluator()

original   = Image.open("test_input.jpg")
edited_rag = Image.open("edit_with_rag.jpg")
edited_base= Image.open("edit_no_rag.jpg")

# Ablation: RAG vs no-RAG
results = run_ablation(original, edited_base, edited_rag,
                       instruction="Remove the car")
print(results)
```

---

## Novelties

### Novelty 1 — Prompt-Decomposed RAG Retrieval
Complex prompts are split into atomic sub-tasks before retrieval:
- `"Remove the car and make the sky sunset, keep mountains"`
- → `["Remove the car", "make the sky sunset"]` + preserve: `["mountains"]`
- FAISS retrieval runs **per sub-task**, improving reference quality

### Novelty 2 — Edit State Memory with Constraint Propagation
After each edit step, the system tracks:
- Objects removed / added
- Elements explicitly preserved
- Applied edit history

Subsequent prompts are automatically enriched with these constraints to prevent diffusion drift in multi-step editing.

---

## Files

```
rag_pipeline/
  build_index.py        ← FAISS index builder from MagicBrush
  prompt_decomposer.py  ← Decomposition + state memory
  index/                ← Generated FAISS index (after first run)

backend/
  editor.py             ← InstructPix2Pix wrapper
  server.py             ← FastAPI endpoints

frontend/
  index.html            ← Complete single-file UI

utils/
  metrics.py            ← CLIP similarity, ablation evaluation

main.py                 ← Entry point
requirements.txt
```

---

## Baselines

| Baseline | `use_rag` | Decomposition |
|----------|-----------|---------------|
| Vanilla InstructPix2Pix | `false` | `false` |
| LLM + Diffusion (no RAG) | `false` | `true` |
| **Proposed** | `true` | `true` |

Toggle `use_rag` in the UI or API to run ablation comparisons.
