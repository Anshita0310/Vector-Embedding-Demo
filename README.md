# Vector Embedding Demo

A local semantic intent classification system built with sentence embeddings and a vector database. Given a natural language query, the system finds the most similar banking intent from the **PolyAI/Banking77** dataset using cosine similarity, scores confidence, and routes the query to the appropriate workflow — all without keyword matching or traditional classifiers.

Exposed as a REST API and tested via Postman.

---

## Architecture

```
HuggingFace Dataset (PolyAI/Banking77)
            ↓
Sentence Transformer (all-MiniLM-L6-v2)
            ↓
  Vector Embeddings (384-d, normalized)
            ↓
  Qdrant Vector Database (Cosine Similarity)
            ↓
        User Query
            ↓
   Query Embedding (384-d)
            ↓
  Top-K Semantic Search (k = 3)
            ↓
       Confidence Scoring
            ↓
   Workflow Decision Engine
```

---

## How It Works

### 1. Dataset

The project uses the **PolyAI/Banking77** dataset — 77 intents covering common banking queries, with 10–13k total utterances (roughly 10–13 per intent). Each record contains a `text` (e.g. `"How do I change my card pin?"`) and a `label` (integer, mapped to an `intent_name`). Every utterance is stored as a separate vector in Qdrant, tagged with `intent_label`, `intent_name`, and `utterance_text`.

### 2. Embeddings

The `all-MiniLM-L6-v2` sentence transformer was chosen for its balance of speed and accuracy. All embeddings are L2-normalized before storage so cosine similarity values fall cleanly in the 0–1 range.

### 3. Semantic Search

When a user query arrives, it is embedded using the same model and a top-3 nearest-neighbour search is run against Qdrant. Each result returns a `similarity_score` (cosine similarity) and a payload containing the matched intent.

### 4. Confidence Scoring

Returning only the top-1 result would be naïve — a high similarity score doesn't always mean a clear winner. The confidence score accounts for both:

- **Absolute similarity** — how close the best match is
- **Relative gap** — how far ahead the top match is from the second-best

If all top-k results share the same intent label (intra-intent agreement), confidence is boosted using `min/max(0.95, top_score)`, pushing the score toward 95–100%. This captures cases where the system is clearly right but the raw scores didn't quite reflect it.

> **Key distinction:** Cosine similarity measures semantic closeness. Confidence measures intent separability. A high similarity score with a low gap between top results means the system is uncertain which intent the user meant — and should not auto-execute.

### 5. Workflow Routing

Based on the confidence score, the system routes to one of three outcomes:

| Confidence | Action |
|---|---|
| ≥ 0.75 | Execute the matched workflow automatically |
| 0.50 – 0.74 | Ask the user for clarification |
| < 0.50 | Fall back to a human agent |

---

## Why Semantic Search Over Traditional Classifiers

Traditional approaches rely on keyword matching or regex, which fail on paraphrases and unseen phrasing. This system uses cosine similarity over dense vector embeddings, which:

- Matches on **meaning** (direction of vector), not exact words
- Generalises to **unseen phrasings** without retraining
- Avoids brittle regex rules that need manual maintenance

---

## Project Structure

```
Vector-Embedding-Demo/
├── app/              # FastAPI application and route handlers
├── qdrant_data/      # Local Qdrant storage
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- Docker (for running Qdrant locally)

### Install dependencies

```bash
git clone https://github.com/Anshita0310/Vector-Embedding-Demo.git
cd Vector-Embedding-Demo

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use the included `qdrant_data/` folder to run Qdrant with persistent local storage.

### Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## Dependencies

```
sentence-transformers
datasets
qdrant-client
fastapi
uvicorn
numpy
scikit-learn
torch
```

---

## License

This project does not currently specify a license. All rights reserved by the author.
