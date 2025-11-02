# Customer Support AI — Enfuce Support Agent Assistant

This repository contains a small, end‑to‑end demo for an AI‑assisted customer support workflow:
1) Clean and translate multilingual ticket data, mask PII, and create model‑ready text fields.
2) Build sentence embeddings, index them with FAISS, and surface similar historical tickets.
3) Use a lightweight chat LLM to turn internal resolution notes into a concise suggestion for agents.
4) Visualize everything in a Streamlit UI.

## Repository Structure

```
.
├── agent_demo.py
├── clean_data.py
├── conda
│   └── environment.yml
├── data
│   ├── customer_support_tickets.csv
│   ├── customer_support_tickets---sample.csv
│   ├── customer_support_tickets_sample.csv
│   ├── customer_support_tickets_with_names.csv
│   ├── enfuce_support_tickets_synthetic.jsonl
│   └── enfuce_support_tickets_synthetic_sample.jsonl
├── README.md  ← you are here
└── utils
    ├── EDA.ipynb
    ├── text_cleaner.py
    └── translate.py
```

Notes:
- Paths inside the scripts reference absolute locations under `/home/faiz/...`. Update these paths if your project lives elsewhere.
- The Streamlit app file is `agent_demo.py` (in earlier iterations it was referred to as `support_agent_demo.py`).

## Quick Start

### 1) Create the environment
Use the provided Conda spec or install dependencies manually.

```bash
conda env create -f conda/environment.yml
conda activate customer-support-ai
```

If you install manually, you will need (minimal list): `python>=3.10`, `streamlit`, `pandas`, `numpy`, `sentence-transformers`, `faiss-cpu` or `faiss-gpu`, `scikit-learn`, `transformers`, `torch`, `langdetect`, `tqdm`.

### 2) Prepare and clean data
Clean and translate the raw JSONL tickets into a single CSV that the app reads.

```bash
python clean_data.py
```

This reads `data/enfuce_support_tickets_synthetic.jsonl` and writes a cleaned CSV to:
```
/home/faiz/Documents/github/Customer_Support/data/customer_support_tickets.csv
```
Adjust the input and output paths in `clean_data.py` if needed.

### 3) Run the Streamlit demo
```bash
streamlit run agent_demo.py
```

The app:
- Loads the cleaned CSV.
- Encodes the knowledge base tickets using a multilingual sentence transformer.
- Builds a FAISS index on first run (or loads a cached index from `.faiss_cache/`).
- Lets you select a demo ticket, shows translations and metadata, retrieves similar tickets, and generates a single actionable suggestion from internal notes using a small chat LLM.

## Data Flow

1. **Raw Data** (`*.jsonl`)
   - Each row contains subject, body, internal comments, and ticket metadata.

2. **Cleaning & Translation** (`clean_data.py` → `utils/text_cleaner.py` and `utils/translate.py`)
   - Mask PII: emails, phone numbers, IBANs, card numbers, and names detected via multilingual NER.
   - Remove emojis and simple artifacts.
   - Detect language (Finnish, Swedish, English/other).
   - Translate Finnish and Swedish to English.
   - Construct standardized text fields used by the app:
     - `translated_subject`, `translated_body`, `translated_internal_comments`
     - `customer_query_text` (subject + body in English)
     - `agent_actions_text` (internal comments in English)
     - `full_conversation_text` for experimentation
     - `detected_language` is retained for inspection

3. **Indexing & Retrieval** (`agent_demo.py`)
   - Sentence embeddings computed for `customer_query_text` of knowledge base tickets.
   - FAISS `IndexFlatIP` with normalized embeddings to approximate cosine similarity.
   - The last 1% of rows acts as demo queries; the remaining 99% forms the knowledge base.

4. **Suggestion Generation** (`agent_demo.py`)
   - From top similar tickets, pick the best resolved case and convert its internal notes to an actionable, single‑sentence suggestion using a small chat LLM.

## Models Used and Their Purpose

### Embeddings and Retrieval
- **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
  - Purpose: Generate multilingual sentence embeddings for `customer_query_text`.
  - Scope: Supports Finnish, Swedish, English (and many other languages) well enough for semantic search.
  - Used in: `agent_demo.py`.

### Suggestion Generation (LLM)
- **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**
  - Purpose: Convert internal resolution notes into a concise, actionable, one‑sentence suggestion for agents.
  - Rationale: Small, fast, and fits on modest GPUs; adequate for short controlled generations.
  - Used in: `agent_demo.py`.

### Translation
- **`Helsinki-NLP/opus-mt-fi-en`**
  - Purpose: Finnish to English translation.
  - Used in: `utils/text_cleaner.py`.
- **`Helsinki-NLP/opus-mt-sv-en`**
  - Purpose: Swedish to English translation.
  - Used in: `utils/text_cleaner.py`.

### Named Entity Recognition (for masking names)
- **`Babelscape/wikineural-multilingual-ner`**
  - Purpose: Multilingual NER to detect person names for replacement with `[NAME]`.
  - Used in: `utils/text_cleaner.py`.

### Other Libraries and Components
- **FAISS**: Vector index and similarity search (`IndexFlatIP` with L2‑normalized vectors acts like cosine similarity).
- **langdetect**: Lightweight language detection to choose whether to translate content.
- **scikit‑learn**: Auxiliary utilities; demo imports `cosine_similarity` though FAISS is the primary retrieval backend.
- **Streamlit**: UI for demo interaction and visualization.
- **PyTorch and Transformers**: Model inference across translation, NER, and LLMs.
- **pandas / numpy**: Data processing.

## Caching and Performance

- FAISS index and embeddings are cached to `.faiss_cache/`:
  - `kb.index` for the FAISS index, `kb_embeddings.npy` for embeddings, and `kb_meta.npy` for row ids.
- On app start, the code tries to load the cache if the dataset size matches. Otherwise, it rebuilds the index.
- The LLM in `agent_demo.py` uses a compact model for low latency. It generates only the new tokens and avoids prompt echoing.

GPU considerations:
- `TinyLlama-1.1B-Chat-v1.0` typically fits into 4–6 GB of VRAM in `float16`.
- Translation models can be heavier. If you run out of VRAM, switch to CPU for translation or reduce batch sizes in `utils/translate.py`.
- `TextCleaner` chooses `cuda` when available; adjust device selection logic if your system differs.

## Running With Your Own Data

1. Place your raw tickets in `data/` as JSONL with fields like `ticket_id`, `subject`, `body`, and `internal_comments`.
2. Update paths inside `clean_data.py` if needed.
3. Run `python clean_data.py` to produce a cleaned CSV.
4. Run `streamlit run agent_demo.py` and use the sidebar to explore demo tickets and similar historical cases.

## Troubleshooting

- Index size mismatch
  - Delete `.faiss_cache/*` and re‑run the app so it rebuilds the index.

