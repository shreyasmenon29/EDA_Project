# RAG Design Audit

## A. Current project state

The project is an existing academic research workflow centered on Indian Supreme Court judgments and a legal information retrieval experiment. The corpus and EDA are already complete and must remain intact:

- Raw corpus: `data/raw/laww_dataset.jsonl`
- Research sample: `data/raw/sampled_judgments.jsonl`
- Processed corpus: `data/processed/judgments_processed.jsonl`
- Sample size: 12,000 judgments
- Sampling procedure: proportional by decade, seed = 42
- Preprocessing already preserves the original legal text and paragraph structure while cleaning noise
- EDA scripts and outputs are present and must not be deleted or rewritten for retrieval work

The project’s central research question is not “build a chatbot” in the abstract, but rather:

> Does structure-aware chunking improve retrieval relative to generic chunking under otherwise identical conditions?

This means the retrieval pipeline should compare two chunking strategies using the same corpus, questions, embedding model, similarity metric, and evaluation procedure.

---

## B. Available metadata

The processed dataset contains the following metadata keys (observed from the real sample):

- `Judgement_type`
- `articles_cited`
- `bench`
- `case_no`
- `citation_extracted`
- `clean_text`
- `csv_id`
- `diary_no`
- `full_text`
- `has_citation`
- `is_short_order`
- `judgement_by`
- `judgment_dates`
- `language`
- `outcome`
- `pet`
- `pet_adv`
- `res`
- `res_adv`
- `sections_cited`
- `status`
- `temp_link`
- `text_length`

Key observations:

- `full_text` remains available and must not be discarded.
- `clean_text` is the version used for analysis and should be the conservative retrieval/text input, not a rewritten legal abstraction.
- `judgment_dates` is present in `DD-MM-YYYY` format and is usable for temporal stratification.
- `sections_cited` and `articles_cited` are structured reference fields and are the appropriate source for legal references.
- `citation_extracted` is reported citation metadata, not a clean list of authorities cited in the body of the judgment.
- Party names and bench metadata are available for display and filtering, but they should not silently be inserted into legal reasoning text unless there is an explicit, justified design choice.

---

## C. Available structural signals

The EDA scripts explicitly detect and measure several legal-format cues in `clean_text`. These have been used to establish temporal structural drift and are the strongest candidates for structure-aware chunking.

Observed signals from the project:

- `CORAM`
- `JUDGMENT`
- `ORDER`
- `BENCH`
- `PETITIONER`
- `RESPONDENT`
- `APPELLANT`
- `FACTS`
- `ISSUES`
- `HELD`
- `CONCLUSION`
- `REASONS`
- `JUDGMENT BY`
- case-number patterns
- numbered paragraphs such as `1.`, `2.`, `3)`
- page markers such as `Page 1 of 19`
- supreme court header markers

The EDA shows a strong temporal pattern in structural density:

- paragraph density: Spearman rho ≈ +0.7866
- numbered paragraphs density: ≈ +0.5510
- page markers density: ≈ -0.6310

This is strong evidence that structure is not uniform across decades, and therefore structure-aware chunking is a credible research variable.

---

## D. Historical structural differences

Actual corpus examples confirm that historical judgments do not share a single unified modern structure.

### 1950s example

A 1950 case contains:

- `SUPREME COURT OF INDIA`
- `Page 1 of 19`
- `PETITIONER` / `RESPONDENT`
- `DATE OF JUDGMENT`
- `BENCH`
- legal citations and a procedural/acts header
- paragraph count around 19
- long, dense text with 11,056 words

This shows a strongly formalized older layout with page markers and header-style metadata.

### 1970s example

A 1970 case contains:

- `HEADNOTE`
- `CITATOR INFO`
- `ACT:`
- `BENCH`
- repetitive header metadata
- paragraph count around 6
- text remains long but less heavily segmented than modern judgments

### 1990s example

A 1990 case contains:

- `BENCH`
- `HEADNOTE`
- procedural cues and legal references
- a less regular modern argument/reasoning split

### 2010s example

A 2010 case contains:

- `ORDER`
- `CIVIL APPELLATE JURISDICTION`
- `CIVIL APPEAL NO...`
- `Leave granted.`
- numbered paragraphs like `2.`
- more modern case-procedure syntax and shorter average lengths

### 2020s example

A 2020 case contains:

- `IN THE SUPREME COURT OF INDIA`
- `CRIMINAL APPELLATE JURISDICTION`
- `JUDGMENT`
- numbered paragraphs
- blocks starting with `1.` and subsequent legal reasoning

### Core conclusion

The corpus shows a real drift from older page/metadata-heavy layouts toward modern, numbered-paragraph-heavy and issue-oriented formatting. However, historical markers are not consistent enough to assume a universal legal segmentation rule. Structure-aware chunking must be robust to heterogeneous headers, older page-marked formats, and procedural orders.

---

## E. Recommended generic chunking definition

Recommended generic definition:

- Use the cleaned text as the primary input.
- Split by paragraph boundaries where possible.
- Merge paragraphs into fixed-size chunks of roughly 500–800 tokens or about 2,000–3,000 characters, whichever is more practical for the embedding model chosen.
- Use modest overlap of approximately 50–100 tokens or 150–250 characters.
- Do not encode legal metadata or structural labels into chunk content.
- Preserve document metadata separately for citation and filtering.

This is the cleanest control condition because it does not exploit legal format cues and therefore isolates the effect of chunking strategy.

Important principle:

- Generic chunking should not be “just paragraph chunks of variable length”; it should be a reproducible fixed-policy baseline that is easy to compare against the structure-aware condition.

---

## F. Recommended structure-aware chunking definition

Recommended structure-aware definition:

- Start from the same `clean_text` and preserve the same embedding model, similarity metric, and retrieval method.
- Detect blocks using cues such as:
  - `JUDGMENT`
  - `ORDER`
  - `HEADNOTE`
  - numbered paragraphs
  - legal-section boundaries
  - recurring header clusters such as `PETITIONER`, `RESPONDENT`, `BENCH`
- Prefer chunk boundaries at section/heading transitions and at numbered-paragraph boundaries.
- Keep chunks semantically coherent: one issue or one legal reasoning block rather than arbitrary 800-token windows.
- If no robust structure markers are present, fall back to paragraph-aware chunking instead of strict forced splits.

This should be expressed as a document segmentation policy rather than a hard-coded assumption that every modern judgment follows a single template.

The research design should compare:

1. Generic paragraph/fixed-window chunks
2. Structure-aware section/paragraph-aware chunks

with everything else held constant.

---

## G. Potential failure cases

The corpus has several issues that could distort retrieval if not handled carefully:

- Old judgments contain repeated header metadata and page markers, so page numbers or “court name” lines may look like real legal content.
- Some records include OCR-like artifacts and hyphenation across line breaks (for example, “exploi-\nsion”).
- Modern judgments often use numbered paragraphs, but older judgments may not.
- Short procedural orders may contain fragments that are meaningful but very short; they must not be dropped just because they are statistically unusual.
- `citation_extracted` can be mistaken for a usable authority list; it is not a clean legal citation graph and should not be treated as such.
- Structure-aware chunking may overfit to specific formatting conventions if implemented too rigidly.
- Some records show `case_no` values like `- 0` or metadata noise; these should be treated as metadata artifacts, not legal reasoning.
- A chunking rule that depends heavily on `JUDGMENT` or `ORDER` may fail on older judgments where those markers appear less consistently.

---

## H. Required evaluation data

No fake evaluation should be created.

The retrieval experiment requires a benchmark with the following characteristics:

- same questions for both chunking conditions
- same documents for both conditions
- same retrieval pipeline for both conditions
- explicit question IDs and metadata
- relevance labels created by human annotation or a documented annotation protocol

Recommended benchmark structure:

- `data/evaluation/questions.jsonl`
- fields such as:
  - `question_id`
  - `question`
  - `legal_topic`
  - `year_range`
  - `relevance_judgments` (human-annotated, if available)
  - `notes`
- separate file for judged passages or judged documents if done at document level

Important: the benchmark design may include `REQUIRES HUMAN ANNOTATION` explicitly for any manual relevance judgments. Do not fill in fake labels.

---

## I. Recommended embedding approach for CPU

Use a compact sentence embedding model suitable for CPU-only environment, with reproducible settings.

Recommended choice:

- `sentence-transformers/all-MiniLM-L6-v2`

Why this is appropriate:

- CPU-friendly
- widely used and academically defensible
- small enough for 12,000-chunk retrieval pipelines
- good semantic retrieval quality for legal text without requiring a paid API

Alternative if needed:

- `BAAI/bge-small-en-v1.5` (slightly stronger semantic retrieval but still manageable on CPU)

The choice should be fixed before the benchmark comparison and recorded in the experiment config.

---

## J. Recommended vector index

For this project size and research design, the most defensible default is:

- FAISS `IndexFlatIP` with L2-normalized embeddings

Why:

- simple and reproducible
- exact top-k retrieval under cosine similarity
- easy to compare generic vs structure-aware chunking
- no need for a heavyweight vector database
- works well for the scale here (12,000 judgments / chunked retrieval corpus)

This is preferable to more complex HNSW or hybrid infrastructure unless the benchmark specifically requires it.

---

## K. Dependencies required

Minimum dependencies for the planned retrieval pipeline:

- `pandas`
- `numpy`
- `scikit-learn`
- `sentence-transformers`
- `faiss-cpu` or `faiss`
- `tqdm`
- `pyyaml` (if using config files)
- `python-dateutil` (optional but useful)

Possible later-stage dependencies for the chatbot/dashboard:

- `fastapi` or `flask`
- `streamlit` for dashboard display
- `uvicorn` for local serving

These should only be introduced as needed, not before the retrieval benchmark is built.

---

## L. Risks to research validity

These are the main risks that could compromise the research conclusions:

1. Mixing metadata into the legal text used for retrieval.
2. Ignoring the temporal change in legal formatting and using a single fixed segmentation rule for all decades.
3. Using the wrong legal reference fields (`citation_extracted` instead of `sections_cited` / `articles_cited`).
4. Creating synthetic questions or relevance judgments.
5. Changing the dataset or preprocessing pipeline after the EDA.
6. Comparing different embedding models or retrieval settings between experimental arms.
7. Overfitting chunk boundaries to a subset of modern judgments.
8. Treating long, old, or procedural records as noise without inspection.
9. Using a large language model or paid API without prior approval and without controlling evaluation.
10. Letting the benchmark leak into the index-building process.

---

## M. Files that would be created/modified

This audit does not create the retrieval pipeline yet. The anticipated future files are:

### New files

- `src/rag/__init__.py`
- `src/rag/config.py`
- `src/rag/chunking.py`
- `src/rag/benchmark.py`
- `src/rag/evaluation.py`
- `src/rag/embeddings.py`
- `src/rag/indexing.py`
- `src/rag/retrieval.py`
- `src/rag/rerank.py`
- `src/rag/answering.py`
- `src/rag/citation.py`
- `src/chatbot/app.py`
- `src/dashboard/streamlit_app.py`
- `outputs/reports/checkpoint_01_benchmark_design.md`
- `outputs/reports/checkpoint_02_chunking_validation.md`
- `outputs/reports/checkpoint_03_structure_aware_chunking_validation.md`
- `outputs/reports/checkpoint_08_retrieval_evaluation.md`

### Existing files likely to be referenced or extended

- `scripts/05_preprocess.py` (must remain as-is; only referenced as a preserved source of truth)
- `scripts/10_eda_structure.py`
- `scripts/11_structural_drift.py`
- `scripts/12_structural_statistics.py`
- `README.md` (if project-level documentation is later expanded)

---

## Final recommendation

The most defensible next step is not to build a full chatbot yet, but to formalize the retrieval benchmark and then implement the two chunking strategies under the same retrieval conditions. The evidence already supports a meaningful structure-aware chunking experiment because the corpus exhibits strong temporal drift in paragraph density, numbered paragraph patterns, and page-marker usage, while the total document length itself is non-monotonic.

The project should therefore proceed with a rigorous controlled comparison of:

- generic fixed-window / paragraph-aware chunking
- structure-aware section/numbered-paragraph-aware chunking

using a single CPU-compatible embedding model and a single FAISS index configuration.

This design respects the research objective and the constraints of the existing corpus and EDA.
