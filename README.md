
# Legal Intelligence System

## Chunking configuration

Generic chunks use a fixed token window and configured word overlap. Structure-aware chunks prioritize legal section and paragraph coherence, so `structure_aware_overlap` defaults to `0`. When configured above zero, overlap is added only as complete preceding paragraphs whose source offsets remain exact; no partial or synthetic text is introduced.

## Files excluded from Git

The following files are larger than 100 MB and are excluded in `.gitignore`:

```text
data/chunks.zip
data/chunks/generic_chunks.jsonl
data/chunks/structure_aware_chunks.jsonl
data/embeddings/generic_embeddings.npy
data/embeddings/generic_mapping.json
data/embeddings/structure_aware_embeddings.npy
data/embeddings/structure_aware_mapping.json
data/faiss/generic.index
data/faiss/structure_aware.index
data/processed/judgments_processed.jsonl
data/processed/judgments_vocab.jsonl
data/raw/laww_dataset.jsonl
data/raw/sampled_judgments.jsonl
```
