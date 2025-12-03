# Embeddings Directory

Pre-computed sentence embeddings are not included in this repository to save space and because they can be regenerated from the data.

## Option 1: Generate Embeddings (Recommended)

After downloading the dataset to `data/data.csv`, generate embeddings:

```bash
python scripts/encode_embeddings.py
```

**Time required:** ~5-15 minutes on CPU, ~2 minutes on GPU

**Output files:**
- `embeddings.npy` - Sentence embeddings array (float32, shape: [1075, 768])
- `index.csv` - Metadata mapping embeddings to original data

## Option 2: Download Pre-computed Embeddings

If you prefer to skip embedding generation, pre-computed embeddings may be available on Zenodo alongside the dataset:

[https://doi.org/10.5281/zenodo.17798456](https://doi.org/10.5281/zenodo.17798456)

Extract files to this directory.

## Expected File Structure

```
embeddings/
├── embeddings.npy    (sentence embeddings, ~6-10 MB)
└── index.csv         (metadata index, ~150 KB)
```

## Technical Details

**Embedding Model:** `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers)
- Dimensions: 768
- Language: Multilingual (optimized for French)
- Pooling: Mean pooling

**Configuration:**
- Model name stored in: `configs/embedding_model.txt`
- Generation script: `scripts/encode_embeddings.py`
- Deterministic: Results are reproducible (no randomness in encoding)

## Troubleshooting

**If embedding generation fails:**

1. Check you have the dataset: `data/data.csv` must exist
2. Verify sentence-transformers is installed: `pip install sentence-transformers`
3. Check available memory: Model requires ~2GB RAM
4. Try CPU if GPU errors occur: Set `device='cpu'` in `encode_embeddings.py`

**If embeddings files are missing during analysis:**

The pipeline will automatically check for embeddings and prompt you to generate them if missing.
