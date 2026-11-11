# Neural Topic Modeling of HPV Vaccine Perception Focus Groups

This repository contains the complete analysis pipeline for applying BERTopic (neural topic modeling) to focus group transcripts exploring HPV vaccine perceptions in Tunisia.

**Paper:** Arfaoui, H., Hergli, M.I., Benzina, B., & BenMiled, S. (2025). A Reproducible Framework for Neural Topic Modeling in Focus Group Analysis. *In Preparation*. 

**Dataset:** Available at [https://doi.org/10.5281/zenodo.17798456](https://doi.org/10.5281/zenodo.17798455)

## Overview

This project demonstrates a systematic framework for applying transformer-based topic modeling to small, conversational qualitative datasets. The pipeline includes:

- Sentence embedding generation using multilingual transformers
- Comprehensive hyperparameter grid search (27 configurations)
- Bootstrap stability analysis (NMI, ARI)
- Topic quality metrics on stop-word-filtered keywords (C_V, C_NPMI, C_UCI, Topic Diversity)
- Multi-criteria model selection (coherence, stability, granularity, coverage)
- Comparison with an LDA baseline with tuned Dirichlet priors
- Sensitivity analysis of outlier reassignment
- Human validation by domain experts

**Key Finding:** BERTopic achieved 9.5% higher coherence than an LDA baseline with tuned priors (C_V 0.606 vs 0.553) with 7 topics validated by independent experts (ICC = 0.700).

## Repository Structure

```
NLPHPV_final/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── configs/
│   └── embedding_model.txt      # Selected sentence transformer model
├── data/
│   └── data.csv                 # [Download from Zenodo]
├── embeddings/
│   ├── embeddings.npy           # Pre-computed embeddings [Generated]
│   └── index.csv                # Embedding metadata [Generated]
├── notebooks/
│   └── run_pipeline.ipynb       # Main analysis pipeline
├── scripts/
│   ├── settings.py              # Configuration (seed, grids, bootstrap settings)
│   ├── metrics.py               # Stop-word list, topic keywords, coherence and diversity metrics
│   ├── encode_embeddings.py     # Generate embeddings
│   ├── grid_search.py           # Hyperparameter search with bootstrap stability
│   ├── fit_final_model.py       # Fit selected model
│   ├── lda_baseline.py          # LDA baseline (number of topics and Dirichlet priors)
│   ├── outlier_sensitivity.py   # Outlier reassignment sensitivity analysis
│   └── human_validation.py      # Sampling procedure and inter-annotator agreement
└── results/
    ├── Figures/                 # Grid search heatmaps
    ├── final_model_*/           # Final model artifacts
    ├── selection/               # Selected configuration
    ├── stability/               # Grid search metrics
    ├── lda/                     # LDA baseline results
    ├── outliers/                # Outlier sensitivity results
    └── human_validation/        # Agreement statistics
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/hegera/focusgroup-bertopic.git
cd focusgroup-bertopic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the dataset from Zenodo: [https://doi.org/10.5281/zenodo.17798456](https://doi.org/10.5281/zenodo.17798456)

Place `data.csv` in the `data/` directory.

### 3. Run Analysis Pipeline

```bash
jupyter notebook notebooks/run_pipeline.ipynb
```

The scripts can also be run directly. They use paths relative to `scripts/`, so run them from that directory:

```bash
cd scripts
python encode_embeddings.py
python grid_search.py
python fit_final_model.py
python lda_baseline.py
python outlier_sensitivity.py
python human_validation.py
```

## Reproducibility

### Random Seed

All random processes use `SEED=42` (set in `scripts/settings.py`) for reproducibility. Bootstrap replicate *i* uses seed 42 + *i*. `requirements.txt` pins the package versions used for the reported results; small numerical differences can occur in other environments for configurations with the largest UMAP settings.

### System Requirements

- **Python:** 3.13 (version used for the reported results)
- **RAM:** 8GB minimum (16GB recommended for grid search)
- **GPU:** Optional (speeds up embedding generation)
- **Time:** about 25 minutes for the grid search on CPU (about 6 minutes with 4 parallel jobs); the other steps take a few minutes

## Key Configuration

The final model uses:
- **Embedding model:** `paraphrase-multilingual-mpnet-base-v2`
- **UMAP parameters:** `n_neighbors=15`, `n_components=10`
- **HDBSCAN parameters:** `min_cluster_size=15`
- **Topic keywords:** c-TF-IDF, top 10 words, NLTK French stop words removed (stop words are kept for the embeddings)
- **Number of topics:** 7 (23.3% of utterances are outliers)

See `results/final_model_*/config.json` for complete configuration.


## Project Workflow

1. **Embedding Generation** (`encode_embeddings.py`)
   - Load and clean transcript data
   - Generate sentence embeddings using multilingual transformer
   - Save embeddings and metadata

2. **Grid Search** (`grid_search.py`)
   - Define hyperparameter grid (UMAP + HDBSCAN)
   - For each configuration:
     - Fit BERTopic model
     - Compute coherence (C_V, C_NPMI, C_UCI) and Topic Diversity on stop-word-filtered keywords
     - Perform bootstrap stability analysis (30 replicates, 85% of documents drawn with replacement)
     - Calculate ARI and NMI
   - Save all metrics and heatmaps

3. **Model Selection** (notebook, then `fit_final_model.py`)
   - Review the grid search figures
   - Select a configuration manually, balancing coherence, stability, granularity and coverage
   - Fit final model
   - Extract topics and assignments

4. **LDA Baseline** (`lda_baseline.py`)
   - Grid over the number of topics (3-15) and both Dirichlet priors (260 models)
   - Same stop-word list, tokenisation and coherence procedure as BERTopic

5. **Outlier Sensitivity** (`outlier_sensitivity.py`)
   - Reassign the outliers of the selected model with `reduce_outliers` and recompute the metrics

6. **Human Validation** (`human_validation.py`)
   - Sampling procedure for the expert review and inter-annotator agreement (ICC, weighted kappa)

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{arfaoui2025bertopic,
  title={A Reproducible Framework for Neural Topic Modeling in Focus Group Analysis},
  author={Arfaoui, Heger and Hergli, Mohamed Iheb and Benzina, Beya and BenMiled, Slimane},
  journal={In Preparation},
  year={2025},
  doi={[DOI to be added]}
}
```

For the dataset:
```bibtex
@dataset{arfaoui2025hpvdata,
  title={HPV Vaccine Perception Focus Group Data - Tunisia},
  author={Arfaoui, Heger and Hergli, Mohamed Iheb and Benzina, Beya and BenMiled, Slimane},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17798456}
}
```

## License

This code is released under the MIT License. See `LICENSE` for details.

The dataset is released under CC BY 4.0 - see the [Zenodo repository](https://doi.org/10.5281/zenodo.17798456) for details.

## Funding

This work was supported, in whole or in part, by the Gates Foundation [INV-059607] through the AMAX (African Modeling and Analytics Academy for Women) project, and by the Vaccine Impact Modelling Consortium (VIMC). At the time of analysis, the VIMC was jointly funded by Gavi, the Vaccine Alliance and the Bill & Melinda Gates Foundation (grant numbers INV-034281 and INV-009125/OPP1157270).

The conclusions and opinions expressed in this work are those of the author(s) alone and shall not be attributed to the Foundation.

## Contact

For questions about the methodology or code:
- Heger Arfaoui: heger.arfaoui@enit.utm.tn

For questions about the dataset:
- See the [Zenodo repository](https://doi.org/10.5281/zenodo.17798456)

## Acknowledgments

We thank the focus group participants for sharing their perspectives and experiences.
