# Neural Topic Modeling of HPV Vaccine Perception Focus Groups

This repository contains the complete analysis pipeline for applying BERTopic (neural topic modeling) to focus group transcripts exploring HPV vaccine perceptions in Tunisia.

**Paper:** Arfaoui, H., Hergli, M.I., Benzina, B., & BenMiled, S. (2025). A Reproducible Framework for Neural Topic Modeling in Focus Group Analysis. *In Preparation*. 

**Dataset:** Available at [https://doi.org/10.5281/zenodo.17798456](https://doi.org/10.5281/zenodo.17798455)

## Overview

This project demonstrates a systematic framework for applying transformer-based topic modeling to small, conversational qualitative datasets. The pipeline includes:

- Sentence embedding generation using multilingual transformers
- Comprehensive hyperparameter grid search (27 configurations)
- Bootstrap stability analysis (NMI, ARI)
- Multi-criteria model selection (coherence, stability, interpretability)
- Comparison with LDA baseline
- Human validation by domain experts

**Key Finding:** BERTopic achieved 18% higher coherence than optimized LDA (0.573 vs 0.486) with 7 interpretable topics validated by independent experts (ICC = 0.700).

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
│   ├── settings.py              # Configuration
│   ├── encode_embeddings.py     # Generate embeddings
│   ├── grid_search.py           # Hyperparameter search
│   ├── fit_final_model.py       # Fit selected model
└── results/
    ├── final_model_*/           # Final model artifacts
    ├── selection/               # Model selection results
    └── stability/               # Grid search metrics
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

## Reproducibility

### Random Seed

All random processes use `SEED=42` (set in `scripts/settings.py`) for reproducibility.

### System Requirements

- **Python:** 3.8+
- **RAM:** 8GB minimum (16GB recommended for grid search)
- **GPU:** Optional (speeds up embedding generation)
- **Time:** ~1-2 hours for full pipeline on CPU

## Key Configuration

The final model uses:
- **Embedding model:** `paraphrase-multilingual-mpnet-base-v2`
- **UMAP parameters:** `n_neighbors=15`, `n_components=10`
- **HDBSCAN parameters:** `min_cluster_size=15`
- **Number of topics:** 7

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
     - Compute coherence (c_v)
     - Perform bootstrap stability analysis (100 samples)
     - Calculate ARI and NMI
   - Save all metrics

3. **Model Selection** (`fit_final_model.py`)
   - Load grid search results
   - Apply selection criteria:
     - Remove collapsed models (≤3 topics)
     - Remove unstable models
     - Select highest coherence
   - Fit final model
   - Extract topics and assignments

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
