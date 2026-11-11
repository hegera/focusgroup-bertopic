SEED = 42

# Hyperparameter grid (UMAP and HDBSCAN); 3 x 3 x 3 = 27 configurations
PARAM_GRID = {
    "umap__n_neighbors": [10, 15, 30],
    "umap__n_components": [5, 10, 50],
    "hdbscan__min_cluster_size": [5, 10, 15],
}

# Bootstrap stability analysis
NUM_BOOTSTRAP_RUNS = 30
SUBSAMPLE_FRACTION = 0.85

# Topic representation / coherence
TOP_N_WORDS = 10

# Number of configurations evaluated in parallel by grid_search.py (results do not depend on it)
N_JOBS = 4

# LDA baseline grid (None = scikit-learn default prior, 1/K)
LDA_K_RANGE = list(range(3, 16))
LDA_DOC_TOPIC_PRIORS = [None, 0.01, 0.1, 0.5, 1.0]
LDA_TOPIC_WORD_PRIORS = [None, 0.01, 0.1, 0.5]
