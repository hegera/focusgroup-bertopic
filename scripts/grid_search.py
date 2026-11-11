import os
# Fix parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid

from metrics import bertopic_topic_words, keyword_vectorizer, quality_metrics, tokenize
from settings import N_JOBS, NUM_BOOTSTRAP_RUNS, PARAM_GRID, SEED, SUBSAMPLE_FRACTION


def build_model(params):
    """BERTopic model for one hyperparameter configuration (fresh UMAP / HDBSCAN objects)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=params["umap__n_neighbors"],
        n_components=params["umap__n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=SEED
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params["hdbscan__min_cluster_size"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=keyword_vectorizer(),  # stop words removed from topic keywords only
        language="french",
        calculate_probabilities=False
    )


def evaluate_configuration(params, docs, embeddings):
    """Full-data fit, quality metrics and bootstrap stability for one configuration."""
    # Fit on full dataset
    full_model = build_model(params)
    full_topics, _ = full_model.fit_transform(docs, embeddings)

    # Compute metrics for full dataset
    topic_info = full_model.get_topic_info()
    outlier_count = topic_info.query("Topic == -1")["Count"].sum() if -1 in topic_info.Topic.values else 0
    outlier_fraction = outlier_count / len(docs)
    num_topics = len(topic_info) - (1 if -1 in topic_info.Topic.values else 0)
    non_outlier = topic_info[topic_info.Topic != -1]
    main_topic_fraction = non_outlier["Count"].max() / len(docs) if len(non_outlier) else 0  # share of ALL documents
    quality = quality_metrics(bertopic_topic_words(full_model), tokenize(docs))

    # Bootstrap replicates
    bootstrap_topic_assignments = []
    for i in range(NUM_BOOTSTRAP_RUNS):
        np.random.seed(SEED + i)  # Ensure reproducible but different subsamples
        sample_indices = np.random.choice(len(docs), size=int(len(docs) * SUBSAMPLE_FRACTION), replace=True)
        docs_b = [docs[j] for j in sample_indices]
        embeddings_b = embeddings[sample_indices]

        # Re-instantiate models for each bootstrap replicate
        bootstrap_model = build_model(params)
        bootstrap_topics, _ = bootstrap_model.fit_transform(docs_b, embeddings_b)

        # Store assignments with original indices
        assignments = {int(idx): int(topic) for idx, topic in zip(sample_indices, bootstrap_topics)}
        bootstrap_topic_assignments.append(assignments)

    # Compute pairwise ARI and NMI
    ari_scores = []
    nmi_scores = []
    for i, j in combinations(range(NUM_BOOTSTRAP_RUNS), 2):
        assignments1 = bootstrap_topic_assignments[i]
        assignments2 = bootstrap_topic_assignments[j]

        # Find common documents
        common_docs = set(assignments1.keys()) & set(assignments2.keys())
        if len(common_docs) > 1:
            labels1 = [assignments1[doc_id] for doc_id in common_docs]
            labels2 = [assignments2[doc_id] for doc_id in common_docs]
            ari_scores.append(adjusted_rand_score(labels1, labels2))
            nmi_scores.append(normalized_mutual_info_score(labels1, labels2))

    return {
        "params": params,
        "coherence_score": quality["c_v"],
        "c_npmi": quality["c_npmi"],
        "c_uci": quality["c_uci"],
        "topic_diversity": quality["topic_diversity"],
        "median_ari": float(np.median(ari_scores)) if ari_scores else 0,
        "median_nmi": float(np.median(nmi_scores)) if nmi_scores else 0,
        "num_topics": int(num_topics),
        "outlier_fraction": float(outlier_fraction),
        "main_topic_fraction": float(main_topic_fraction),
    }


def bootstrap_stability():
    """Perform bootstrap stability analysis for BERTopic with a hyperparameter grid."""
    print("Loading data...")
    embeddings = np.load("../embeddings/embeddings.npy")
    answers_df = pd.read_csv("../data/data.csv")
    docs = answers_df['sentence'].tolist()

    param_grid_list = list(ParameterGrid(PARAM_GRID))
    print(f"Evaluating {len(param_grid_list)} configurations "
          f"({NUM_BOOTSTRAP_RUNS} bootstrap replicates each, {N_JOBS} parallel jobs)...")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(evaluate_configuration)(params, docs, embeddings) for params in param_grid_list
    )

    # Save results
    os.makedirs("../results/stability", exist_ok=True)
    with open("../results/stability/metrics.json", 'w') as f:
        json.dump(results, f, indent=4)

    print("Bootstrap stability analysis complete. Results saved to ../results/stability/metrics.json")

    # Create visualization
    print("\nCreating grid search metrics visualization...")
    create_grid_search_visualization(results)

    print("\n✓ Review the generated figures in ../results/Figures/")
    print("✓ Use these results to manually select your preferred configuration in the notebook.")


def results_to_dataframe(results):
    df_list = []
    for item in results:
        row = item['params'].copy()
        row.update({k: v for k, v in item.items() if k != 'params'})
        df_list.append(row)
    df = pd.DataFrame(df_list)
    return df.rename(columns={
        'hdbscan__min_cluster_size': 'min_cluster_size',
        'umap__n_components': 'n_components',
        'umap__n_neighbors': 'n_neighbors'
    })


def plot_metric_grid(df, metrics, output_path):
    """Heatmaps of the given metrics across the grid, one column per min_cluster_size."""
    n_neighbors_vals = sorted(df['n_neighbors'].unique())
    n_components_vals = sorted(df['n_components'].unique())
    min_cluster_vals = sorted(df['min_cluster_size'].unique())

    fig, axes = plt.subplots(len(metrics), len(min_cluster_vals),
                             figsize=(4 * len(min_cluster_vals), 3 * len(metrics)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    if len(min_cluster_vals) == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (metric, title, cmap, fmt) in enumerate(metrics):
        for col_idx, min_cluster in enumerate(min_cluster_vals):
            subset = df[df['min_cluster_size'] == min_cluster]
            heatmap_data = subset.pivot(index='n_neighbors', columns='n_components', values=metric)
            heatmap_data = heatmap_data.reindex(index=n_neighbors_vals, columns=n_components_vals)

            sns.heatmap(
                heatmap_data,
                ax=axes[row_idx, col_idx],
                annot=True,
                fmt=fmt,
                vmin=df[metric].min(),
                vmax=df[metric].max(),
                cbar=col_idx == len(min_cluster_vals)-1,
                cmap=cmap,
                annot_kws={"fontsize": 11}
            )

            axes[row_idx, col_idx].set_title(f'{title}\nmin_cluster_size = {min_cluster}',
                                             fontsize=15, fontweight='bold')
            axes[row_idx, col_idx].set_xlabel('n_components', fontsize=13)
            axes[row_idx, col_idx].set_ylabel('n_neighbors', fontsize=13)
            axes[row_idx, col_idx].tick_params(axis='both', labelsize=11)

            if col_idx != 0:
                axes[row_idx, col_idx].set_ylabel('')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid search visualization to: {output_path}")
    plt.close()


def create_grid_search_visualization(results):
    """Create heatmap visualizations of grid search metrics."""
    df = results_to_dataframe(results)
    os.makedirs("../results/Figures", exist_ok=True)

    main_metrics = [
        ('median_nmi', 'NMI', sns.light_palette("seagreen", as_cmap=True), '.3f'),
        ('median_ari', 'ARI', sns.light_palette("lightblue", as_cmap=True), '.3f'),
        ('coherence_score', 'Coherence (C_V)', 'viridis', '.3f'),
        ('num_topics', 'Num Topics', 'mako', 'd'),
        ('outlier_fraction', 'Outlier %', 'rocket', '.3f'),
        ('main_topic_fraction', 'Main Topic %', 'crest', '.3f'),
    ]
    plot_metric_grid(df, main_metrics, '../results/Figures/gridsearch_metrics.png')

    additional_metrics = [
        ('c_npmi', 'C_NPMI', 'viridis', '.3f'),
        ('c_uci', 'C_UCI', 'viridis', '.2f'),
        ('topic_diversity', 'Topic Diversity', 'crest', '.3f'),
    ]
    plot_metric_grid(df, additional_metrics, '../results/Figures/gridsearch_additional_metrics.png')


if __name__ == "__main__":
    bootstrap_stability()
