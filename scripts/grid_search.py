import os
# Fix parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import json
from itertools import combinations
from tqdm import tqdm
from settings import SEED, PARAM_GRID
import matplotlib.pyplot as plt
import seaborn as sns

def compute_coherence(topic_model, docs):
    """Compute coherence score for a fitted BERTopic model."""
    try:
        # Prepare data for coherence calculation
        docs_tokenized = [doc.lower().split() for doc in docs]
        dictionary = Dictionary(docs_tokenized)
        
        # Get topic words (skip outlier topic -1)
        topics_words = []
        for topic_id in topic_model.get_topics().keys():
            if topic_id != -1:  # Skip outlier topic
                topic_words = [word for word, score in topic_model.get_topic(topic_id)][:10]
                topics_words.append(topic_words)
        
        if not topics_words:
            return 0.0
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=docs_tokenized,
            dictionary=dictionary,
            coherence='c_v',
            processes=1  # Force single process
        )
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"Warning: Coherence calculation failed: {e}")
        return 0.0

def bootstrap_stability():
    """Perform bootstrap stability analysis for BERTopic with a hyperparameter grid."""
    print("Loading data...")
    embeddings = np.load("../embeddings/embeddings.npy")
    answers_df = pd.read_csv("../data/data.csv")
    docs = answers_df['sentence'].tolist()

    num_bootstrap_runs = 30
    subsample_fraction = 0.85
    
    results = []
    param_grid_list = list(ParameterGrid(PARAM_GRID))

    for params in tqdm(param_grid_list, desc="Grid Search"):
        print(f"Testing parameters: {params}")

        # Fit on full dataset
        umap_full = UMAP(
            n_neighbors=params["umap__n_neighbors"],
            n_components=params["umap__n_components"],
            min_dist=0.0,
            metric="cosine",
            random_state=SEED
        )
        hdbscan_full = HDBSCAN(
            min_cluster_size=params["hdbscan__min_cluster_size"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        full_model = BERTopic(
            umap_model=umap_full,
            hdbscan_model=hdbscan_full,
            language="french",
            calculate_probabilities=False
        )
        full_topics, _ = full_model.fit_transform(docs, embeddings)

        # Compute metrics for full dataset
        topic_info = full_model.get_topic_info()
        outlier_count = topic_info.query("Topic == -1")["Count"].sum() if -1 in topic_info.Topic.values else 0
        outlier_fraction = outlier_count / len(docs)
        num_topics = len(full_model.get_topic_info()) - (1 if -1 in topic_info.Topic.values else 0)
        main_topic_fraction = topic_info.iloc[1]["Count"] / len(docs) if len(topic_info) > 1 else 0
        coherence_score = compute_coherence(full_model, docs)

        # Bootstrap replicates
        bootstrap_topic_assignments = []

        #print(f"Running {num_bootstrap_runs} bootstrap replicates...")
        for i in range(num_bootstrap_runs):
            np.random.seed(SEED + i)  # Ensure reproducible but different subsamples
            sample_indices = np.random.choice(len(docs), size=int(len(docs) * subsample_fraction), replace=True)
            docs_b = [docs[j] for j in sample_indices]
            embeddings_b = embeddings[sample_indices]

            # Re-instantiate models for each bootstrap replicate
            umap_b = UMAP(**umap_full.get_params())
            hdbscan_b = HDBSCAN(**hdbscan_full.get_params())

            bootstrap_model = BERTopic(
                umap_model=umap_b,
                hdbscan_model=hdbscan_b,
                language="french",
                calculate_probabilities=False
            )
            bootstrap_topics, _ = bootstrap_model.fit_transform(docs_b, embeddings_b)

            # Store assignments with original indices
            assignments = {idx: topic for idx, topic in zip(sample_indices, bootstrap_topics)}
            bootstrap_topic_assignments.append(assignments)

        # Compute pairwise ARI and NMI
        ari_scores = []
        nmi_scores = []
        for i, j in combinations(range(num_bootstrap_runs), 2):
            assignments1 = bootstrap_topic_assignments[i]
            assignments2 = bootstrap_topic_assignments[j]

            # Find common documents
            common_docs = set(assignments1.keys()) & set(assignments2.keys())
            if len(common_docs) > 1:
                labels1 = [assignments1[doc_id] for doc_id in common_docs]
                labels2 = [assignments2[doc_id] for doc_id in common_docs]
                ari_scores.append(adjusted_rand_score(labels1, labels2))
                nmi_scores.append(normalized_mutual_info_score(labels1, labels2))

        median_ari = np.median(ari_scores) if ari_scores else 0
        median_nmi = np.median(nmi_scores) if nmi_scores else 0

        results.append({
            "params": params,
            "coherence_score": coherence_score,
            "median_ari": median_ari,
            "median_nmi": median_nmi,
            "num_topics": num_topics,
            "outlier_fraction": outlier_fraction,
            "main_topic_fraction": main_topic_fraction,
        })

    # Save results
    os.makedirs("../results/stability", exist_ok=True)
    with open("../results/stability/metrics.json", 'w') as f:
        json.dump(results, f, indent=4)

    print("Bootstrap stability analysis complete. Results saved to ../results/stability/metrics.json")
    
    # Create visualization
    print("\nCreating grid search metrics visualization...")
    create_grid_search_visualization(results)
    
    print("\n✓ Review the generated figure: ../results/Figures/gridsearch_metrics.png")
    print("✓ Use these results to manually select your preferred configuration in the notebook.")


def create_grid_search_visualization(results):
    """Create comprehensive heatmap visualization of grid search metrics."""
    # Convert to DataFrame
    df_list = []
    for item in results:
        row = item['params'].copy()
        row.update({
            'coherence_score': item['coherence_score'],
            'median_ari': item['median_ari'],
            'median_nmi': item['median_nmi'],
            'num_topics': item['num_topics'],
            'outlier_fraction': item['outlier_fraction'],
            'main_topic_fraction': item['main_topic_fraction']
        })
        df_list.append(row)

    df = pd.DataFrame(df_list)
    df = df.rename(columns={
        'hdbscan__min_cluster_size': 'min_cluster_size',
        'umap__n_components': 'n_components', 
        'umap__n_neighbors': 'n_neighbors'
    })
    
    # Get parameter value ranges
    n_neighbors_vals = sorted(df['n_neighbors'].unique())
    n_components_vals = sorted(df['n_components'].unique())
    min_cluster_vals = sorted(df['min_cluster_size'].unique())
    
    # Define metrics to visualize
    metrics = [
        ('median_nmi', 'NMI', sns.light_palette("seagreen", as_cmap=True)),
        ('median_ari', 'ARI', sns.light_palette("lightblue", as_cmap=True)),
        ('coherence_score', 'Coherence', 'viridis'),
        ('num_topics', 'Num Topics', 'mako'),
        ('outlier_fraction', 'Outlier %', 'rocket'),
        ('main_topic_fraction', 'Main Topic %', 'crest')
    ]
    
    # Create the comprehensive figure
    fig, axes = plt.subplots(len(metrics), len(min_cluster_vals), 
                            figsize=(4 * len(min_cluster_vals), 3 * len(metrics)), 
                            constrained_layout=True)
    
    # Handle single column case
    if len(min_cluster_vals) == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, (metric, title, cmap) in enumerate(metrics):
        for col_idx, min_cluster in enumerate(min_cluster_vals):
            subset = df[df['min_cluster_size'] == min_cluster]
            heatmap_data = subset.pivot(index='n_neighbors', columns='n_components', values=metric)
            heatmap_data = heatmap_data.reindex(index=n_neighbors_vals, columns=n_components_vals)
            
            fmt = 'd' if metric == 'num_topics' else '.3f'
            
            sns.heatmap(
                heatmap_data,
                ax=axes[row_idx, col_idx],
                annot=True,
                fmt=fmt,
                vmin=df[metric].min(),
                vmax=df[metric].max(),
                cbar=col_idx == len(min_cluster_vals)-1,
                cmap=cmap,
            )
            
            axes[row_idx, col_idx].set_title(f'{title}\nmin_cluster_size = {min_cluster}', 
                                           fontsize=12, fontweight='bold')
            axes[row_idx, col_idx].set_xlabel('n_components', fontsize=10)
            
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel('n_neighbors', fontsize=10)
            else:
                axes[row_idx, col_idx].set_ylabel('')
    
    fig.suptitle('Grid Search: All Quality Metrics Across Parameter Space', fontsize=16, fontweight='bold')
    
    # Save the figure
    os.makedirs("../results/Figures", exist_ok=True)
    output_path = '../results/Figures/gridsearch_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid search visualization to: {output_path}")
    plt.close()


# Run the function
bootstrap_stability()
