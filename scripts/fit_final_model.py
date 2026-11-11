import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from datetime import datetime
import json
from metrics import bertopic_topic_words, keyword_vectorizer, quality_metrics, tokenize
from settings import SEED

PUBLIC_COLUMNS = ["sentence", "participant", "gender", "region", "age"]


def fit_selected_model(docs, embeddings, selected_config):
    """Fit BERTopic with the selected configuration; returns (model, topics, probabilities)."""
    umap_model = UMAP(
        n_neighbors=selected_config["umap__n_neighbors"],
        n_components=selected_config["umap__n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=SEED
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=selected_config["hdbscan__min_cluster_size"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    topic_model = BERTopic(
        language="french",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=keyword_vectorizer(),  # stop words removed from topic keywords only
        calculate_probabilities=True,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs


def fit_final_model():
    """Fit final BERTopic model with selected configuration and extract topics."""
    
    # Check if selection exists
    config_path = "../results/selection/best_model_config.json"
    if not os.path.exists(config_path):
        print("ERROR: No model configuration selected!")
        print(f"Please create {config_path} with your chosen configuration.")
        print("Run the model selection step (Section 4) in the notebook first.")
        return
    
    # Load selected configuration
    print("Loading selected model configuration...")
    with open(config_path, 'r') as f:
        selected_config = json.load(f)
    
    print(f"\nSelected configuration:")
    print(f"  n_neighbors: {selected_config['umap__n_neighbors']}")
    print(f"  n_components: {selected_config['umap__n_components']}")
    print(f"  min_cluster_size: {selected_config['hdbscan__min_cluster_size']}")
    print(f"  Rationale: {selected_config.get('selection_rationale', 'N/A')}")
    
    # Load data
    print("\nLoading data...")
    embeddings = np.load("../embeddings/embeddings.npy")
    data_df = pd.read_csv("../data/data.csv")
    docs = data_df['sentence'].tolist()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"../results/final_model_v{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Fit BERTopic model
    print("\nFitting BERTopic model...")
    topic_model, topics, probs = fit_selected_model(docs, embeddings, selected_config)
    
    final_topics = len([t for t in topic_model.get_topics().keys() if t != -1])
    print(f"Number of topics discovered: {final_topics}")

    # Extract topic information with representative documents
    print("\nExtracting topic information...")
    topic_info = topic_model.get_topic_info()
    
    # Get representative documents for each topic
    repr_docs_dict = {}
    for topic_id in topic_info['Topic']:
        if topic_id != -1:  # Skip outlier topic
            repr_docs = topic_model.get_representative_docs(topic_id)
            repr_docs_dict[topic_id] = repr_docs[:3]  # Top 3 representative docs
    
    # Create topic details CSV with representative documents
    topic_details = []
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_label = "Outlier"
            repr_docs = []
        else:
            topic_label = f"Topic {topic_id}"
            repr_docs = repr_docs_dict.get(topic_id, [])
        
        topic_details.append({
            'topic_id': topic_id,
            'topic_label': topic_label,
            'count': row['Count'],
            'top_words': ', '.join([word for word, _ in topic_model.get_topic(topic_id)[:10]]) if topic_id != -1 else '',
            'repr_doc_1': repr_docs[0] if len(repr_docs) > 0 else '',
            'repr_doc_2': repr_docs[1] if len(repr_docs) > 1 else '',
            'repr_doc_3': repr_docs[2] if len(repr_docs) > 2 else ''
        })
    
    topic_details_df = pd.DataFrame(topic_details)
    topic_details_df.to_csv(os.path.join(model_dir, "topic_details.csv"), index=False)
    print(f"Saved topic details to: {model_dir}/topic_details.csv")
    
    # Create document assignments CSV
    print("\nCreating document assignments...")
    doc_assignments = data_df[[c for c in PUBLIC_COLUMNS if c in data_df.columns]].copy()
    doc_assignments['topic_id'] = topics
    doc_assignments['topic_probability'] = [
        probs[i][topics[i]] if topics[i] != -1 and probs is not None else 0.0
        for i in range(len(topics))
    ]
    doc_assignments.to_csv(os.path.join(model_dir, "document_topic_assignments.csv"), index=False)
    print(f"Saved document assignments to: {model_dir}/document_topic_assignments.csv")
    
    # Create datamapplot visualization
    print("\nCreating topic visualization with datamapplot...")
    try:
        fig = topic_model.visualize_document_datamap(
            docs, 
            embeddings=embeddings, 
            custom_labels=False, 
            title="BERTopic Topic Map", 
            topic_prefix=True
        )
        
        # Save as PNG file
        viz_path = os.path.join(model_dir, "topic_map.png")
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved topic visualization to: {viz_path}")
        
    except Exception as e:
        print(f"✗ Could not create visualization: {type(e).__name__}: {e}")
    
    # Save configuration
    config_out = {
        "seed": SEED,
        "embedding_model": open("../configs/embedding_model.txt").read().strip(),
        "umap__n_neighbors": selected_config["umap__n_neighbors"],
        "umap__n_components": selected_config["umap__n_components"],
        "hdbscan__min_cluster_size": selected_config["hdbscan__min_cluster_size"],
        "num_topics": final_topics,
        "outlier_fraction": list(topics).count(-1) / len(topics),
        "quality_metrics": quality_metrics(bertopic_topic_words(topic_model), tokenize(docs)),
        "topic_keywords": "c-TF-IDF, top 10 words, NLTK French stop words removed",
        "selection_rationale": selected_config.get('selection_rationale', 'N/A'),
        "timestamp": timestamp
    }
    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config_out, f, indent=4)
    
    # Save the fitted model
    print("\nSaving fitted model...")
    model_path = os.path.join(model_dir, "topic_model")
    topic_model.save(model_path, serialization="pickle", save_ctfidf=True, save_embedding_model=False)
    print(f"✓ Saved model to: {model_path}")
    
    # Calculate summary statistics
    outlier_count = list(topics).count(-1)
    outlier_fraction = outlier_count / len(topics)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL MODEL SUMMARY")
    print("="*80)
    print(f"Output directory: {model_dir}")
    print(f"Number of topics: {final_topics} (+ {outlier_count} outliers)")
    print(f"Outlier fraction: {outlier_fraction:.2%}")
    print(f"\nOutputs:")
    print(f"  - Topic details: topic_details.csv")
    print(f"  - Document assignments: document_topic_assignments.csv")
    print(f"  - Topic visualization: topic_map.png")
    print(f"  - Fitted model: topic_model/")
    print(f"  - Configuration: config.json")
    print("="*80)


if __name__ == "__main__":
    fit_final_model()
