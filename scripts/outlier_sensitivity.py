"""Sensitivity analysis: reassigning HDBSCAN outliers of the selected model with BERTopic.reduce_outliers.

The selected model itself is not changed; each variant starts from its partition and only moves outliers.
Output: ../results/outliers/outlier_reduction_results.csv
"""
import copy
import json
import os

import numpy as np
import pandas as pd

from fit_final_model import fit_selected_model
from metrics import bertopic_topic_words, keyword_vectorizer, quality_metrics, tokenize

OUT_DIR = "../results/outliers"
VARIANTS = ([("probabilities", t) for t in (0.0, 0.1, 0.2)] + [("embeddings", t) for t in (0.0, 0.4, 0.5, 0.6)]
            + [("c-tf-idf", t) for t in (0.0, 0.1)] + [("distributions", 0.0)])


def describe(model, docs, topics, base_topics, texts, name, threshold):
    outliers = np.where(base_topics == -1)[0]
    moved = topics[outliers]
    moved = moved[moved != -1]
    metrics = quality_metrics(bertopic_topic_words(model), texts)
    row = {"strategy": name, "threshold": threshold,
           "n_outliers": int((topics == -1).sum()), "outlier_pct": round(100 * (topics == -1).mean(), 2),
           "n_reassigned": int(len(moved)),
           "pct_of_reassigned_to_topic0": round(100 * (moved == 0).mean(), 1) if len(moved) else np.nan,
           **metrics, "topic0_pct_of_all": round(100 * (topics == 0).mean(), 2)}
    for t in sorted(set(base_topics) - {-1}):
        row[f"n_topic{t}"] = int((topics == t).sum())
    return row


def outlier_sensitivity():
    with open("../results/selection/best_model_config.json") as f:
        selected_config = json.load(f)
    embeddings = np.load("../embeddings/embeddings.npy")
    docs = pd.read_csv("../data/data.csv")["sentence"].tolist()
    texts = tokenize(docs)

    base_model, base_topics, probs = fit_selected_model(docs, embeddings, selected_config)
    base_topics = np.array(base_topics)
    rows = [describe(base_model, docs, base_topics, base_topics, texts, "none (selected model)", np.nan)]

    for strategy, threshold in VARIANTS:
        model = copy.deepcopy(base_model)
        new_topics = np.array(model.reduce_outliers(
            docs, list(base_topics), strategy=strategy, threshold=threshold,
            probabilities=probs if strategy == "probabilities" else None,
            embeddings=embeddings if strategy == "embeddings" else None))
        model.update_topics(docs, topics=list(new_topics), vectorizer_model=keyword_vectorizer())
        rows.append(describe(model, docs, new_topics, base_topics, texts, strategy, threshold))
        print(f"{strategy} (threshold {threshold}): {rows[-1]['n_outliers']} outliers left, C_V={rows[-1]['c_v']:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "outlier_reduction_results.csv"), index=False)

    max_prob = probs[base_topics == -1].max(axis=1)
    with open(os.path.join(OUT_DIR, "outlier_membership_summary.json"), "w") as f:
        json.dump({"n_outliers": int((base_topics == -1).sum()),
                   "max_soft_membership_probability": {"median": float(np.median(max_prob)), "min": float(max_prob.min()),
                                                       "max": float(max_prob.max())},
                   "share_with_probability_ge_0.5": float((max_prob >= 0.5).mean())}, f, indent=4)
    print(f"Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    outlier_sensitivity()
