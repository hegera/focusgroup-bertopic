"""Human validation: document sampling procedure and inter-annotator agreement.

`sample_documents` reproduces the sampling procedure used for the expert review: for each topic,
10 documents drawn at random without replacement with probability proportional to the document's
topic-assignment probability. The original draw was made without a fixed seed, so it cannot be
regenerated exactly; the list of documents actually rated is provided separately.

Running the script recomputes the agreement statistics from the annotators' ratings.
Output: ../results/human_validation/agreement.json
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from settings import SEED

OUT_DIR = "../results/human_validation"

# Coherence ratings (1 = incoherent, 5 = highly coherent); rows = topics 0..6, columns = annotators 1..3
RATINGS = np.array([[2, 2, 1], [5, 5, 3], [5, 5, 4], [4, 5, 4], [3, 4, 3], [4, 5, 4], [4, 4, 4]], dtype=float)
TOPIC_SIZES = [349, 248, 98, 50, 36, 28, 15]


def sample_documents(assignments, n_per_topic=10, seed=SEED):
    """Probability-weighted sample of documents per topic from a document_topic_assignments.csv table."""
    rng = np.random.default_rng(seed)
    samples = {}
    for topic_id, group in assignments[assignments.topic_id != -1].groupby("topic_id"):
        p = group.topic_probability.to_numpy(dtype=float)
        idx = rng.choice(len(group), size=min(n_per_topic, len(group)), replace=False, p=p / p.sum())
        samples[int(topic_id)] = group.sentence.iloc[idx].tolist()
    return samples


def icc_2_1(x):
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout & Fleiss, 1979)."""
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    residual = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)


def agreement():
    kappas = {f"annotator{i + 1}-annotator{j + 1}": float(cohen_kappa_score(
        RATINGS[:, i].astype(int), RATINGS[:, j].astype(int), weights="quadratic", labels=[1, 2, 3, 4, 5]))
        for i, j in combinations(range(RATINGS.shape[1]), 2)}
    rho, p = spearmanr(TOPIC_SIZES, RATINGS.mean(1))
    result = {
        "icc_2_1": float(icc_2_1(RATINGS)),
        "quadratic_weighted_kappa_pairs": kappas,
        "quadratic_weighted_kappa_mean": float(np.mean(list(kappas.values()))),
        "topic_mean_ratings": [round(float(m), 2) for m in RATINGS.mean(1)],
        "overall_mean_rating": float(RATINGS.mean()),
        "topics_within_one_point": int((RATINGS.max(1) - RATINGS.min(1) <= 1).sum()),
        "topics_with_mean_at_least_4": int((RATINGS.mean(1) >= 4).sum()),
        "spearman_topic_size_vs_mean_rating": {"rho": float(rho), "p": float(p)},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "agreement.json"), "w") as f:
        json.dump(result, f, indent=4)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    agreement()
