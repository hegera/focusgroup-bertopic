"""LDA baseline: grid over the number of topics and both Dirichlet priors.

Uses the same stop-word list, tokenisation and coherence procedure as the BERTopic evaluation.
Outputs: ../results/lda/lda_grid.csv, lda_seed_sensitivity.csv, lda_summary.json
"""
import json
import os

import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import LatentDirichletAllocation

from metrics import keyword_vectorizer, quality_metrics, tokenize
from settings import (LDA_DOC_TOPIC_PRIORS, LDA_K_RANGE, LDA_TOPIC_WORD_PRIORS, N_JOBS, SEED, TOP_N_WORDS)

OUT_DIR = "../results/lda"


def label(prior):
    return "default(1/K)" if prior is None else prior


def fit_lda(dtm, feature_names, texts, k, alpha, eta, seed):
    lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=eta,
                                    random_state=seed).fit(dtm)  # batch variational Bayes, max_iter=10
    topics = [[feature_names[i] for i in comp.argsort()[:-TOP_N_WORDS - 1:-1]] for comp in lda.components_]
    return {"k": k, "alpha": label(alpha), "eta": label(eta), "seed": seed, **quality_metrics(topics, texts),
            "topics": " | ".join(", ".join(t) for t in topics)}


def lda_baseline():
    docs = pd.read_csv("../data/data.csv")["sentence"].tolist()
    texts = tokenize(docs)
    vectorizer = keyword_vectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()
    print(f"Document-term matrix: {dtm.shape}")

    jobs = [(k, a, e) for k in LDA_K_RANGE for a in LDA_DOC_TOPIC_PRIORS for e in LDA_TOPIC_WORD_PRIORS]
    print(f"Fitting {len(jobs)} LDA models...")
    grid = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(fit_lda)(dtm, features, texts, k, a, e, SEED) for k, a, e in jobs))

    os.makedirs(OUT_DIR, exist_ok=True)
    grid.to_csv(os.path.join(OUT_DIR, "lda_grid.csv"), index=False)

    is_default = (grid.alpha == "default(1/K)") & (grid.eta == "default(1/K)")
    best = grid.sort_values("c_v", ascending=False).iloc[0]
    best_default = grid[is_default].sort_values("c_v", ascending=False).iloc[0]
    best_k7 = grid[grid.k == 7].sort_values("c_v", ascending=False).iloc[0]

    # seed sensitivity of the selected (best C_V) configuration
    alpha = None if best.alpha == "default(1/K)" else float(best.alpha)
    eta = None if best.eta == "default(1/K)" else float(best.eta)
    seeds = pd.DataFrame(Parallel(n_jobs=N_JOBS)(
        delayed(fit_lda)(dtm, features, texts, int(best.k), alpha, eta, s) for s in range(10)))
    seeds.to_csv(os.path.join(OUT_DIR, "lda_seed_sensitivity.csv"), index=False)

    cols = ["k", "alpha", "eta", "c_v", "c_npmi", "c_uci", "topic_diversity", "topics"]
    as_dict = lambda row: {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in cols}
    summary = {
        "n_models": int(len(grid)), "vocabulary_size": int(dtm.shape[1]),
        "c_v_range": [float(grid.c_v.min()), float(grid.c_v.max())],
        "best_tuned": as_dict(best), "best_default_priors": as_dict(best_default), "best_at_k7": as_dict(best_k7),
        "seed_sensitivity_c_v": {"mean": float(seeds.c_v.mean()), "sd": float(seeds.c_v.std()),
                                 "min": float(seeds.c_v.min()), "max": float(seeds.c_v.max())},
    }
    with open(os.path.join(OUT_DIR, "lda_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Best LDA (tuned priors): K={best.k}, alpha={best.alpha}, eta={best.eta}, C_V={best.c_v:.3f}")
    print(f"Best LDA (default priors): K={best_default.k}, C_V={best_default.c_v:.3f}")
    print(f"Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    lda_baseline()
