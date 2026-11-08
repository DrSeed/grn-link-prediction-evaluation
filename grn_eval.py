# Supervised TF-target link prediction, evaluated the honest way (TF-held-out splits, TF-matched negatives).
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold


def simulate_grn(n_tfs=24, n_genes=400, n_cells=600, n_states=5, chip_sensitivity=0.7, seed=0):
    """Cells x genes expression (TFs first), true edges, and incomplete 'ChIP-seq' labels."""
    rng = np.random.default_rng(seed)
    state = rng.integers(0, n_states, n_cells)
    tf = rng.normal(0, 1, (n_states, n_tfs))[state] + rng.normal(0, 0.6, (n_cells, n_tfs))
    degree = np.clip(rng.lognormal(2.5, 1.0, n_tfs).astype(int), 3, n_genes // 3)  # a few hub TFs
    popularity = rng.lognormal(0, 1.0, n_genes)  # some genes sit in open chromatin and get bound by many TFs
    W = np.zeros((n_tfs, n_genes))
    for t in range(n_tfs):
        targets = rng.choice(n_genes, degree[t], replace=False, p=popularity / popularity.sum())
        W[t, targets] = rng.choice([-1, 1], degree[t]) * rng.uniform(0.2, 0.8, degree[t])
    genes = tf @ W + rng.normal(0, 2.0, (n_cells, n_genes))
    expr = np.hstack([tf, genes])
    true = W != 0
    observed = true & (rng.uniform(size=true.shape) < chip_sensitivity)
    return expr, true, observed


def gene_embeddings(expr, n_pcs=8):
    """Each gene's loadings on the top PCs: the kind of per-gene vector a GNN or transformer learns."""
    Z = (expr - expr.mean(0)) / (expr.std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Vt[:n_pcs].T, Z


def pair_features(expr, tfs, targets, n_tfs, n_pcs=8):
    emb, Z = gene_embeddings(expr, n_pcs)
    corr = (Z[:, tfs] * Z[:, n_tfs + targets]).mean(0)
    return np.column_stack([corr, np.abs(corr), emb[tfs], emb[n_tfs + targets]]), np.abs(corr)


def sample_pairs(observed, negatives="random", ratio=3, seed=0):
    """Positives = observed edges. Negatives are either uniform over all unlabelled pairs, or
    TF-matched: each TF gets `ratio` negatives per positive, drawn from genes that are labelled
    targets of some other TF, so neither 'hub TF' nor 'popular gene' gives the label away."""
    rng = np.random.default_rng(seed)
    pos_t, pos_g = np.nonzero(observed)
    if negatives == "random":
        un_t, un_g = np.nonzero(~observed)
        pick = rng.choice(len(un_t), ratio * len(pos_t), replace=False)
        neg_t, neg_g = un_t[pick], un_g[pick]
    else:
        neg_t, neg_g = [], []
        bound_somewhere = observed.any(axis=0)
        for t in range(observed.shape[0]):
            cand = np.flatnonzero(~observed[t] & bound_somewhere)
            k = min(ratio * observed[t].sum(), len(cand))
            neg_g.append(rng.choice(cand, k, replace=False))
            neg_t.append(np.full(k, t))
        neg_t, neg_g = np.concatenate(neg_t), np.concatenate(neg_g)
    tfs = np.concatenate([pos_t, neg_t])
    targets = np.concatenate([pos_g, neg_g])
    y = np.r_[np.ones(len(pos_t)), np.zeros(len(neg_t))]
    return tfs, targets, y


def cross_validate(X, y, tfs, split="pair", n_folds=3, seed=0):
    """Out-of-fold scores. 'pair' shuffles TF-target pairs; 'tf' holds out whole TFs (GroupKFold)."""
    scores = np.zeros(len(y))
    folds = (KFold(n_folds, shuffle=True, random_state=seed).split(X) if split == "pair"
             else GroupKFold(n_folds).split(X, y, groups=tfs))
    for train, test in folds:
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
        scores[test] = model.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return scores


def metrics(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)
