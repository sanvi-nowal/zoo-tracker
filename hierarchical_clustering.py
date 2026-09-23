import numpy as np

def gower_distance_matrix(numeric, categorical, numeric_ranges=None):
    """
    Gower distance per feature:
      - numeric:      |x_i - x_j| / range(feature)      (0..1)
      - categorical:  0 if equal else 1                  (0..1)
    Final distance = mean over all p+q features (equal weighting).
    """
    numeric = np.asarray(numeric, dtype=float)
    categorical = np.asarray(categorical, dtype=object)
    n, p = numeric.shape
    q = categorical.shape[1] if categorical.ndim == 2 else 0

    if numeric_ranges is None:
        numeric_ranges = numeric.max(axis=0) - numeric.min(axis=0)
    numeric_ranges = np.where(numeric_ranges == 0, 1.0, numeric_ranges)  # avoid /0
    diff = np.abs(numeric[:, None, :] - numeric[None, :, :]) / numeric_ranges[None, None, :]
    numeric_component = diff.sum(axis=2)  # (n, n)

    categorical_component = np.zeros((n, n))
    for col in range(q):
        col_vals = categorical[:, col]
        mismatch = (col_vals[:, None] != col_vals[None, :]).astype(float)
        categorical_component += mismatch

    total_features = p + q
    D = (numeric_component + categorical_component) / total_features
    np.fill_diagonal(D, 0.0)
    return D

class AgglomerativeClusteringFromScratch:
    """
    Lance-Williams formula, merging clusters i and j into ij, distance
    to any other cluster k:
        d(ij, k) = a_i * d(i,k) + a_j * d(j,k) + b * d(i,j) + g * |d(i,k) - d(j,k)|
    """
    VALID_LINKAGES = ("single", "complete", "average", "ward")

    def __init__(self, linkage="average"):
        if linkage not in self.VALID_LINKAGES:
            raise ValueError(f"linkage must be one of {self.VALID_LINKAGES}")
        self.linkage = linkage
        self.Z_ = None  # linkage matrix, scipy-compatible (n-1, 4)

    def fit(self, D):
        """
        D: (n, n) precomputed distance matrix (symmetric, zero diagonal).
        Builds self.Z_ : (n-1, 4) linkage matrix
            [id_a, id_b, merge_distance, size_of_new_cluster]
        Leaf clusters are ids 0..n-1; every merge creates a new id
        n, n+1, ... (this matches scipy's convention exactly).
        """
        D = np.array(D, dtype=float, copy=True)
        n = D.shape[0]

        active = list(range(n))                      
        sizes = {i: 1 for i in range(n)}             
        next_id = n
        Z = np.zeros((n - 1, 4))

        INF = np.inf
        dist = D.copy()
        np.fill_diagonal(dist, INF)  

        for step in range(n - 1):
            flat_idx = np.argmin(dist)
            a, b = np.unravel_index(flat_idx, dist.shape)
            if a > b:
                a, b = b, a
            d_ab = dist[a, b]
            id_a, id_b = active[a], active[b]
            n_i, n_j = sizes[id_a], sizes[id_b]
            new_size = n_i + n_j

            Z[step] = [id_a, id_b, d_ab, new_size]
            other_mask = np.ones(dist.shape[0], dtype=bool)
            other_mask[[a, b]] = False
            d_ik = dist[a, other_mask]
            d_jk = dist[b, other_mask]

            if self.linkage == "single":
                new_d = 0.5 * d_ik + 0.5 * d_jk - 0.5 * np.abs(d_ik - d_jk)
            elif self.linkage == "complete":
                new_d = 0.5 * d_ik + 0.5 * d_jk + 0.5 * np.abs(d_ik - d_jk)
            elif self.linkage == "average":
                new_d = (n_i * d_ik + n_j * d_jk) / new_size
            elif self.linkage == "ward":
                other_ids = [active[idx] for idx, keep in enumerate(other_mask) if keep]
                n_k = np.array([sizes[oid] for oid in other_ids], dtype=float)
                denom = n_i + n_j + n_k
                a_i = (n_i + n_k) / denom
                a_j = (n_j + n_k) / denom
                b_coef = -n_k / denom
                new_d = a_i * d_ik + a_j * d_jk + b_coef * d_ab

            keep_idx = np.where(other_mask)[0]
            m = len(keep_idx)
            new_dist = np.full((m + 1, m + 1), INF)
            new_dist[:m, :m] = dist[np.ix_(keep_idx, keep_idx)]
            new_dist[:m, m] = new_d
            new_dist[m, :m] = new_d
            new_dist[m, m] = INF

            dist = new_dist
            active = [active[idx] for idx in keep_idx] + [next_id]
            sizes[next_id] = new_size
            next_id += 1

        self.Z_ = Z
        self.n_leaves_ = n
        return self

    def cut(self, n_clusters):
        """
        Cut the dendrogram to produce `n_clusters` flat clusters.
        """
        if self.Z_ is None:
            raise RuntimeError("Call fit() before cut().")
        n = self.n_leaves_
        if not (1 <= n_clusters <= n):
            raise ValueError("n_clusters must be between 1 and n_leaves")

        parent = list(range(2 * n - 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y, new_id):
            rx, ry = find(x), find(y)
            parent[rx] = new_id
            parent[ry] = new_id
            parent[new_id] = new_id

        n_merges_to_do = n - n_clusters
        for step in range(n_merges_to_do):
            a, b = int(self.Z_[step, 0]), int(self.Z_[step, 1])
            union(a, b, n + step)

        # map each leaf to its root, then relabel roots -> 0..n_clusters-1
        roots = np.array([find(i) for i in range(n)])
        unique_roots = {r: idx for idx, r in enumerate(np.unique(roots))}
        labels = np.array([unique_roots[r] for r in roots])
        return labels

    @classmethod
    def fit_auto(cls, D, linkage="average"):
        model = cls(linkage=linkage)
        if linkage == "ward":
            model.fit(D ** 2)
            model.Z_[:, 2] = np.sqrt(model.Z_[:, 2])
        else:
            model.fit(D)
        return model

    def best_k_by_gap(self, k_range=range(2, 15)):
        n = self.n_leaves_
        merge_dists = self.Z_[:, 2]
        candidates = []
        for k in k_range:
            merge_idx = n - 1 - k
            if 0 <= merge_idx < n - 1:
                gap = merge_dists[merge_idx] - merge_dists[merge_idx - 1] if merge_idx > 0 else merge_dists[merge_idx]
                candidates.append((gap, k))
        if not candidates:
            return 2
        candidates.sort(reverse=True)
        return candidates[0][1]
