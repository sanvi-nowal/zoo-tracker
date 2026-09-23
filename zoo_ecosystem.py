import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment

from hierarchical_clustering import AgglomerativeClusteringFromScratch, gower_distance_matrix

NUMERIC_COLS = ["Size", "Speed", "Color", "Aggression"]
CATEGORICAL_COLS = ["Diet", "Habitat"]
DATA_PATH = "adap-ecosys-dataset.csv"
OUT_DIR = "outputs"

def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)

def global_numeric_ranges(df):
    return (df[NUMERIC_COLS].max() - df[NUMERIC_COLS].min()).values.astype(float)

def features_for_timestep(df, timestep, numeric_ranges):
    sub = df[df["TimeStep"] == timestep].sort_values("SpeciesID").reset_index(drop=True)
    numeric = sub[NUMERIC_COLS].values
    categorical = sub[CATEGORICAL_COLS].values
    D = gower_distance_matrix(numeric, categorical, numeric_ranges=numeric_ranges)
    return sub, D

def compare_linkages_timestep0(df, numeric_ranges):
    sub, D = features_for_timestep(df, timestep=0, numeric_ranges=numeric_ranges)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    models = {}
    for ax, method in zip(axes.ravel(), AgglomerativeClusteringFromScratch.VALID_LINKAGES):
        model = AgglomerativeClusteringFromScratch.fit_auto(D, linkage=method)
        models[method] = model
        dendrogram(model.Z_, truncate_mode="lastp", p=40, ax=ax, no_labels=True)
        ax.set_title(f"{method} linkage (last 40 merges)")
        ax.set_xlabel("cluster (or #species in it)")
        ax.set_ylabel("merge distance")
    fig.suptitle("Hierarchical clustering of Timestep-0 species: linkage comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/linkage_comparison_timestep0.png", dpi=150)
    plt.close(fig)
    return sub, D, models


def choose_and_plot_final_model(sub, D, models, linkage="average"):
    model = models[linkage]
    k = model.best_k_by_gap(range(2, 15))
    labels = model.cut(k)
    print(f"[Task 1] Chosen linkage='{linkage}', gap heuristic suggests k={k} clusters")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(model.Z_, truncate_mode="lastp", p=50, ax=ax, no_labels=True,
               color_threshold=model.Z_[-(k - 1), 2] if k > 1 else None)
    ax.set_title(f"Timestep 0 dendrogram ({linkage} linkage, cut at k={k}) - last 50 merges shown")
    ax.set_xlabel("cluster (or #species in it)")
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dendrogram_timestep0_final.png", dpi=150)
    plt.close(fig)
    print(f"[Task 1] Saved {OUT_DIR}/dendrogram_timestep0_final.png")

    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"[Task 1] Cluster sizes:\n{sizes.to_string()}")
    return model, k, labels

def align_labels(prev_labels, cur_labels, k):
    """
    prev_labels, cur_labels: dict {species_id: cluster_label}
    Returns cur_labels with values remapped to align with prev_labels.
    """
    common_ids = set(prev_labels) & set(cur_labels)
    overlap = np.zeros((k, k))
    for sid in common_ids:
        overlap[prev_labels[sid], cur_labels[sid]] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap)  # maximize total overlap
    mapping = {cur_id: prev_id for prev_id, cur_id in zip(row_ind, col_ind)}
    return {sid: mapping.get(lbl, lbl) for sid, lbl in cur_labels.items()}


def track_evolution(df, numeric_ranges, linkage="average", k=None):
    """
    For every timestep, run the SAME clustering pipeline (one linkage,
    the same k) and track:
      - cluster sizes per timestep
      - fraction of species whose cluster label changed vs. previous timestep
        (species matched by SpeciesID, which stays stable across time)
    """
    timesteps = sorted(df["TimeStep"].unique())
    label_history = {}   # timestep -> {species_id: cluster_label}
    size_history = []    # list of pd.Series (cluster sizes) per timestep
    models_by_ts = {}

    for t in timesteps:
        sub, D = features_for_timestep(df, t, numeric_ranges)
        model = AgglomerativeClusteringFromScratch.fit_auto(D, linkage=linkage)
        this_k = k if k is not None else model.best_k_by_gap(range(2, 15))
        labels = model.cut(this_k)
        cur_label_dict = dict(zip(sub["SpeciesID"].values, labels))

        if t != timesteps[0]:
            prev_t = timesteps[timesteps.index(t) - 1]
            cur_label_dict = align_labels(label_history[prev_t], cur_label_dict, this_k)

        label_history[t] = cur_label_dict
        size_history.append(pd.Series(list(cur_label_dict.values())).value_counts().sort_index())
        models_by_ts[t] = model
        print(f"[Task 2] Timestep {t:2d}: k={this_k} clusters, sizes={dict(size_history[-1])}")

    churn = [0.0]  # timestep 0 has no "previous"
    for i in range(1, len(timesteps)):
        prev_t, cur_t = timesteps[i - 1], timesteps[i]
        prev_labels, cur_labels = label_history[prev_t], label_history[cur_t]
        changed = sum(
            1 for sid in cur_labels
            if sid in prev_labels and prev_labels[sid] != cur_labels[sid]
        )
        churn.append(changed / len(cur_labels))

    n_clusters_over_time = [len(s) for s in size_history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(timesteps, n_clusters_over_time, marker="o")
    axes[0].set_title("Number of clusters found, per timestep")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("# clusters")

    axes[1].plot(timesteps, [c * 100 for c in churn], marker="o", color="darkorange")
    axes[1].set_title("Cluster 'churn': % of species that changed cluster\ncompared to the previous timestep")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("% species reassigned")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ecosystem_evolution_summary.png", dpi=150)
    plt.close(fig)
    print(f"[Task 2] Saved {OUT_DIR}/ecosystem_evolution_summary.png")

    return label_history, size_history, churn, models_by_ts

class ZooEcosystem:
    """
    A live, editable snapshot of the ecosystem at one timestep, backed by
    the clustering we already computed. New species are assigned to a
    cluster by nearest-neighbour distance.
    """

    def __init__(self, species_df, labels, numeric_ranges):
        self.df = species_df.reset_index(drop=True).copy()
        self.df["Cluster"] = labels
        self.numeric_ranges = numeric_ranges
        self._next_species_id = int(self.df["SpeciesID"].max()) + 1

    def _distances_to_all(self, numeric_row, categorical_row):
        numeric = self.df[NUMERIC_COLS].values
        categorical = self.df[CATEGORICAL_COLS].values
        diff = np.abs(numeric - numeric_row) / self.numeric_ranges
        num_component = diff.sum(axis=1)
        cat_component = (categorical != categorical_row).sum(axis=1).astype(float)
        total_features = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)
        return (num_component + cat_component) / total_features

    def _nearest_cluster(self, numeric_row, categorical_row, exclude_species_id=None):
        dists = self._distances_to_all(numeric_row, categorical_row)
        if exclude_species_id is not None:
            mask = self.df["SpeciesID"].values == exclude_species_id
            dists = dists.copy()
            dists[mask] = np.inf
        nearest_idx = np.argmin(dists)
        return int(self.df.loc[nearest_idx, "Cluster"]), float(dists[nearest_idx]), nearest_idx

    def add_species(self, traits, name=None):
        numeric_row = np.array([traits[c] for c in NUMERIC_COLS], dtype=float)
        categorical_row = np.array([traits[c] for c in CATEGORICAL_COLS], dtype=object)
        cluster, dist, _ = self._nearest_cluster(numeric_row, categorical_row)

        new_id = self._next_species_id
        self._next_species_id += 1
        new_row = {**traits, "TimeStep": self.df["TimeStep"].iloc[0],
                   "SpeciesID": new_id,
                   "SpeciesName": name or f"NewSpecies_{new_id}",
                   "Cluster": cluster}
        self.df.loc[len(self.df)] = new_row
        print(f"[Zoo Tool] Added species {new_id} ('{new_row['SpeciesName']}') "
              f"-> assigned to cluster {cluster} (nearest-neighbour distance {dist:.4f})")
        return new_id, cluster

    def mutate_species(self, species_id, trait_changes):
        """trait_changes: dict of {trait_name: new_value} to overwrite."""
        mask = self.df["SpeciesID"] == species_id
        if not mask.any():
            raise ValueError(f"No species with id {species_id}")
        old_cluster = int(self.df.loc[mask, "Cluster"].iloc[0])

        for trait, value in trait_changes.items():
            self.df.loc[mask, trait] = value

        row = self.df.loc[mask].iloc[0]
        numeric_row = row[NUMERIC_COLS].values.astype(float)
        categorical_row = row[CATEGORICAL_COLS].values.astype(object)
        new_cluster, dist, _ = self._nearest_cluster(numeric_row, categorical_row,
                                                       exclude_species_id=species_id)
        self.df.loc[mask, "Cluster"] = new_cluster

        print(f"[Zoo Tool] Mutated species {species_id}: {trait_changes} "
              f"-> cluster {old_cluster} -> {new_cluster}"
              f"{' (unchanged)' if old_cluster == new_cluster else ' (MOVED CLUSTERS)'}")
        return old_cluster, new_cluster

    def save_state(self, filepath):
        state = {
            "df": self.df,
            "numeric_ranges": self.numeric_ranges,
            "next_species_id": self._next_species_id,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"[Zoo Tool] Saved ecosystem state -> {filepath}")

    @classmethod
    def load_state(cls, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["df"].drop(columns=["Cluster"]), state["df"]["Cluster"].values,
                   state["numeric_ranges"])
        obj._next_species_id = state["next_species_id"]
        print(f"[Zoo Tool] Loaded ecosystem state <- {filepath}")
        return obj

if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_dataset()
    ranges = global_numeric_ranges(df)

    # --- Task 1 ---
    sub0, D0, models0 = compare_linkages_timestep0(df, ranges)
    final_model, k, labels0 = choose_and_plot_final_model(sub0, D0, models0, linkage="average")

    # --- Task 2 ---
    label_history, size_history, churn, models_by_ts = track_evolution(
        df, ranges, linkage="average", k=k
    )

    # --- Task 3: Zoo Evolution tool demo ---
    print("\n=== Zoo Evolution Tool demo ===")
    zoo = ZooEcosystem(sub0, labels0, ranges)

    new_id, new_cluster = zoo.add_species({
        "Size": 9.5, "Speed": 9.0, "Color": 200.0,
        "Diet": "Carnivore", "Habitat": "Mountain", "Aggression": 9.2
    }, name="Shadow Fang")

    some_existing_id = int(sub0["SpeciesID"].iloc[0])
    zoo.mutate_species(some_existing_id, {"Speed": 9.8, "Aggression": 9.5, "Diet": "Carnivore"})

    # --- Task 3: save / load demo ---
    state_path = f"{OUT_DIR}/ecosystem_state_demo.pkl"
    zoo.save_state(state_path)
    zoo_reloaded = ZooEcosystem.load_state(state_path)
    assert zoo_reloaded.df.shape == zoo.df.shape
