### Zoo Ecosystem

Implements agglomerative hierarchical clustering from scratch and applies it to a
synthetic, evolving zoo dataset.

#### Procedure

- **Linkage comparison:** builds a Gower distance matrix for timestep 0
  and compares `single`, `complete`, `average`, and `ward` linkages via dendrograms.
  Picks a final model (`average` linkage) and cuts it at a `k` chosen by a gap-statistic
  heuristic.
- **Evolution tracking:** re-runs clustering at every timestep, aligns cluster
  labels across time (via Hungarian/`linear_sum_assignment` matching on label overlap),
  and plots cluster count and "churn" (% of species that switched clusters) over time.
- **Zoo Evolution tool:** a `ZooEcosystem` class that lets you add new species
  or mutate existing ones, assigning them to the nearest cluster by Gower distance, and
  save/load ecosystem snapshots to/from pickle files.
