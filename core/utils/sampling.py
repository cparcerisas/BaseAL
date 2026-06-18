"""
Sampling strategies for active learning
"""

import logging
from typing import List, Optional, Tuple

# from skactiveml.pool import UncertaintySampling  # Not currently used in the code; temporarily commented out.
import faiss
import numpy as np
import pandas as pd
from skactiveml.pool import CoreSet
from skactiveml.utils import MISSING_LABEL, rand_argmax
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def densityEstimation(
    embeddings: Optional[np.ndarray] = None, method="cosine", beta: int = 1, k: int = 20
):
    if method == "cosine":
        similarity = cosine_similarity(embeddings)
    elif method == "euclidean":
        similarity = euclidean_distances(embeddings)
    elif method == "knn":
        knn = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distance, _ = knn.kneighbors(embeddings)
        similarity = distance.T
    else:
        raise Exception("Unknown similarity estimation method, ")

    density = np.power(
        np.sum(similarity, axis=0) / np.sum(similarity, axis=0).max(), beta
    )
    return density


def KMeansEstimation(
    embeddings: Optional[np.ndarray] = None,
    num_classes: int = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    use_umap = False

    # Normalise embeddings before clustering
    embeddings_norm = normalize(embeddings, norm="l2")

    if use_umap:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        embeddings_for_clustering = reducer.fit_transform(embeddings_norm)
        print("UMAP embeddings shape computed")
    else:
        embeddings_for_clustering = embeddings_norm

    kmeans = KMeans(n_clusters=num_classes, random_state=random_state, n_init="auto")
    kmeans.fit(embeddings_for_clustering)

    # For each centroid, pick the closest actual sample
    centroids = kmeans.cluster_centers_
    selected_local = []
    for centroid in centroids:
        dists = np.linalg.norm(embeddings_for_clustering - centroid, axis=1)
        closest = np.argmin(dists)
        selected_local.append(closest)

    # Deduplicate (two centroids may map to the same sample)
    selected_local = list(set(selected_local))

    # # Plot check for sanity
    # import matplotlib.pyplot as plt
    # reducer = umap.UMAP(n_components=2, random_state=42)
    # umap_embeddings = reducer.fit_transform(self.embeddings)
    # plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])
    # # plot the top_indices samples with a different color
    # plt.scatter(umap_embeddings[top_indices, 0], umap_embeddings[top_indices, 1], c='red')
    # plt.show()
    return selected_local


def uniformEmbeddingSampling(
    embeddings: Optional[np.ndarray] = None,
    n_samples: int = 20,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Uniform sampling in embedding space: selects samples that are uniformly distributed in the embedding space.

    This implementation projects embeddings onto the two principal eigenvectors of the covariance
    matrix, allocates the warmup budget between the two directions in proportion to their eigenvalues,
    and samples roughly uniformly across quantile bins of each projection. The function returns a
    utility array of shape (n_samples_total,) with 1.0 for selected items and 0.0 otherwise.

    Args:
        embeddings: Numpy array of shape (n_total, embedding_dim) containing the embeddings of the samples.
        n_samples: Number of samples to select.
        random_state: Optional random seed for reproducibility.

    Returns:
        utility: Array of utility scores for samples [0, 1] where 1 = selected samples, 0 = non-selected samples.
    """
    if embeddings is None:
        raise ValueError("embeddings must be provided for uniformEmbeddingSampling")

    n = embeddings.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    if n_samples >= n:
        return np.ones(n, dtype=np.float32)

    rng = np.random.default_rng(random_state)

    # covariance and eigen decomposition (sort descending)
    cov = embeddings.T @ embeddings
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(-eigenvalues)
    eig1 = eigenvectors[:, idx[0]]
    eig2 = eigenvectors[:, idx[1]] if eigenvectors.shape[1] > 1 else eig1
    ev1 = float(eigenvalues[idx[0]])
    ev2 = float(eigenvalues[idx[1]]) if eigenvalues.shape[0] > 1 else 0.0

    proj1 = embeddings @ eig1
    proj2 = embeddings @ eig2

    # allocate samples proportionally to eigenvalues
    total_ev = ev1 + ev2 if (ev1 + ev2) > 0 else 1.0
    n1 = int(round(n_samples * (ev1 / total_ev)))
    n1 = max(1, min(n_samples - 1, n1))
    n2 = n_samples - n1

    selected = []

    def sample_from_projection(proj: np.ndarray, target_count: int):
        if target_count <= 0:
            return []
        q_edges = np.linspace(0.0, 1.0, target_count + 1)
        picks = []
        for i in range(target_count):
            lo_q = q_edges[i]
            hi_q = q_edges[i + 1]
            lo_val = np.quantile(proj, lo_q)
            hi_val = np.quantile(proj, hi_q)
            if i == target_count - 1:
                idxs = np.where((proj >= lo_val) & (proj <= hi_val))[0]
            else:
                idxs = np.where((proj >= lo_val) & (proj < hi_val))[0]
            if idxs.size == 0:
                continue
            if idxs.size == 1:
                picks.append(int(idxs[0]))
            else:
                picks.append(int(rng.choice(idxs, size=1)[0]))
        return picks

    selected.extend(sample_from_projection(proj1, n1))
    selected.extend(sample_from_projection(proj2, n2))

    # unique while preserving order
    seen = set()
    uniq_selected = []
    for s in selected:
        if s not in seen:
            seen.add(s)
            uniq_selected.append(s)
    selected = uniq_selected

    # fill up if fewer than required
    if len(selected) < n_samples:
        remaining = np.setdiff1d(np.arange(n), np.array(selected, dtype=int))
        if remaining.size > 0:
            need = n_samples - len(selected)
            more = rng.choice(remaining, size=min(need, remaining.size), replace=False)
            selected.extend([int(x) for x in more])

    # trim if oversubscribed
    if len(selected) > n_samples:
        selected = rng.choice(
            np.array(selected, dtype=int), size=n_samples, replace=False
        ).tolist()

    utility = np.zeros(n, dtype=np.float32)
    utility[np.array(selected, dtype=int)] = 1.0
    return utility


def _sample_pool(
    unlabeled: np.ndarray, pool_size: int, rng: np.random.Generator
) -> np.ndarray:
    if pool_size <= 0 or len(unlabeled) <= pool_size:
        return unlabeled
    return rng.choice(unlabeled, size=pool_size, replace=False)


def _build_hnsw_index(x: np.ndarray, m: int = 32, ef_search: int = 128) -> faiss.Index:
    d = int(x.shape[1])
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efSearch = ef_search
    index.add(x.astype(np.float32, copy=False))
    return index


def _project_with_pca(
    x: np.ndarray, out_dim: int, train_rows: int = 20000
) -> np.ndarray:
    if out_dim <= 0 or out_dim >= x.shape[1]:
        return x
    n_train = min(train_rows, x.shape[0])
    if n_train <= out_dim:
        return x
    pca = faiss.PCAMatrix(x.shape[1], out_dim)
    pca.train(x[:n_train].astype(np.float32, copy=False))
    return pca.apply_py(x.astype(np.float32, copy=False))


class SamplingStrategy:
    """
    Unified sampling strategy class that handles all sampling methods.

    This class contains the selection logic and various sampling methods.
    Data is stored as instance attributes and accessed by sampling methods.
    """

    def __init__(
        self,
        method: str = "random",
        n_samples: int = 20,
        random_state: Optional[int] = None,
        label_to_idx: Optional[list] = None,
    ):
        """
        Initialize sampling strategy

        Args:
            method: Sampling method to use ('random', 'margin', 'custom', 'margin_multilabel', 'coreset_farthest', 'nn_disagreement')
            n_samples: Number of samples to select per iteration
            random_state: Optional random seed for reproducibility
        """
        self.method = method
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)
        self.label_to_idx = label_to_idx

        # Available sampling methods
        available_methods = [
            "random",
            "margin",
            "custom",
            "bald",
            # 'margin_multilabel',d
            "coreset_farthest",
            "nn_disagreement",
            "margin_multilabel",
            "sklearn_coreset",
            "sklearn_typiclust",
            "all_quantiles",
            "high_quantile_per_class",
            "best_single",
            "best_multiclass",
            "most_confident_classes",
            "balance_class_by_clusters",
            "balance_class_by_confidence",
            "sample_from_dense_clusters",
            "sklearn_coreset_no_noise",
            "sklearn_coreset_far_from_noise",
        ]

        if method not in available_methods:
            raise ValueError(
                f"Unknown sampling strategy: {method}. "
                f"Available strategies: {available_methods}"
            )

        # Map method names to their implementation functions
        self._method_map = {
            "random": self._random,
            "margin": self._margin,
            "custom": self._custom,
            "bald": self._bald,
            # 'margin_multilabel': self._margin_multilabel,
            "coreset_farthest": self._coreset_farthest,
            "nn_disagreement": self._nn_disagreement,
            "margin_multilabel": self._margin_multilabel,
            "sklearn_coreset": self._sklearn_coreset,
            "sklearn_typiclust": self._sklearn_typiclust,
            "all_quantiles": self._all_quantiles,
            "high_quantile_per_class": self._high_quantile_per_class,
            "best_single": self._best_single,
            "best_multiclass": self._best_multiclass,
            "most_confident_classes": self._most_confident_classes,
            "balance_class_by_clusters": self._balance_class_by_clusters,
            "balance_class_by_confidence": self._balance_class_by_confidence,
            "sample_from_dense_clusters": self._sample_from_dense_clusters,
            "sklearn_coreset_no_noise": self._sklearn_coreset_no_noise,
            "sklearn_coreset_far_from_noise": self._sklearn_coreset_far_from_noise,
        }

        # Data attributes (see selct)
        self.unlabeled_indices = None
        self.predictions = None
        self.embeddings = None
        self.model = None
        self.metadata = None
        self.labeled_indices = None
        self.labels = None

        self.quantiles = [0, 0.25, 0.85, 1]
        self.clusters = None
        logger.info(
            f"Initialized SamplingStrategy with method='{method}' and n_samples={n_samples}"
        )

    def select(
        self,
        unlabeled_indices: List[int],
        predictions: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        model=None,
        metadata: Optional[pd.DataFrame] = None,
        labeled_indices: Optional[List[int]] = None,
        labels: Optional[np.ndarray] = None,
        mc_predictions: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Select samples for annotation and compute per-sample utility.

        This is the main selection method that stores the input data as instance
        attributes and calls the appropriate sampling method.

        Args:
            unlabeled_indices: List/array of unlabeled sample indices
            predictions: Optional numpy array of model predictions (N x num_classes)
            embeddings: Optional numpy array of embeddings (N x embedding_dim)
            model: Optional reference to the model itself
            metadata: Optional DataFrame containing metadata
            labeled_indices: Optional list/array of labeled sample indices
            labels: Optional ground-truth labels for all samples
            mc_predictions: Optional repeated forward-pass predictions (mc_passes, N, num_classes)

        Returns:
            Tuple of (selected_indices, utility):
                - selected_indices: List of selected sample indices
                - utility: Normalized utility scores for unlabeled samples [0, 1]
                  where 1 = maximum utility, 0 = lowest utility
        """
        if len(unlabeled_indices) == 0:
            logger.warning("No unlabeled samples available for selection")
            return [], np.array([])

        # Store data as instance attributes for sampling methods to access
        self.unlabeled_indices = unlabeled_indices
        self.predictions = predictions
        self.embeddings = embeddings
        self.model = model
        self.metadata = metadata
        self.labeled_indices = labeled_indices if labeled_indices is not None else []
        self.labels = labels
        self.mc_predictions = mc_predictions

        # Call the appropriate sampling method to get utility scores
        sampling_func = self._method_map[self.method]
        utility = np.asarray(sampling_func(), dtype=np.float32)

        if len(utility) != len(unlabeled_indices):
            raise ValueError(
                f"Sampling method '{self.method}' returned {len(utility)} scores, "
                f"expected {len(unlabeled_indices)}"
            )

        # Select samples with highest utility
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(utility)[-n_samples:]  # Highest uncertainties

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        logger.info(f"Selected {len(selected)} samples using {self.method} sampling")
        logger.info(
            f"utility range: min={utility.min():.4f}, max={utility.max():.4f}, mean={utility.mean():.4f}"
        )

        return selected, utility

    def _random(self) -> np.ndarray:
        """
        Random sampling strategy - assigns equal utility to all samples.

        For random sampling, all samples have equal utility (1.0), so selection
        is effectively random.

        Returns:
            utility: Array of 1.0 for all unlabeled samples (equal utility)
        """
        # Sample random utility to make top-k selection random.
        utility = self.rng.random(len(self.unlabeled_indices), dtype=np.float32)
        return utility

    def _margin(self) -> np.ndarray:
        """
        Margin sampling - selects samples with smallest margin between top two predictions.

        The margin is the difference between the highest and second-highest predicted
        class probabilities. Smaller margins indicate more ambiguous predictions.

        Returns:
            utility: Normalized utility scores (1 - margin) for all unlabeled samples [0, 1]
        """
        if self.predictions is None:
            raise ValueError("Margin sampling requires predictions")

        unlabeled_preds = self.predictions[self.unlabeled_indices]

        # Sort predictions for each sample to get top 2
        sorted_preds = np.sort(unlabeled_preds, axis=1)

        # Calculate margin: difference between top two predictions
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]

        # Uncertainty = 1 - margin (smaller margin = higher uncertainty, already normalized to [0, 1])
        utility = 1.0 - margins

        logger.info(
            f"Margin sampling - margins min: {margins.min():.4f}, max: {margins.max():.4f}"
        )
        logger.info(
            f"Margin sampling - utility min: {utility.min():.4f}, max: {utility.max():.4f}"
        )
        return utility

    def _custom(self) -> np.ndarray:
        """
        Custom sampling template.

        INSTRUCTIONS FOR IMPLEMENTING CUSTOM SAMPLING:
        ===============================================

        1. This method computes utility scores for all unlabeled samples.

        2. The utility scores should be normalized to [0, 1] where:
           - 1.0 = maximum utility (highest priority for annotation)
           - 0.0 = lowest utility (lowest priority for annotation)

        3. Available instance attributes (set by select() method):
           - self.unlabeled_indices: List of indices in the unlabeled pool
           - self.predictions: Model predictions array of shape (n_total_samples, num_classes)
                              Contains probabilities for all classes
           - self.embeddings: Full embeddings array of shape (n_total_samples, embedding_dim)
                             The raw feature vectors before classification
           - self.model: Reference to the trained model (if you need to extract features/gradients)
           - self.metadata: DataFrame containing annotation data and metadata
                              Can contain custom metadata fields for advanced sampling strategies

        Returns:
            utility: Array of utility scores for samples [0, 1]
        """
        # TODO: Implement your custom sampling logic here
        # For now, default to random sampling
        # logger.warning("Custom sampling not implemented, falling back to random sampling")
        return self._random()

    def _all_quantiles(self) -> np.ndarray:
        """
        All quantiles per class. This method selects per predicted class a fixed amount of samples of each quantile, regardless of the prediction of the other classes
        """
        n_classes = self.predictions.shape[1]
        n_per_class = int(self.n_samples / n_classes)
        n_per_quantile = max(int(n_per_class / (len(self.quantiles) + 1)), 1)
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["utility"] = np.zeros(len(samples))

        for c in np.arange(n_classes):
            samples[f"quantile_{c}"] = pd.qcut(
                samples[c], self.quantiles, duplicates="drop", labels=False
            )
            for _, quantile in samples.groupby(f"quantile_{c}"):
                randomly_selected_samples = quantile.sample(
                    n_per_quantile, random_state=self.rng.integers(80)
                )
                samples.loc[randomly_selected_samples.index, "utility"] = 1

        return samples["utility"].values

    def _high_quantile_per_class(self) -> np.ndarray:
        """
        high quantile per predicted class: this method selects, per each class an equal amount of samples from the highest quantile.
        """
        n_classes = self.predictions.shape[1]
        n_per_class = int(self.n_samples / n_classes)
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["utility"] = np.zeros(len(samples))

        for c in np.arange(n_classes):
            quantiles = pd.qcut(
                samples[c], self.quantiles, duplicates="drop", labels=False
            )
            randomly_selected_samples = samples[
                quantiles == (len(self.quantiles) - 1)
            ].sample(n_per_class, random_state=self.rng.integers(80))
            samples.loc[randomly_selected_samples.index, "utility"] = 1

        return samples["utility"].values

    def _best_single(self) -> np.ndarray:
        """
        Best isolated call, per predicted class an equal amount of samples. This method attemps to select the samples which are high presence confidence (highest quantile) for ONLY one class,
        and high absence confidence for all the other classes (lowest quantile).

        """
        n_classes = self.predictions.shape[1]
        n_per_class = int(self.n_samples / n_classes)
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["utility"] = np.zeros(len(samples))

        quantiles_columns = []
        for c in np.arange(n_classes):
            samples[f"quantile_{c}"] = pd.qcut(
                samples[c], self.quantiles, duplicates="drop", labels=False
            )
            quantiles_columns.append(f"quantile_{c}")

        high_quantile = len(self.quantiles) - 1
        samples["utility"] = 0
        for c in np.arange(n_classes):
            # Samples which are in quantile max for one class and in quantile 0 for the others
            high_quality_class_samples = (samples[f"quantile_{c}"] == high_quantile) & (
                (samples[quantiles_columns]).sum(axis=1) == 2
            )
            selection = samples[high_quality_class_samples].sample(
                min(n_per_class, high_quality_class_samples.sum())
            )
            samples.loc[selection.index, "utility"] = 1

        return samples["utility"].values

    def _best_multiclass(self) -> np.ndarray:
        """
        This sampling strategy takes samples where the model is very sure about presence of multiple classes (all highest quantile),
        regardless of which/how many classes, and sure about the absence of the rest of the classes (lowest quantile). It minimizes the selection of samples where the model is doubtful
        """
        n_classes = self.predictions.shape[1]
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["utility"] = np.zeros(len(samples))

        quantiles_columns = []
        for c in np.arange(n_classes):
            samples[f"quantile_{c}"] = pd.qcut(
                samples[c], self.quantiles, duplicates="drop", labels=False
            )
            quantiles_columns.append(f"quantile_{c}")

        high_quantile = len(self.quantiles) - 1
        samples["utility"] = (samples[quantiles_columns].isin([0, high_quantile])).sum(
            axis=1
        )
        samples["utility"] = samples["utility"] / samples["utility"].max()

        return samples["utility"].values

    def _most_confident_classes(self) -> np.ndarray:
        """
        This sampling strategy takes samples where the model is very sure about presence of multiple classes (all highest quantile),
        regardless of which/how many classes, and sure about the absence of the rest of the classes (lowest quantile).
        The difference with best_multiclass is that it gives priority to samples with multiple positive presence of classes (the most classes the best)
        """
        # define the quantiles
        n_classes = self.predictions.shape[1]
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["utility"] = np.zeros(len(samples))

        quantiles_columns = []
        for c in np.arange(n_classes):
            samples[f"quantile_{c}"] = pd.qcut(
                samples[c], self.quantiles, duplicates="drop", labels=False
            )
            quantiles_columns.append(f"quantile_{c}")

        high_quantile = len(self.quantiles) - 1
        samples["utility"] = (samples[quantiles_columns] == high_quantile).sum(axis=1)
        samples["utility"] = samples["utility"] / samples["utility"].max()

        # Alternatively, it could also be done with probabs directly
        utility = unlabeled_predictions.sum(axis=1)
        utility = utility / utility.max()

        return samples["utility"].values

    def _balance_class_by_clusters(self) -> np.ndarray:
        # Normalise embeddings before clustering
        n_classes = self.predictions.shape[1]
        if self.clusters is None:
            embeddings_norm = normalize(self.embeddings, norm="l2")
            kmeans = KMeans(
                n_clusters=n_classes, random_state=self.rng.integers(80), n_init="auto"
            )
            self.clusters = kmeans.fit_predict(embeddings_norm)

        # For each cluster check labels and confidences
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["cluster"] = self.clusters[self.unlabeled_indices]
        samples["utility"] = 0

        known_samples = pd.DataFrame(index=self.labeled_indices, data=self.labels)
        known_samples["cluster"] = self.clusters[self.labeled_indices]

        sampled_classes = {}
        unsampled_classes = []
        for l in np.arange(self.labels.shape[1]):
            n_samples = self.labels[:, l].sum()
            if n_samples > 0:
                sampled_classes[l] = n_samples
            else:
                unsampled_classes.append(l)

        to_select_n_samples = int(np.percentile(list(sampled_classes.values()), 10))

        # Find the noise cluster, we don't want to sample from there
        possible_noise_labels = ["nan", None, "no_call", np.nan, "NaN"]
        noise_indexes = []
        for p in possible_noise_labels:
            if p in self.label_to_idx.keys():
                noise_indexes.append(self.label_to_idx[p])

        noise_clusters = []
        for c, cluster_samples in known_samples.groupby("cluster"):
            if len(cluster_samples) > 2:
                proportion_noise_samples = (
                    cluster_samples[noise_indexes].sum(axis=1) >= 1
                ).sum() / (
                    cluster_samples[self.label_to_idx.values()].sum(axis=1) >= 1
                ).sum()
                if proportion_noise_samples > 0.6:
                    noise_clusters.append(c)

        # first step, check if there are clusters which are NOT sampled:
        sampled_clusters = set(known_samples.cluster.unique())
        total_clusters = set(samples.cluster.unique())
        unsampled_clusters = total_clusters - sampled_clusters
        if len(unsampled_clusters) > 0:
            for c in unsampled_clusters:
                selected_samples_cluster = samples.loc[samples.cluster == c].sample(
                    int(to_select_n_samples)
                )
                samples.loc[selected_samples_cluster.index, "utility"] = 1

        # Then try to sample the classes which are not sampled by selecting the samples with higher confidence
        # for u_class in unsampled_classes:
        #     candidate_samples = samples.loc[samples.utility == 0]
        #     selected_samples = candidate_samples.sort_values(by=u_class).iloc[:median_num_samples+1]
        #     samples.loc[selected_samples.index, 'utility'] = 1

        # then check which classes are the least sampled and sample in the corresponding clusters
        n_left_to_sample = self.n_samples - samples.utility.sum()

        candidates_samples = samples.loc[~samples.cluster.isin(noise_clusters)]
        candidates_samples = candidates_samples.loc[candidates_samples.utility == 0]

        if n_left_to_sample > 0:
            max_num_samples = max(list(sampled_classes.values()))
            left_for_balance = (
                max_num_samples - np.array(list(sampled_classes.values()))
            ).sum()
            # all the classes are already sampled, lets check which clusters have potentially samples of that class and sample randomly there
            for class_idx, n_samples in sampled_classes.items():
                n_to_sample = np.ceil(
                    ((max_num_samples - n_samples) / left_for_balance)
                    * n_left_to_sample
                )
                if (n_to_sample > 0) and (len(candidates_samples) > 0):
                    clusters_with_class = known_samples.loc[
                        known_samples[class_idx] == 1
                    ].cluster.unique()
                    samples_in_selected_clusters = candidates_samples.loc[
                        candidates_samples.cluster.isin(clusters_with_class)
                    ]
                    if len(samples_in_selected_clusters) > 0:
                        selected_samples_clusters = samples_in_selected_clusters.sample(
                            int(n_to_sample)
                        )  # this could also be based on confidence
                        samples.loc[selected_samples_clusters.index, "utility"] = 0.8

        print((samples["utility"] > 0).sum())
        return samples["utility"].values

    def _sample_from_dense_clusters(self) -> np.ndarray:
        # Normalise embeddings before clustering
        n_classes = self.predictions.shape[1]
        if self.clusters is None:
            embeddings_norm = normalize(self.embeddings, norm="l2")
            kmeans = KMeans(
                n_clusters=n_classes, random_state=self.rng.integers(80), n_init="auto"
            )
            self.clusters = kmeans.fit_predict(embeddings_norm)

        # For each cluster check labels and confidences
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["cluster"] = self.clusters[self.unlabeled_indices]
        samples["utility"] = 0

        known_samples = pd.DataFrame(index=self.labeled_indices, data=self.labels)
        known_samples["cluster"] = self.clusters[self.labeled_indices]

        # Find the noise cluster, we don't want to sample from there
        possible_noise_labels = ["nan", None, "no_call", np.nan, "NaN"]
        not_noise_columns = self.label_to_idx.copy()
        for p in possible_noise_labels:
            if p in self.label_to_idx.keys():
                not_noise_columns.pop(p)

        # first step, check if there are clusters which are NOT sampled:
        sampled_classes = {}

        for l in np.arange(self.labels.shape[1]):
            n_samples = self.labels[:, l].sum()
            if n_samples > 0:
                sampled_classes[l] = n_samples

        to_select_n_samples = int(np.percentile(list(sampled_classes.values()), 25))

        # first step, check if there are clusters which are NOT sampled:
        sampled_clusters = set(known_samples.cluster.unique())
        total_clusters = set(samples.cluster.unique())
        unsampled_clusters = total_clusters - sampled_clusters
        if len(unsampled_clusters) > 0:
            for c in unsampled_clusters:
                selected_samples_cluster = samples.loc[samples.cluster == c].sample(
                    int(to_select_n_samples)
                )
                samples.loc[selected_samples_cluster.index, "utility"] = 1

        n_left_to_sample = self.n_samples - samples.utility.sum()
        candidates_samples = samples.loc[samples.utility == 0]
        total_clusters = set(candidates_samples.cluster.unique())

        cluster_importance = {}
        for c in total_clusters:
            known_samples_cluster = known_samples.loc[known_samples.cluster == c]
            n_pos_detections_cluster = (
                known_samples_cluster[not_noise_columns.values()].sum(axis=1) > 1
            ).sum()
            # cluster_importance[c] = n_pos_detections_cluster / len(known_samples_cluster)
            cluster_importance[c] = (
                n_pos_detections_cluster / (samples.cluster == c).sum()
            )

        for c, samples_in_cluster in candidates_samples.groupby("cluster"):
            n_to_sample = np.ceil(
                (
                    cluster_importance[c]
                    / np.array(list(cluster_importance.values())).sum()
                )
                * n_left_to_sample
            )
            if n_to_sample is not None:
                selected_samples_clusters = samples_in_cluster.sample(
                    int(n_to_sample)
                )  # this could also be based on confidence
                samples.loc[selected_samples_clusters.index, "utility"] = 1

        return samples["utility"].values

    def _balance_class_by_confidence(self) -> np.ndarray:
        # For each cluster check labels and confidences
        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["utility"] = 0

        sampled_classes = {}
        for l in np.arange(self.labels.shape[1]):
            n_samples = self.labels[:, l].sum()
            sampled_classes[l] = n_samples

        max_num_samples = max(list(sampled_classes.values()))
        left_for_balance = np.array(list(sampled_classes.values())).sum()

        # Then try to sample the classes which are not sampled by selecting the samples with higher confidence
        for u_class, n_samples in sampled_classes.items():
            n_to_sample = np.ceil(
                ((max_num_samples - n_samples) / left_for_balance) * self.n_samples
            )

            candidate_samples = samples.loc[samples.utility == 0]
            selected_samples = candidate_samples.sort_values(by=u_class).iloc[
                : int(n_to_sample) + 1
            ]
            samples.loc[selected_samples.index, "utility"] = 1

        return samples["utility"].values

    def _bald(self) -> np.ndarray:
        # self.mc_predictions: (n_passes, n_samples, n_classes)

        if self.mc_predictions is None:
            raise ValueError(
                "MC predictions unavailable, this method requires multiple mc_dropout_passes to be set"
            )

        mc = self.mc_predictions[
            :, self.unlabeled_indices, :
        ]  # (n_passes, n_unlabeled, n_classes)
        mean_p = mc.mean(axis=0)  # (n_unlabeled, n_classes)
        H_mean = -np.sum(mean_p * np.log(mean_p + 1e-8), axis=1)  # predictive entropy
        mean_H = -np.mean(
            np.sum(mc * np.log(mc + 1e-8), axis=2), axis=0
        )  # expected entropy
        return H_mean - mean_H  # BALD score

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if len(scores) == 0:
            return np.array([], dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        if np.isclose(max_v, min_v):
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_v) / (max_v - min_v)

    @staticmethod
    def _labels_to_prob_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert single-label or multi-label targets to a probability-like matrix."""
        if labels.ndim == 2:
            return labels.astype(np.float32, copy=False)

        one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        idx = labels.astype(np.int64, copy=False)
        one_hot[np.arange(labels.shape[0]), idx] = 1.0
        return one_hot

    # def _margin_multilabel(self) -> np.ndarray:
    #     """
    #     Marginal query for multi-label classification:
    #     utility = 1 - 2 * min(|p - 0.5|) for each sample.
    #     """
    #     if self.predictions is None:
    #         raise ValueError("margin_multilabel requires predictions")

    #     unlabeled_probs = self.predictions[self.unlabeled_indices]
    #     margins = np.min(np.abs(unlabeled_probs - 0.5), axis=1)
    #     utility = np.clip(1.0 - 2.0 * margins, 0.0, 1.0).astype(np.float32)
    #     return utility

    def _coreset_farthest(self) -> np.ndarray:
        """
        Coreset farthest query with FAISS acceleration:
        1) approximate pool subsampling
        2) FAISS HNSW nearest-anchor distance
        3) far-distance preselection
        4) optional FAISS PCA + farthest-first refinement
        """
        if self.embeddings is None:
            raise ValueError("coreset_farthest requires embeddings")

        unlabeled = np.asarray(self.unlabeled_indices, dtype=int)
        k = min(self.n_samples, len(unlabeled))
        utility = np.zeros(len(unlabeled), dtype=np.float32)
        if len(unlabeled) == 0 or k <= 0:
            return utility

        labeled = np.asarray(self.labeled_indices, dtype=int)
        if len(labeled) == 0:
            return self._random()

        approx_pool_size = 80000
        anchor_limit = 30000
        preselect_factor = 8
        pca_dim = 64

        pool = _sample_pool(unlabeled, approx_pool_size, self.rng)
        x_pool = self.embeddings[pool].astype(np.float32, copy=False)
        if len(pool) == 0:
            return utility

        if len(labeled) > anchor_limit:
            labeled = self.rng.choice(labeled, size=anchor_limit, replace=False)
        x_labeled = self.embeddings[labeled].astype(np.float32, copy=False)

        anchor_index = _build_hnsw_index(x_labeled)
        dists, _ = anchor_index.search(x_pool, 1)
        dists = dists.reshape(-1)

        # Base utility for the sampled pool from anchor distances.
        pool_utility = 0.95 * self._normalize(dists)
        unlabeled_pos = {idx: pos for pos, idx in enumerate(unlabeled)}
        for pool_pos, global_idx in enumerate(pool):
            utility[unlabeled_pos[int(global_idx)]] = pool_utility[pool_pos]

        preselect = min(len(pool), max(k, k * preselect_factor))
        far_order = np.argsort(-dists)[:preselect]
        picked_indices = pool[far_order]
        x_picked = x_pool[far_order]

        x_proj = _project_with_pca(x_picked, out_dim=pca_dim)
        if x_proj.shape[0] == 0:
            return utility

        target_k = min(k, x_proj.shape[0])
        selected_local = np.empty((target_k,), dtype=int)
        min_dist = np.full((x_proj.shape[0],), np.inf, dtype=np.float32)

        seed_choice = int(self.rng.integers(0, x_proj.shape[0]))
        selected_local[0] = seed_choice
        seed_vec = x_proj[seed_choice]
        min_dist = np.minimum(min_dist, np.sum((x_proj - seed_vec) ** 2, axis=1))
        min_dist[seed_choice] = -np.inf

        selected_count = 1
        for _ in range(1, target_k):
            nxt = int(np.argmax(min_dist))
            if not np.isfinite(min_dist[nxt]):
                break
            selected_local[selected_count] = nxt
            selected_count += 1
            nxt_vec = x_proj[nxt]
            min_dist = np.minimum(min_dist, np.sum((x_proj - nxt_vec) ** 2, axis=1))
            min_dist[nxt] = -np.inf

        selected_indices = picked_indices[selected_local[:selected_count]]
        for rank, global_idx in enumerate(selected_indices):
            utility[unlabeled_pos[int(global_idx)]] = 1.0 - rank * 1e-6

        return utility.astype(np.float32)

    def _nn_disagreement(self) -> np.ndarray:
        """
        Nearest-neighbor disagreement query.
        Uses pool subsampling + FAISS HNSW neighbors.
        Disagreement is mean absolute difference between model probabilities and
        neighborhood label distribution estimated from labeled samples.
        """
        if self.embeddings is None or self.predictions is None:
            raise ValueError("nn_disagreement requires embeddings and predictions")
        if self.labels is None:
            return self._random()

        unlabeled = np.asarray(self.unlabeled_indices, dtype=int)
        k = min(self.n_samples, len(unlabeled))
        utility = np.zeros(len(unlabeled), dtype=np.float32)
        if len(unlabeled) == 0:
            return utility

        labeled = np.asarray(self.labeled_indices, dtype=int)
        if len(labeled) == 0:
            return self._random()

        # Keep pool size fixed for now (same style as coreset_farthest).
        approx_pool_size = 80000
        nn_train_limit = 50000
        n_neighbors = 15

        pool = _sample_pool(unlabeled, approx_pool_size, self.rng)
        if len(pool) == 0 or k <= 0:
            return utility

        if len(labeled) > nn_train_limit:
            labeled = self.rng.choice(labeled, size=nn_train_limit, replace=False)

        x_pool = self.embeddings[pool].astype(np.float32, copy=False)
        x_labeled = self.embeddings[labeled].astype(np.float32, copy=False)

        model_probs = self.predictions[pool]
        if model_probs.ndim == 1:
            model_probs = model_probs[:, None]
        num_classes = model_probs.shape[1]

        # self.labels is a dense array aligned with self.labeled_indices (same sorted order).
        # Use searchsorted to map the (possibly subsampled) global indices in `labeled` back
        # to their positions in the dense labels array.
        labeled_indices_sorted = np.asarray(self.labeled_indices, dtype=int)
        labeled_positions = np.searchsorted(labeled_indices_sorted, labeled)
        label_probs = self._labels_to_prob_matrix(np.asarray(self.labels), num_classes)
        labeled_targets = label_probs[labeled_positions]

        index = _build_hnsw_index(x_labeled)
        nn_k = min(n_neighbors, len(labeled))
        _, nbr_idx = index.search(x_pool, nn_k)
        nn_probs = labeled_targets[nbr_idx].mean(axis=1)

        if nn_probs.shape[1] != model_probs.shape[1]:
            common_dim = min(nn_probs.shape[1], model_probs.shape[1])
            logger.warning(
                "Probability dimension mismatch in nn_disagreement: "
                f"model={model_probs.shape[1]}, nn={nn_probs.shape[1]}. "
                f"Using first {common_dim} dims."
            )
            nn_probs = nn_probs[:, :common_dim]
            model_probs = model_probs[:, :common_dim]

        disagreement = np.mean(np.abs(model_probs - nn_probs), axis=1)
        pool_utility = self._normalize(disagreement)

        unlabeled_pos = {idx: pos for pos, idx in enumerate(unlabeled)}
        for pool_pos, global_idx in enumerate(pool):
            utility[unlabeled_pos[int(global_idx)]] = pool_utility[pool_pos]

        order = np.argsort(-disagreement)
        selected_indices = pool[order[:k]]
        for rank, global_idx in enumerate(selected_indices):
            utility[unlabeled_pos[int(global_idx)]] = 1.0 - rank * 1e-6

        return utility.astype(np.float32)

    def _margin_multilabel(self) -> np.ndarray:
        """
        Margin sampling with mean aggregation for multilabel classification.

        For each sample, computes the mean distance of each label probability from
        the decision boundary (|p - 0.5|) and averages across labels. Samples
        closest to the boundary (lowest mean margin) are most uncertain.

        This is the multilabel analogue of:
            skactiveml.pool.UncertaintySampling(method='margin_sampling')
        which cannot be used directly here because it requires a classifier object
        to call predict_proba -- unnecessary when predictions are already available.

        scikit-activeml reference:
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.UncertaintySampling.html

        Returns:
            utility: Normalized scores [0, 1] where 1 = on the decision boundary
        """
        if self.predictions is None:
            raise ValueError("margin_multilabel requires predictions")

        unlabeled_preds = self.predictions[self.unlabeled_indices]

        # Per-label distance from the decision boundary: |p - 0.5| in [0, 0.5]
        # Mean across labels: small value => high uncertainty
        mean_margin = np.mean(np.abs(unlabeled_preds - 0.5), axis=1)

        # Normalize to [0, 1]: 0.5 is the max possible mean_margin
        utility = (1.0 - (mean_margin / 0.5)).astype(np.float32)

        logger.info(
            f"margin_multilabel - mean_margin min: {mean_margin.min():.4f}, max: {mean_margin.max():.4f}"
        )
        return utility

    def _sklearn_coreset_no_noise(self) -> np.ndarray:
        """
        Greedy k-center (CoreSet) sampling using scikit-activeml. This modification igores clusters with high density of noise

        Selects a diverse set of samples by minimising the maximum distance from
        any unlabeled point to its nearest selected (or already-labeled) point.

        scikit-activeml reference:
            skactiveml.pool.CoreSet
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.CoreSet.html

        Returns:
            utility: Normalized min-distance-to-labeled scores [0, 1]
                     where 1 = farthest from any labeled point
        """
        from skactiveml.utils import MISSING_LABEL

        if self.embeddings is None:
            raise ValueError("sklearn_coreset requires embeddings")

        n_total = self.embeddings.shape[0]
        n_samples = min(self.n_samples, len(self.unlabeled_indices))

        n_classes = self.predictions.shape[1]
        if self.clusters is None:
            embeddings_norm = normalize(self.embeddings, norm="l2")
            kmeans = KMeans(
                n_clusters=n_classes, random_state=self.rng.integers(80), n_init="auto"
            )
            self.clusters = kmeans.fit_predict(embeddings_norm)

        known_samples = pd.DataFrame(index=self.labeled_indices, data=self.labels)
        known_samples["cluster"] = self.clusters[self.labeled_indices]

        unlabeled_predictions = self.predictions[self.unlabeled_indices, :]
        samples = pd.DataFrame(index=self.unlabeled_indices, data=unlabeled_predictions)
        samples["cluster"] = self.clusters[self.unlabeled_indices]

        # Find the noise cluster, we don't want to sample from there
        possible_noise_labels = ["nan", None, "no_call", np.nan, "NaN"]
        noise_indexes = []
        for p in possible_noise_labels:
            if p in self.label_to_idx.keys():
                noise_indexes.append(self.label_to_idx[p])

        noise_clusters = []
        for c, cluster_samples in known_samples.groupby("cluster"):
            if len(cluster_samples) > 2:
                proportion_noise_samples = (
                    cluster_samples[noise_indexes].sum(axis=1) >= 1
                ).sum() / (
                    cluster_samples[self.label_to_idx.values()].sum(axis=1) >= 1
                ).sum()
                if proportion_noise_samples > 0.6:
                    noise_clusters.append(c)

        print(noise_clusters)
        candidate_indices = samples.loc[~samples.cluster.isin(noise_clusters)]

        # Cold-start fallback: no labeled samples means no anchor for distance computation.
        if len(self.labeled_indices) == 0:
            logger.warning(
                "sklearn_coreset: no labeled samples, falling back to random"
            )
            return self._random()

        # Build y: labeled samples get a dummy label (0), everything else (unlabeled
        # AND validation) gets MISSING_LABEL so only true labeled samples serve as anchors.
        # CoreSet only uses y to distinguish labeled from unlabeled -- label values are ignored.
        y = np.full(n_total, MISSING_LABEL, dtype=float)
        y[self.labeled_indices] = 0

        strategy = CoreSet(missing_label=MISSING_LABEL)

        selected_indices, utilities = strategy.query(
            X=self.embeddings,
            y=y,
            candidates=np.array(candidate_indices.index),
            batch_size=n_samples,
            return_utilities=True,
        )

        # utilities[0] = initial min-distance-to-labeled for every candidate -- use for
        # visualisation / relative ranking of non-selected samples.
        util_scores = utilities[0][self.unlabeled_indices]
        utility = self._normalize(np.clip(util_scores, 0, None)) * 0.99

        # Override utility to 1.0 for the samples scikit-activeml actually chose via its
        # greedy k-center algorithm. This ensures select() reproduces the greedy selection
        # rather than a naive top-K on the initial distances (which clusters picks together).
        unlabeled_pos = {idx: pos for pos, idx in enumerate(self.unlabeled_indices)}
        for idx in selected_indices:
            if idx in unlabeled_pos:
                utility[unlabeled_pos[idx]] = 1.0

        logger.info(
            f"sklearn_coreset - selected {len(selected_indices)} samples via greedy k-center"
        )
        return utility.astype(np.float32)

    def _sklearn_coreset_far_from_noise(self) -> np.ndarray:
        """
        Greedy k-center (CoreSet) sampling using scikit-activeml.

        Selects a diverse set of samples by minimising the maximum distance from
        any unlabeled point to its nearest selected (or already-labeled) point.

        scikit-activeml reference:
            skactiveml.pool.CoreSet
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.CoreSet.html

        Returns:
            utility: Normalized min-distance-to-labeled scores [0, 1]
                     where 1 = farthest from any labeled point
        """
        from skactiveml.utils import MISSING_LABEL

        if self.embeddings is None:
            raise ValueError("sklearn_coreset requires embeddings")

        n_total = self.embeddings.shape[0]
        n_samples = min(self.n_samples, len(self.unlabeled_indices))

        # Cold-start fallback: no labeled samples means no anchor for distance computation.
        if len(self.labeled_indices) == 0:
            logger.warning(
                "sklearn_coreset: no labeled samples, falling back to random"
            )
            return self._random()

        # Build y: labeled samples get a dummy label (0), everything else (unlabeled
        # AND validation) gets MISSING_LABEL so only true labeled samples serve as anchors.
        # CoreSet only uses y to distinguish labeled from unlabeled -- label values are ignored.
        y = np.full(n_total, MISSING_LABEL, dtype=float)
        y[self.labeled_indices] = 0

        # Find the noise cluster, we don't want to sample from there
        possible_noise_labels = ["nan", None, "no_call", np.nan, "NaN"]
        noise_indices = []
        for p in possible_noise_labels:
            if p in self.label_to_idx.keys():
                noise_indices.append(self.label_to_idx[p])

        strategy = CoreSetFarFromNoise(missing_label=MISSING_LABEL)

        labeled_noise_indices = np.array(self.labeled_indices)[
            (self.labels[:, noise_indices] == 1).any(axis=1)
        ]
        selected_indices, utilities = strategy.query(
            X=self.embeddings,
            y=y,
            labeled_noise_indices=labeled_noise_indices,
            unlabeled_indices=self.unlabeled_indices,
            candidates=np.array(self.unlabeled_indices),
            batch_size=n_samples,
            return_utilities=True,
            noise_indices=noise_indices,
        )

        # utilities[0] = initial min-distance-to-labeled for every candidate -- use for
        # visualisation / relative ranking of non-selected samples.
        util_scores = utilities[0][self.unlabeled_indices]
        utility = self._normalize(np.clip(util_scores, 0, None)) * 0.99

        # Override utility to 1.0 for the samples scikit-activeml actually chose via its
        # greedy k-center algorithm. This ensures select() reproduces the greedy selection
        # rather than a naive top-K on the initial distances (which clusters picks together).
        unlabeled_pos = {idx: pos for pos, idx in enumerate(self.unlabeled_indices)}
        for idx in selected_indices:
            if idx in unlabeled_pos:
                utility[unlabeled_pos[idx]] = 1.0

        logger.info(
            f"sklearn_coreset - selected {len(selected_indices)} samples via greedy k-center"
        )
        return utility.astype(np.float32)

    def _sklearn_coreset(self) -> np.ndarray:
        """
        Greedy k-center (CoreSet) sampling using scikit-activeml.

        Selects a diverse set of samples by minimising the maximum distance from
        any unlabeled point to its nearest selected (or already-labeled) point.

        scikit-activeml reference:
            skactiveml.pool.CoreSet
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.CoreSet.html

        Returns:
            utility: Normalized min-distance-to-labeled scores [0, 1]
                     where 1 = farthest from any labeled point
        """

        if self.embeddings is None:
            raise ValueError("sklearn_coreset requires embeddings")

        n_total = self.embeddings.shape[0]
        n_samples = min(self.n_samples, len(self.unlabeled_indices))

        # Cold-start fallback: no labeled samples means no anchor for distance computation.
        if len(self.labeled_indices) == 0:
            logger.warning(
                "sklearn_coreset: no labeled samples, falling back to random"
            )
            return self._random()

        # Build y: labeled samples get a dummy label (0), everything else (unlabeled
        # AND validation) gets MISSING_LABEL so only true labeled samples serve as anchors.
        # CoreSet only uses y to distinguish labeled from unlabeled -- label values are ignored.
        y = np.full(n_total, MISSING_LABEL, dtype=float)
        y[self.labeled_indices] = 0

        strategy = CoreSet(missing_label=MISSING_LABEL)

        selected_indices, utilities = strategy.query(
            X=self.embeddings,
            y=y,
            candidates=np.array(self.unlabeled_indices),
            batch_size=n_samples,
            return_utilities=True,
        )

        # utilities[0] = initial min-distance-to-labeled for every candidate -- use for
        # visualisation / relative ranking of non-selected samples.
        util_scores = utilities[0][self.unlabeled_indices]
        utility = self._normalize(np.clip(util_scores, 0, None)) * 0.99

        # Override utility to 1.0 for the samples scikit-activeml actually chose via its
        # greedy k-center algorithm. This ensures select() reproduces the greedy selection
        # rather than a naive top-K on the initial distances (which clusters picks together).
        unlabeled_pos = {idx: pos for pos, idx in enumerate(self.unlabeled_indices)}
        for idx in selected_indices:
            if idx in unlabeled_pos:
                utility[unlabeled_pos[idx]] = 1.0

        logger.info(
            f"sklearn_coreset - selected {len(selected_indices)} samples via greedy k-center"
        )
        return utility.astype(np.float32)

    def _sklearn_typiclust(self) -> np.ndarray:
        """
        TypiClust sampling using scikit-activeml.

        Clusters the embedding space (KMeans) then, for each cluster without a
        labeled sample, selects the most typical (highest-density) unlabeled point.
        Promotes coverage and typicality rather than outlier selection.

        scikit-activeml reference:
            skactiveml.pool.TypiClust
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.TypiClust.html

        Returns:
            utility: Normalized typicality scores [0, 1]
                     where 1 = most typical in its cluster
        """
        from skactiveml.pool import TypiClust
        from skactiveml.utils import MISSING_LABEL

        if self.embeddings is None:
            raise ValueError("sklearn_typiclust requires embeddings")

        n_total = self.embeddings.shape[0]
        n_samples = min(self.n_samples, len(self.unlabeled_indices))

        # Project to a lower-dimensional space before clustering.
        # KMeans is slow and noisy in high dimensions; PCA to 64-D gives a large
        # speed-up with minimal loss of cluster structure (same as _coreset_farthest).
        pca_dim = 64
        X = _project_with_pca(
            self.embeddings.astype(np.float32, copy=False), out_dim=pca_dim
        )

        # Build y: labeled samples get a dummy label (0), everything else (unlabeled
        # AND validation) gets MISSING_LABEL so only true labeled samples serve as anchors.
        y = np.full(n_total, MISSING_LABEL, dtype=float)
        y[self.labeled_indices] = 0

        strategy = TypiClust(missing_label=MISSING_LABEL)

        selected_indices, utilities = strategy.query(
            X=X,
            y=y,
            candidates=np.array(self.unlabeled_indices),
            batch_size=n_samples,
            return_utilities=True,
        )

        # utilities[0] = typicality scores; -inf means the cluster is already covered by a
        # labeled sample so the point should not be selected -- map to 0 for display.
        util_scores = utilities[0][self.unlabeled_indices]
        util_scores = np.where(np.isfinite(util_scores), util_scores, 0.0)
        utility = self._normalize(util_scores) * 0.99

        # Override utility to 1.0 for the samples TypiClust actually chose (one per
        # uncovered cluster). This ensures select() reproduces TypiClust's coverage-aware
        # selection rather than a naive top-K that may pick multiple from the same cluster.
        unlabeled_pos = {idx: pos for pos, idx in enumerate(self.unlabeled_indices)}
        for idx in selected_indices:
            if idx in unlabeled_pos:
                utility[unlabeled_pos[idx]] = 1.0

        logger.info(
            f"sklearn_typiclust - typicality min: {util_scores.min():.4f}, max: {util_scores.max():.4f}"
        )
        return utility.astype(np.float32)


class WarmupStrategy:
    """
    Warmup sampling strategies for pre-training initialisation.

    Warmup selects an initial labeled set before any model training, so methods
    only have access to raw embeddings -- no model predictions are available yet.
    """

    def __init__(
        self,
        method: str = "density",
        n_samples: int = 0,
        num_classes: int = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            method: Warmup method ('density', 'random', 'custom')
            n_samples: Number of samples to select
            random_state: Optional random seed for reproducibility
        """
        self.method = method
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.rng = np.random.default_rng(random_state)

        available_methods = ["density", "random", "custom"]
        if method not in available_methods:
            raise ValueError(
                f"Unknown warmup strategy: '{method}'. "
                f"Available: {available_methods}"
            )

        self._method_map = {
            "density": self._density,
            "random": self._random,
            "custom": self._custom,
            "kmeans": self._kmeans,
            "eigenvalues": self._eigenvalues,
            "metadata": self._metadata,
        }

        # Set by select() before calling method implementations
        self.candidate_indices: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None

        logger.info(
            f"Initialized WarmupStrategy with method='{method}' and n_samples={n_samples}"
        )

    def select(
        self, candidate_indices: np.ndarray, embeddings: np.ndarray
    ) -> List[int]:
        """
        Select warmup samples from the candidate pool.

        Args:
            candidate_indices: Sorted array of global sample indices (all non-validation samples)
            embeddings: Full embeddings array of shape (n_total_samples, embedding_dim)

        Returns:
            List of selected global indices (length <= n_samples)
        """
        n = min(self.n_samples, len(candidate_indices))
        if n <= 0:
            return []

        self.candidate_indices = candidate_indices
        self.embeddings = embeddings

        utility = np.asarray(self._method_map[self.method](), dtype=np.float32)
        top_local = np.argsort(utility)[-n:]
        selected = candidate_indices[top_local].tolist()

        logger.info(f"WarmupStrategy '{self.method}' selected {len(selected)} samples")
        return selected

    def _density(self) -> np.ndarray:
        """
        High-density warmup: selects samples surrounded by many neighbours.

        Returns:
            utility: KNN-density scores for each candidate
        """
        k = min(20, len(self.candidate_indices) - 1)
        return densityEstimation(
            embeddings=self.embeddings[self.candidate_indices],
            method="knn",
            k=k,
            beta=1,
        )

    def _random(self) -> np.ndarray:
        """
        Random warmup: uniform random utility scores.

        Returns:
            utility: Random scores for each candidate
        """
        return self.rng.random(len(self.candidate_indices)).astype(np.float32)

    def _custom(self) -> np.ndarray:
        """
        Custom warmup strategy template.

        INSTRUCTIONS FOR IMPLEMENTING CUSTOM WARMUP:
        =============================================

        1. This method computes utility scores for all candidate samples.

        2. Scores should be in [0, 1] where:
           - 1.0 = highest priority for initial annotation
           - 0.0 = lowest priority

        3. Available instance attributes (set by select()):
           - self.candidate_indices: sorted array of global indices for all
                                     non-validation samples
           - self.embeddings:        full embedding array (n_total x embedding_dim);
                                     access candidate embeddings via
                                     self.embeddings[self.candidate_indices]

        Note: No model predictions exist at warmup time -- only raw embeddings
        are available.

        Returns:
            utility: Array of utility scores for candidates, shape (n_candidates,)
        """
        # TODO: Implement your custom warmup logic here
        logger.warning(
            "Custom warmup not implemented, falling back to density sampling"
        )
        return self._density()

    def _kmeans(self) -> np.ndarray:
        """
        Custom warmup strategy template.

        INSTRUCTIONS FOR IMPLEMENTING CUSTOM WARMUP:
        =============================================

        1. This method computes utility scores for all candidate samples.

        2. Scores should be in [0, 1] where:
           - 1.0 = highest priority for initial annotation
           - 0.0 = lowest priority

        3. Available instance attributes (set by select()):
           - self.candidate_indices: sorted array of global indices for all
                                     non-validation samples
           - self.embeddings:        full embedding array (n_total x embedding_dim);
                                     access candidate embeddings via
                                     self.embeddings[self.candidate_indices]

        Note: No model predictions exist at warmup time -- only raw embeddings
        are available.

        Returns:
            utility: Array of utility scores for candidates, shape (n_candidates,)
        """
        return KMeansEstimation(
            embeddings=self.embeddings[self.candidate_indices],
            num_classes=self.n_classes,
            random_state=self.rng.integers(80),
        )

    def _eigenvalues(self) -> np.ndarray:
        """
        Custom warmup strategy template.

        INSTRUCTIONS FOR IMPLEMENTING CUSTOM WARMUP:
        =============================================

        1. This method computes utility scores for all candidate samples.

        2. Scores should be in [0, 1] where:
           - 1.0 = highest priority for initial annotation
           - 0.0 = lowest priority

        3. Available instance attributes (set by select()):
           - self.candidate_indices: sorted array of global indices for all
                                     non-validation samples
           - self.embeddings:        full embedding array (n_total x embedding_dim);
                                     access candidate embeddings via
                                     self.embeddings[self.candidate_indices]

        Note: No model predictions exist at warmup time -- only raw embeddings
        are available.

        Returns:
            utility: Array of utility scores for candidates, shape (n_candidates,)
        """
        return uniformEmbeddingSampling(
            embeddings=self.embeddings[self.candidate_indices],
            n_samples=self.n_samples,
            random_state=self.rng.integers(80),
        )

    def _metadata(self) -> np.ndarray:
        """
        Custom warmup strategy template.

        INSTRUCTIONS FOR IMPLEMENTING CUSTOM WARMUP:
        =============================================

        1. This method computes utility scores for all candidate samples.

        2. Scores should be in [0, 1] where:
           - 1.0 = highest priority for initial annotation
           - 0.0 = lowest priority

        3. Available instance attributes (set by select()):
           - self.candidate_indices: sorted array of global indices for all
                                     non-validation samples
           - self.embeddings:        full embedding array (n_total x embedding_dim);
                                     access candidate embeddings via
                                     self.embeddings[self.candidate_indices]

        Note: No model predictions exist at warmup time -- only raw embeddings
        are available.

        Returns:
            utility: Array of utility scores for candidates, shape (n_candidates,)
        """
        # TODO: Implement your custom warmup logic here
        logger.warning(
            "Custom warmup not implemented, falling back to density sampling"
        )
        return self._density()


if __name__ == "__main__":

    print("=" * 60)
    print("SamplingStrategy - Unit Test on Dummy Data")
    print("=" * 60)

    # --- Dummy data setup ---
    N_SAMPLES = 20
    N_TOTAL = 200  # total samples in the pool
    N_UNLABELED = 160  # samples without labels
    N_LABELED = 40  # samples already labeled
    N_CLASSES = 8
    EMBED_DIM = 16
    RANDOM_STATE = 42

    rng = np.random.default_rng(RANDOM_STATE)

    all_indices = list(range(N_TOTAL))
    labeled_indices = all_indices[:N_LABELED]
    unlabeled_indices = all_indices[N_LABELED:]

    # Softmax-like predictions (rows sum to 1)
    raw = rng.random((N_TOTAL, N_CLASSES))

    # # Create a dummy prediction array with size N_TOTAL x N_CLASSES and values increasing by step from 0 to 1
    # predictions = np.arange(0, 1, 1/N_TOTAL, dtype=np.float32)
    # # Concatenate the predictions for each class
    # predictions = np.tile(predictions, (N_CLASSES, 1)).T

    # Create long-tail predictions using a Dirichlet distribution
    # Low concentration (alpha < 1) produces sparse, peaked distributions
    # mimicking a model that is often confident about one class
    alpha = np.ones(N_CLASSES) * 0.4
    raw_predictions = rng.dirichlet(alpha, size=N_TOTAL).astype(np.float32)
    predictions = raw_predictions

    embeddings = rng.standard_normal((N_TOTAL, EMBED_DIM)).astype(np.float32)
    labels = rng.integers(0, N_CLASSES, size=N_TOTAL)

    # Minimal metadata DataFrame
    metadata = pd.DataFrame(
        {
            "sample_id": all_indices,
            "split": ["labeled"] * N_LABELED + ["unlabeled"] * N_UNLABELED,
        }
    )

    # --- Methods to test ---
    method_to_test = "quantiles"
    print(f"\n--- Testing method: '{method_to_test}' ---")

    strategy = SamplingStrategy(
        method=method_to_test, n_samples=N_SAMPLES, random_state=RANDOM_STATE
    )

    selected, utility = strategy.select(
        unlabeled_indices=unlabeled_indices,
        predictions=predictions,
        embeddings=embeddings,
        labeled_indices=labeled_indices,
        labels=labels,
        metadata=metadata,
    )

    print(f"Selected indices : {selected}")
    print(f"Utility : {utility}")
    print(f"Prediction selected samples: {predictions[selected]}")

    print("\n" + "=" * 60)

    # Plot the distribution of samples and selected samples
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    class_names = [f"Class {i}" for i in range(N_CLASSES)]
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4 * N_CLASSES, 4), sharey=True)

    unlabeled_preds = predictions[unlabeled_indices]  # shape: (N_UNLABELED, N_CLASSES)
    selected_preds = predictions[selected]  # shape: (N_SAMPLES, N_CLASSES)

    for i, ax in enumerate(axes):
        data = unlabeled_preds[:, i]
        kde = gaussian_kde(data, bw_method="scott")
        x = np.linspace(data.min(), data.max(), 300)

        ax.plot(x, kde(x), color="steelblue", linewidth=2, label="Unlabeled pool")
        ax.fill_between(x, kde(x), alpha=0.15, color="steelblue")

        # Selected samples as dots on the x-axis (rug) at y=0
        sel_probs = selected_preds[:, i]
        ax.scatter(
            sel_probs,
            np.zeros_like(sel_probs),
            color="tomato",
            s=60,
            zorder=5,
            marker="|",
            linewidths=2,
            label="Selected",
        )

        ax.set_title(class_names[i])
        ax.set_xlabel("Predicted probability")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(f"Prediction density — method: '{method_to_test}'", fontsize=13)
    plt.tight_layout()
    plt.show()


class CoreSetFarFromNoise(CoreSet):
    def query(
        self,
        X,
        y,
        labeled_noise_indices,
        unlabeled_indices,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        noise_indices=None,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        _, mapping = self._transform_candidates(candidates, X, y)

        query_indices, utilities = self.k_greedy_center_noise(
            X,
            y,
            labeled_noise_indices,
            unlabeled_indices,
            batch_size,
        )

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices

    def k_greedy_center_noise(
        self,
        X,
        y,
        labeled_indices,
        unlabeled_indices,
        batch_size=1,
        n_new_cand=None,
    ):
        """
        An active learning method that greedily forms a batch to minimize the
        maximum distance to a cluster center among all unlabeled datapoints.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
        y : np.ndarray of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        batch_size : int, default=1
        The number of samples to be selected in one AL cycle.
        random_state : None or int or np.random.RandomState, default=None
        Random state for candidate selection.
        missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
        mapping : None or np.ndarray of shape (n_candidates,), default=None
        Index array that maps `candidates` to `X` (`candidates = X[mapping]`).
        n_new_cand : int or None, default=None
        The number of new candidates that are additionally added to `X`.
        Only used for the case, that in the query function with the shape of
        `candidates` is `(n_candidates, n_feature)`.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.

            - If `candidates` is `None` or of shape
            `(n_candidates,)`, the indexing refers to the samples in
            `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
            the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
            `(n_candidates,)`, the indexing refers to the samples in
            `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
            the indexing refers to the samples in `candidates`.
        """

        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be a integer")

        # initialize the utilities matrix with
        if n_new_cand is None:
            utilities = np.zeros(shape=(batch_size, X.shape[0]))
        elif isinstance(n_new_cand, int):
            if n_new_cand == len(unlabeled_indices):
                utilities = np.zeros(shape=(batch_size, n_new_cand))
            else:
                raise ValueError("n_new_cand must equal to the length of mapping array")
        else:
            raise TypeError("Only n_new_cand with type int is supported.")

        query_indices = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            if i == 0:
                update_dist = self.update_distances_noise(
                    X, labeled_indices, unlabeled_indices
                )
            else:
                latest_dist = utilities[i - 1]
                update_dist = self.update_distances_noise(
                    X=X,
                    cluster_centers=[query_indices[i - 1]],
                    mapping=unlabeled_indices,
                    latest_distance=latest_dist,
                )

            if n_new_cand is None:
                utilities[i] = update_dist
            else:
                utilities[i] = update_dist[unlabeled_indices]

            # select index
            query_indices[i] = rand_argmax(
                utilities[i], random_state=self.random_state_
            )[0]

        return query_indices, utilities

    def update_distances_noise(self, X, cluster_centers, mapping, latest_distance=None):
        """
        Update minimum distances by given cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled and
            unlabeled samples.
        cluster_centers : array-like of shape (n_cluster_centers)
            Indices of cluster centers.
        mapping : np.ndarray of shape (n_candidates, ), default=None
            Index array that maps `candidates` to `X` (`candidates = X[mapping]`).
        latest_distance : array-like of shape (n_samples) default None
            The distance between each sample and its nearest center. Used to
            speed up the computation of distances for the next selected sample.

        Returns
        -------
        result-dist : np.ndarray of shape (1, n_samples)
            - If there aren't any cluster centers existing, the default distance
            will be 0.
            - If there are some cluster center exist, the return will be the
            distance between each sample and its nearest center after each selected
            sample of the batch. In the case of cluster center the value will be
            `np.nan`.
            - For the case, that indices aren't in `mapping`, the corresponding
            value in `result-dist` will be also `np.nan`.
        """
        dist = np.zeros(shape=X.shape[0])

        if len(cluster_centers) > 0:
            cluster_center_feature = X[cluster_centers]
            _, dist = pairwise_distances_argmin_min(X, cluster_center_feature)

        if latest_distance is not None:
            sum_dist = np.nansum(latest_distance)
            latest_distance_tmp = latest_distance
            if sum_dist == 0:
                latest_distance_tmp = latest_distance.copy()
                latest_distance_tmp[latest_distance_tmp == 0] = np.inf
            l_distance = np.zeros(shape=X.shape[0])
            l_distance[mapping] = latest_distance_tmp[mapping]
            dist = np.minimum(l_distance, dist)

        result_dist = np.full(X.shape[0], np.nan)
        result_dist[mapping] = dist[mapping]
        result_dist[cluster_centers] = np.nan

        return result_dist
