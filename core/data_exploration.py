# %%
import pathlib

import numpy as np
import pandas as pd
from renumics import spotlight
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

whale_set = pathlib.Path("/mnt/fscompute_shared/biodcase_AL26/ATBFL_BASEAL")
bird_set = pathlib.Path("/mnt/fscompute_shared/biodcase_AL26/BirdSet_BASEAL")

output_path = pathlib.Path("/mnt/fscompute_shared/biodcase_AL26/")
all_sets = [bird_set]


all_detections = None
for dataset in all_sets:
    for db_path in dataset.glob("*"):
        if db_path.is_dir():
            if not db_path.joinpath("all_samples.pkl").exists():
                set_name = db_path.name
                metadata = pd.read_csv(db_path.joinpath("metadata.csv"))
                labels = pd.read_csv(db_path.joinpath("labels.csv"))

                detections = pd.merge(left=metadata, right=labels, on="filename")
                embeddings_path = db_path.joinpath("embeddings")

                embeddings = []
                selected_indices = []
                for i, row in tqdm(detections.iterrows(), total=len(metadata)):
                    if set_name != "ATBFL":
                        embedding_i_path = embeddings_path.joinpath(
                            "perch_v2", row.filename
                        )
                    else:
                        if row.dataset is np.nan:
                            continue
                        embedding_i_path = embeddings_path.joinpath(
                            "perch_v2",
                            row.dataset + "__" + row.filename.replace(".wav", ".npy"),
                        )
                    if embedding_i_path.exists():
                        embedding = np.load(embedding_i_path)
                        selected_indices.append(i)
                        embeddings.append(embedding)

                detections = detections.loc[selected_indices]
                detections["embedding"] = embeddings
                detections["set"] = set_name

                detections["noise"] = detections.label.isna()
                extra_labels = detections.label.str.split(";", expand=True)
                unique_labels = set()
                for c in extra_labels.columns:
                    unique_labels.update(set(extra_labels[c].unique()))

                for l in unique_labels:
                    if type(l) == str:
                        detections[l] = detections.label.str.contains(l)

                embeddings_norm = normalize(np.array(embeddings), norm="l2")
                kmeans = KMeans(
                    n_clusters=len(unique_labels), random_state=42, n_init="auto"
                )
                detections["clusters"] = kmeans.fit_predict(embeddings_norm)

                detections.to_pickle(db_path.joinpath("all_samples.pkl"))
            else:
                detections = pd.read_pickle(db_path.joinpath("all_samples.pkl"))

            detections.local_time = pd.to_datetime(detections.local_time)
            if all_detections is None:
                all_detections = detections
            else:
                all_detections = pd.concat([all_detections, detections])

# %%
import matplotlib.pyplot as plt

for s, set_samples in all_detections.groupby("set"):
    percentages_list = []
    for c, cluster_samples in set_samples.groupby("clusters"):
        percentage_noise = cluster_samples.noise.sum() / len(cluster_samples)
        print(s, c, percentage_noise)
        percentages_list.append(percentage_noise)
    plt.hist(percentages_list)
    plt.show()

dtype = {"embedding": spotlight.media.Embedding}
spotlight.show(
    all_detections, port=54426, host="127.0.0.1", dtype=dtype, no_browser=True
)

# %%
