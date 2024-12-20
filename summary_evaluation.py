import os
from functools import reduce
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from skimage.measure import label

ANN_ROOT = "data/annotations"
ANNOTATORS = ["anwai", "caro", "constantin", "luca", "marei"]
CONSENSUS_ROOT = "data/consensus_labels/proofread"
VERSION_TO_SPLIT = {
    "v1": 1, "v2": 1, "v3": 1, "v4": 1,
    "v5": 2, "v6": 2,
    "v7": 3, "v8": 3,
}


def _size_filter(labels, min_size):
    if min_size is not None:
        ids, sizes = np.unique(labels, return_counts=True)
        filter_ids = ids[sizes < min_size]
        labels[np.isin(labels, filter_ids)] = 0
    return labels


def _load_labels(version, name, im_name, min_size=None):
    # Cellpose
    if version in [4, 6, 8]:
        label_path = os.path.join(ANN_ROOT, f"v{version}", name, f"{im_name}_seg.npy")
        labels = np.load(label_path, allow_pickle=True).item()["masks"]
    else:
        label_path = os.path.join(ANN_ROOT, f"v{version}", name, f"{im_name}.tif")
        labels = imageio.imread(label_path)
    labels = label(labels, connectivity=1)
    labels = _size_filter(labels, min_size)
    return labels


def _get_n_objects(version, name, im_name, min_size):
    labels = _load_labels(version, name, im_name, min_size=min_size)
    n_objects = len(np.unique(labels)) - 1
    return n_objects


def _to_seconds(t):
    if isinstance(t, str):
        parts = [int(pp) for pp in t.split(":")]
        if len(parts) == 2:
            ann_time = parts[0] * 60 + parts[1]
        else:
            assert len(parts) == 3
            assert parts[0] == 0
            ann_time = parts[1] * 60 + parts[2]

    else:
        # The time interpretation is messy. It's sometimes interpreted as
        # HH:MM (which is incorrect), sometimes as MM:SS (which is correct).
        if t.hour == 0 and t.second == 0:
            ann_time = t.minute
        elif t.hour == 0:
            ann_time = t.minute * 60 + t.second
        else:
            ann_time = t.hour * 60 + t.minute

    return ann_time


def _to_hours(t):
    if isinstance(t, str):
        parts = [int(pp) for pp in t.split(":")]
        assert len(parts) == 3
        assert parts[0] == 0
        ann_time = parts[0] + parts[1] / 60.0

    else:
        ann_time = t.hour + t.minute / 60.0

    return ann_time


def evaluate_annotation_times():
    time_file = "train_and_eval/results/user-study-annotation-times.xlsx"

    summary = {
        "version": [],
        "time": [],
        "time_dev": [],
    }
    for version in range(1, 9):
        results = pd.read_excel(time_file, sheet_name=f"v{version}")
        results = results.drop("sushmita", axis=1)
        results = results.dropna()
        results = results[results.image.str.startswith("im")]
        results = results[~results.image.str.endswith("training")]

        annotator_time = {name: 0.0 for name in ANNOTATORS}
        annotator_objects = {name: 0.0 for name in ANNOTATORS}
        for _, row in results.iterrows():
            im_name = row.image
            for name in ANNOTATORS:
                ann_time = _to_seconds(row[name])
                annotator_time[name] += ann_time

                n_objects = _get_n_objects(version, name, im_name, min_size=50)
                annotator_objects[name] += n_objects

        times = np.array(list(annotator_time.values()))
        n_objects = np.array(list(annotator_objects.values()))
        time_per_obj = times / n_objects
        time = np.mean(time_per_obj)
        time_dev = np.std(time_per_obj)

        summary["version"].append(f"v{version}")
        summary["time"].append(time)
        summary["time_dev"].append(time_dev)

    return pd.DataFrame(summary)


def evaluate_processing_times():
    n_images = 6
    time_file = "train_and_eval/results/user-study-annotation-times.xlsx"

    versions = ["v5", "v6", "v7", "v7"]
    keys = ["preprocessing", "*training", "training", "preprocessing"]

    summary = {
        "version": [],
        "key": [],
        "time": [],
        "time_dev": [],
    }
    for version, key in zip(versions, keys):
        results = pd.read_excel(time_file, sheet_name=version)
        results = results.drop("sushmita", axis=1)
        # Exclude results for caro (too old laptop).
        results = results.drop("caro", axis=1)
        results = results.dropna()

        if "*" in key:
            assert key.startswith("*")
            pattern = key.replace("*", "")
            res = results[results.image.str.endswith(pattern)]
            res = res.drop("image", axis=1)
            res = res.applymap(_to_seconds)

            res = res.mean(axis=1)

            time = res.values.mean()
            dev = res.values.std()

        else:
            res = results[results["image"] == key]
            res = res.drop("image", axis=1)
            if key == "training":
                res = res.applymap(_to_hours)
            else:
                res = res.applymap(_to_seconds)

            time = res.values.mean()
            dev = res.values.std()

            # compute the time per image for preprocessing
            if key == "preprocessing":
                time /= n_images
                dev /= n_images

        summary["version"].append(version)
        summary["key"].append(key)
        summary["time"].append(time)
        summary["time_dev"].append(dev)

    return pd.DataFrame(summary)


def evaluate_annotation_quality():
    summary = {
        "version": [],
        "msa_ann": [],
        "msa_ann_dev": [],
    }

    min_size = 50

    for v in range(1, 9):
        version = f"v{v}"
        split = VERSION_TO_SPLIT[version]
        consensus_folder = os.path.join(CONSENSUS_ROOT, f"split{split}")
        consensus_labels = sorted(glob(os.path.join(consensus_folder, "*.tif")))
        msas = []

        for label_path in consensus_labels:
            consensus = imageio.imread(label_path)
            consensus = _size_filter(consensus, min_size)

            im_name = Path(label_path).stem
            for name in ANNOTATORS:
                seg = _load_labels(v, name, im_name, min_size=min_size)
                this_msa = mean_segmentation_accuracy(seg, consensus)
                msas.append(this_msa)

        summary["version"].append(version)
        summary["msa_ann"].append(np.mean(msas))
        summary["msa_ann_dev"].append(np.std(msas))

    summary = pd.DataFrame(summary)
    return summary


def _eval_generalization(result_file, version_list):
    results = pd.read_csv(result_file)
    assert len(results) == len(version_list)

    summary = {
        "version": [],
        "msa_test": [],
        "msa_test_dev": []
    }

    versions = np.unique(version_list)
    for version in versions:
        indices = np.where(version_list == version)[0]
        res = results.iloc[indices]

        if len(indices) == 1:
            msa = res.msa.values[0]
            msa_dev = ""
        else:
            msa = res.msa.mean()
            msa_dev = res.msa.std()

        summary["version"].append(version)
        summary["msa_test"].append(msa)
        summary["msa_test_dev"].append(msa_dev)

    return pd.DataFrame(summary)


def evaluate_generalization():
    # Evaluate the micro-sam results.
    sam_versions = np.array(["v2", "v3"] + 5 * ["v5"] + 5 * ["v7"])
    sam_results = _eval_generalization("train_and_eval/results/sam_test_images.csv", sam_versions)

    # Evaluate the cellpose results
    cp_versions = np.array(["v4"] + 5 * ["v6"] + 5 * ["v8"])
    cp_results = _eval_generalization("train_and_eval/results/cellpose_test_images.csv", cp_versions)

    summary = pd.concat([sam_results, cp_results])
    summary = summary.sort_values(by="version")
    return summary


def get_main_summary():
    summary_time = evaluate_annotation_times()
    summary_quality = evaluate_annotation_quality()
    summary_gen = evaluate_generalization()

    # merge and save the summary results
    summaries = [summary_time, summary_quality, summary_gen]
    summary = reduce(lambda left, right: pd.merge(left, right, on="version", how="outer"), summaries)
    return summary


def main():
    summary = get_main_summary()
    print(summary)
    summary.to_excel("./result_summary.xlsx")

    # Times are in seconds / image EXCEPT Training time, which is in hours
    # additional time evaluation for runtimes.
    additional_rt = evaluate_processing_times()
    print(additional_rt)
    additional_rt.to_excel("./runtime_summary.xlsx")


main()
