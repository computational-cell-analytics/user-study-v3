import os
import pickle
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from tqdm import tqdm


DATA_ROOT = "../data"

SPLITS_TO_VERSIONS = {
    1: ["v1", "v2", "v3", "v4"],
    2: ["v5", "v6"],
    3: ["v7", "v8"],
}


def load_labels(path, min_size):
    if path.endswith(".tif"):
        seg = imageio.imread(path)
    else:
        seg = np.load(path, allow_pickle=True).item()["masks"]

    seg = label(seg)
    ids, sizes = np.unique(seg, return_counts=True)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    seg = relabel_sequential(seg)[0]
    return seg


def _compute_pairwise_scores(image_labels):
    n_anns = len(image_labels)

    scores = {}
    for i in range(n_anns):
        for j in range(n_anns):
            if i >= j:
                continue

            overlaps = label_overlap(image_labels[i], image_labels[j])[0]
            overlaps = intersection_over_union(overlaps)
            scores[(i, j)] = overlaps

    return scores


# TODO cache the pairwise overlaps
def _compute_all_scores(split, annotations):
    cache_folder = "overlaps"
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, f"overlaps_split{split}.pickle")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            pairwise_scores = pickle.load(f)
        return pairwise_scores

    pairwise_scores = {}
    for im_name, image_labels in tqdm(annotations.items(), desc="Compute overlaps", total=len(annotations)):
        scores = _compute_pairwise_scores(image_labels)
        pairwise_scores[im_name] = scores

    with open(cache_path, "wb") as f:
        pickle.dump(pairwise_scores, f)

    return pairwise_scores


def _create_consensus(labels, overlaps, overlap_threshold, matching_threshold):
    matches = [np.zeros(len(np.unique(lab))) for lab in labels]

    n_pairs = len(labels) - 1
    for pair, scores in overlaps.items():
        i, j = pair
        matches_i, matches_j = matches[i], matches[j]
        this_matches = np.where(scores > overlap_threshold)
        this_matches_i, this_matches_j = this_matches

        matches_i[this_matches_i] += 1
        matches_j[this_matches_j] += 1

        matches[i], matches[j] = matches_i, matches_j

    matches = [match / n_pairs for match in matches]
    selections = [match > matching_threshold for match in matches]

    done = [[]] * len(selections)
    for i, selection in enumerate(selections):
        selected = np.where(selection)

    breakpoint()


def make_consensus_annotations(
    split, image_names, annotation_folders,
    overlap_threshold, matching_threshold,
    min_size=50
):
    n_images = len(image_names)
    annotations = {im_name: [] for im_name in image_names}

    for folder in annotation_folders:
        name = "-".join(folder.split("/")[-2:])
        label_paths = sorted(glob(os.path.join(folder, "*.tif")))
        if len(label_paths) == 0:
            label_paths = sorted(glob(os.path.join(folder, "*.npy")))
        assert len(label_paths) == n_images, f"{name}: {len(label_paths)} != {n_images}"

        for lpath in label_paths:
            labels = load_labels(lpath, min_size=min_size)
            im_name = Path(lpath).stem.rstrip("_seg")
            annotations[im_name].append(labels)

    pairwise_scores = _compute_all_scores(split, annotations)

    consensus_labels = []
    for im_name, scores in pairwise_scores.items():
        im_labels = annotations[im_name]
        print(im_name, "has", len(im_labels), "annotations.")
        this_consensus = _create_consensus(im_labels, scores, overlap_threshold, matching_threshold)
        consensus_labels.append(this_consensus)

    return consensus_labels


def create_consensus_annotations(split, view=True, save=False):
    versions = SPLITS_TO_VERSIONS[split]

    images = sorted(glob(os.path.join(DATA_ROOT, "for_annotation", f"split{split}", "*.tif")))
    image_names = [Path(im_path).stem for im_path in images]

    annotation_folders = []
    for version in versions:
        version_folder = os.path.join(DATA_ROOT, "annotations", version)
        annotation_folders.extend(sorted(glob(os.path.join(version_folder, "*"))))

    overlap_threshold = 0.8
    matching_threshold = 0.8
    labels = make_consensus_annotations(
        split, image_names, annotation_folders,
        overlap_threshold=overlap_threshold,
        matching_threshold=matching_threshold,
    )
    return

    if view:
        import napari

        assert len(images) == len(labels)
        for im, lab in zip(images, labels):
            im, lab = imageio.imread(im), imageio.imread(lab)
            v = napari.Viewer()
            v.add_image(im)
            v.add_labels(lab)
            napari.run()

    # TODO
    if save:
        pass


def main():
    split = 1
    create_consensus_annotations(split)


if __name__ == "__main__":
    main()
