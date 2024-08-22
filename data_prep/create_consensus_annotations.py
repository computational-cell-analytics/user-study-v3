import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import napari
import numpy as np

from affogato.affinities import compute_affinities
from elf.segmentation.mutex_watershed import mutex_watershed
from skimage.measure import label
from skimage.segmentation import relabel_sequential, watershed

DATA_ROOT = "../data"
SPLITS_TO_VERSIONS = {
    1: ["v1", "v2", "v3", "v4"],
    2: ["v5", "v6"],
    3: ["v7", "v8"],
}
ANNOTATORS = ["anwai", "caro", "constantin", "luca", "marei"]


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


def _run_segmentation(affs, offsets, consensus_mask, min_size):
    seg = mutex_watershed(
        affs, offsets, strides=[3, 3], randomize_strides=True, mask=consensus_mask
    )
    ids, sizes = np.unique(seg, return_counts=True)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0

    hmap = np.max(affs[:2], axis=0)
    seg = watershed(hmap, markers=seg, mask=consensus_mask)

    return seg


def make_consensus(im_name, path, annotation_folders, view, t_fg):
    all_labels = []
    all_affs = []

    min_size = 50

    offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]
    # load all the annotations
    for folder in annotation_folders:
        label_path = os.path.join(folder, f"{im_name}.tif")
        if not os.path.exists(label_path):
            label_path = os.path.join(folder, f"{im_name}_seg.npy")
        assert os.path.exists(label_path), label_path
        labels = load_labels(label_path, min_size=min_size)
        all_labels.append(labels)

        affs, mask = compute_affinities(labels, offsets, have_ignore_label=False)
        affs = 1. - affs
        mask = mask.astype("bool")
        affs[~mask] = 0
        all_affs.append(affs)

    n_labels = float(len(all_labels))

    stacked_binary_labels = np.stack(all_labels) != 0
    consensus_probs = np.sum(stacked_binary_labels.astype("float"), axis=0)
    consensus_probs /= n_labels
    consensus_mask = consensus_probs > t_fg

    stacked_affs = np.stack(all_affs)
    normalized_affs = stacked_affs.sum(axis=0) / n_labels

    consensus_labels = _run_segmentation(affs, offsets, consensus_mask, min_size)
    if view:
        image = imageio.imread(path)
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(consensus_probs, visible=False)
        v.add_image(normalized_affs, visible=False)
        v.add_labels(consensus_mask, visible=False)
        v.add_labels(consensus_labels)
        napari.run()

    return consensus_labels


def create_consensus_annotations(split, view=True, save=False):
    versions = SPLITS_TO_VERSIONS[split]

    # Foreground threshold
    t_fg = 0.6

    images = sorted(glob(os.path.join(DATA_ROOT, "for_annotation", f"split{split}", "*.tif")))
    image_names = [Path(im_path).stem for im_path in images]

    annotation_folders = []
    for version in versions:
        version_folder = os.path.join(DATA_ROOT, "annotations", version)
        annotation_folders.extend([os.path.join(version_folder, ann) for ann in ANNOTATORS])

    save_folder = os.path.join(DATA_ROOT, "consensus_labels", "automatic", f"split{split}")
    for name, path in zip(image_names, images):
        consensus_labels = make_consensus(name, path, annotation_folders, view=view, t_fg=t_fg)
        if save:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{name}.tif")
            imageio.imwrite(save_path, consensus_labels, compression="zlib")


# Algorithm:
# 1. Sum up the foreground per pixel and divide by the number of annotations
#   -> this gives us consensus foreground probs
#   -> everything above a certain threshold 't_fg' is foreground
# 2. Compute affinites for each segmentation, sum up and normalize.
#   -> Run mutex watershed based on these affinities and the foreground map from before.
def main():
    for split in (1, 2, 3):
        print("Processing split", split)
        create_consensus_annotations(split, view=False, save=True)


if __name__ == "__main__":
    main()
