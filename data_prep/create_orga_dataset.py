"""Create our organoid dataset based on the annotations from the user study and the previous annotations.
"""

import os
from glob import glob

import imageio.v3 as imageio
import numpy as np

from skimage.measure import label
from skimage.segmentation import relabel_sequential
from sklearn.model_selection import train_test_split

ROOT = "../data"


def _load_labels(label_file, min_size):
    seg = imageio.imread(label_file)
    seg = label(seg)
    ids, sizes = np.unique(seg, return_counts=True)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    seg = relabel_sequential(seg)[0]
    return seg


def _check_size_histogram(label_images):
    from elf.visualisation.size_histogram import plot_size_histogram

    all_sizes = []
    for lab in label_images:
        _, this_sizes = np.unique(lab, return_counts=True)
        all_sizes.append(this_sizes[1:])
    all_sizes = np.concatenate(all_sizes)

    plot_size_histogram(all_sizes)


def _match_image(im, original_images):
    for name, candidate_im in original_images.items():
        match = np.allclose(im, candidate_im)
        if match:
            return name
    raise ValueError("Could not match image")


def _create_splits(output_folder, images, labels, original_image_paths=None):
    assert len(images) == len(labels)

    sizes = []
    for seg in labels:
        _, seg_sizes = np.unique(seg, return_counts=True)
        sizes.append(np.mean(seg_sizes[1:]))
    assert len(sizes) == len(images)

    n = len(images) // 3

    size_sorted = np.argsort(sizes)
    im_cat1 = images[size_sorted[:n]]
    lab_cat1 = labels[size_sorted[:n]]

    im_cat2 = images[size_sorted[n:2*n]]
    lab_cat2 = labels[size_sorted[n:2*n]]

    im_cat3 = images[size_sorted[2*n:]]
    lab_cat3 = labels[size_sorted[2*n:]]

    assert len(im_cat1) + len(im_cat2) + len(im_cat3) == len(images)
    assert len(lab_cat1) + len(lab_cat2) + len(lab_cat3) == len(labels)

    train_images, val_images, test_images = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for ims, labls in (
        (im_cat1, lab_cat1), (im_cat2, lab_cat2), (im_cat3, lab_cat3)
    ):
        train_ims, test_ims, train_labs, test_labs = train_test_split(
            ims, labls, test_size=2
        )
        train_ims, val_ims, train_labs, val_labs = train_test_split(
            train_ims, train_labs, test_size=1
        )

        train_images.extend(train_ims)
        train_labels.extend(train_labs)

        test_images.extend(test_ims)
        test_labels.extend(test_labs)

        val_images.extend(val_ims)
        val_labels.extend(val_labs)

    if original_image_paths is None:
        original_images = None
    else:
        original_images = {os.path.basename(path): imageio.imread(path) for path in original_image_paths}

    split_images = {"train": train_images, "val": val_images, "test": test_images}
    split_labels = {"train": train_labels, "val": val_labels, "test": test_labels}
    for split in ("train", "val", "test"):

        im_folder = os.path.join(output_folder, split, "images")
        os.makedirs(im_folder, exist_ok=True)

        label_folder = os.path.join(output_folder, split, "masks")
        os.makedirs(label_folder, exist_ok=True)

        for i, (im, lab) in enumerate(zip(split_images[split], split_labels[split])):

            # Try image matching if we have the original image paths.
            if original_images is None:
                fname = f"image{i:02}.tif"
            else:
                fname = _match_image(im, original_images)

            imageio.imwrite(os.path.join(im_folder, fname), im, compression="zlib")
            imageio.imwrite(os.path.join(label_folder, fname), lab, compression="zlib")


def create_orga_dataset_v1(check_sizes=False):
    """Create first version of the dataset based only on annotations from Sushmita.
    """
    image_folders = [
        os.path.join(ROOT, "ground_truth", "images"),
        os.path.join(ROOT, "for_annotation", "split1"),
        os.path.join(ROOT, "for_annotation", "split2"),
        os.path.join(ROOT, "for_annotation", "split3"),
    ]
    annotation_folders = [
        os.path.join(ROOT, "ground_truth", "masks"),
        os.path.join(ROOT, "annotations", "v2", "sushmita"),
        os.path.join(ROOT, "annotations", "v5", "sushmita"),
        os.path.join(ROOT, "annotations", "v7", "sushmita"),
    ]

    # 200 appears a good size cutoff.
    min_size = 200

    images, labels = [], []
    for im_folder, ann_folder in zip(image_folders, annotation_folders):
        for im, ann in zip(
            sorted(glob(os.path.join(im_folder, "*.tif"))), sorted(glob(os.path.join(ann_folder, "*.tif")))
        ):
            images.append(imageio.imread(im))
            labels.append(_load_labels(ann, min_size))

    images, labels = np.array(images), np.array(labels)
    assert len(images) == len(labels)

    # To determine a good min-size.
    if check_sizes:
        _check_size_histogram(labels)
        return

    output_folder = os.path.join(ROOT, "datasets", "orga", "v1")
    os.makedirs(output_folder, exist_ok=True)

    _create_splits(output_folder, images, labels)


def create_orga_dataset_v2(check_sizes=False):
    """Create second and final version of the dataset based on consensus annotations.
    """
    image_folders = [
        os.path.join(ROOT, "ground_truth", "images"),
        os.path.join(ROOT, "for_annotation", "split1"),
        os.path.join(ROOT, "for_annotation", "split2"),
        os.path.join(ROOT, "for_annotation", "split3"),
    ]
    annotation_folders = [
        os.path.join(ROOT, "ground_truth", "masks"),
        os.path.join(ROOT, "consensus_labels", "proofread", "split1"),
        os.path.join(ROOT, "consensus_labels", "proofread", "split2"),
        os.path.join(ROOT, "consensus_labels", "proofread", "split3"),
    ]

    # 150 appears a good size cutoff.
    min_size = 150

    images, labels = [], []
    for im_folder, ann_folder in zip(image_folders, annotation_folders):
        for im, ann in zip(
            sorted(glob(os.path.join(im_folder, "*.tif"))), sorted(glob(os.path.join(ann_folder, "*.tif")))
        ):
            images.append(imageio.imread(im))
            labels.append(_load_labels(ann, min_size))

    images, labels = np.array(images), np.array(labels)
    assert len(images) == len(labels)

    # To determine a good min-size.
    if check_sizes:
        _check_size_histogram(labels)
        return

    output_folder = os.path.join(ROOT, "datasets", "orga", "v2")
    os.makedirs(output_folder, exist_ok=True)

    original_image_paths = glob(os.path.join(ROOT, "ground_truth", "images", "*.tif")) +\
        glob(os.path.join(ROOT, "for_annotation", "all_images", "*.tif"))
    _create_splits(output_folder, images, labels, original_image_paths)


def check_dataset(version, view=True):
    import napari

    n_images = 0
    n_organoids = 0

    root = os.path.join(ROOT, "datasets", "orga", version)
    for split in ("train", "val", "test"):
        images = sorted(glob(os.path.join(root, split, "images", "*.tif")))
        labels = sorted(glob(os.path.join(root, split, "masks", "*.tif")))

        for im, lab in zip(images, labels):
            image = imageio.imread(im)
            seg = imageio.imread(lab)

            if view:
                v = napari.Viewer()
                v.add_image(image)
                v.add_labels(seg)
                v.title = f"{split}:{os.path.basename(im)}"
                napari.run()

            n_images += 1
            n_organoids += len(np.unique(seg)[1:])

    print("Number of images:", n_images)
    print("Number of organoids:", n_organoids)


# Dataset v1:
# Number of images: 30
# Number of organoids: 5779

# Dataset v2:
# Number of images: 30
# Number of organoids: 6252
def main():
    # create_orga_dataset_v1()
    # create_orga_dataset_v2()

    check_dataset("v2", view=False)


if __name__ == "__main__":
    main()
