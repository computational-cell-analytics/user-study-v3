import json
import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np


def create_split(ims, split_dir):
    os.makedirs(split_dir, exist_ok=True)
    for i, im_path in enumerate(ims):
        im = imageio.imread(im_path)
        out_path = os.path.join(split_dir, f"im{i}.tif")
        imageio.imwrite(out_path, im, compression="zlib")


def create_splits():
    image_folder = "data/for_annotation/all_images"
    image_paths = glob(os.path.join(image_folder, "*.tif"))

    n_select = 15
    image_paths = np.random.choice(image_paths, n_select, replace=False)

    with open("data/for_annotation/predictions/cellpose_initial/sizes.json") as f:
        size_measure = json.load(f)

    sizes = []
    for path in image_paths:
        name = Path(path).stem
        seg_name = f"{name}_cp_masks.tif"
        sizes.append(size_measure[seg_name][0])

    # Sort the images by size and divide into three size categories
    size_sorted = np.argsort(sizes)
    cat1 = image_paths[size_sorted[:5]]
    np.random.shuffle(cat1)
    cat2 = image_paths[size_sorted[5:10]]
    np.random.shuffle(cat2)
    cat3 = image_paths[size_sorted[10:]]
    np.random.shuffle(cat3)
    assert all(len(cat) == 5 for cat in (cat1, cat2, cat3))

    cat1, cat2, cat3 = cat1.tolist(), cat2.tolist(), cat3.tolist()

    # Create split 1: 1 image from each category (=3)
    ims1 = cat1[:1] + cat2[:1] + cat3[:1]
    split1_dir = "data/for_annotation/split1"
    create_split(ims1, split1_dir)

    # Create split 2: 2 image from each category (=6)
    ims2 = cat1[1:3] + cat2[1:3] + cat3[1:3]
    split2_dir = "data/for_annotation/split2"
    create_split(ims2, split2_dir)

    # Create split 3: 2 image from each category (=6)
    ims3 = cat1[3:] + cat2[3:] + cat3[3:]
    split3_dir = "data/for_annotation/split3"
    create_split(ims3, split3_dir)


def check_splits():
    import napari

    for split in (1, 2, 3):
        split_folder = f"data/for_annotation/split{split}"
        files = sorted(glob(os.path.join(split_folder, "*.tif")))
        print("Number of files in split", split, ":")
        print(len(files))

        for ff in files:
            im = imageio.imread(ff)

            napari.view_image(im)
            napari.run()


def main():
    # create_splits()
    check_splits()


main()
