import os
from glob import glob

import imageio.v3 as imageio
import napari
from skimage.measure import label


IMAGE_ROOT = "../data/for_annotation"
LABEL_ROOT = "../data/consensus_labels/automatic"
OUT_ROOT = "../data/consensus_labels/proofread"


def proofread_split(split, skip_proofread):
    image_folder = os.path.join(IMAGE_ROOT, f"split{split}")
    split_images = sorted(glob(os.path.join(image_folder, "*.tif")))

    label_folder = os.path.join(LABEL_ROOT, f"split{split}")
    split_labels = sorted(glob(os.path.join(label_folder, "*.tif")))
    assert len(split_images) == len(split_labels)

    out_folder = os.path.join(OUT_ROOT, f"split{split}")
    os.makedirs(out_folder, exist_ok=True)

    for im_path, label_path in zip(split_images, split_labels):
        im = imageio.imread(im_path)
        labels = imageio.imread(label_path)

        out_path = os.path.join(out_folder, os.path.basename(im_path))
        if os.path.exists(out_path) and skip_proofread:
            print("Skipping already proof-read file", out_path)
            continue

        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(labels)

        @v.bind_key("s")
        def save_labels(v):
            labels = v.layers["labels"].data
            labels = label(labels)
            print("Saving labels to", out_path)
            imageio.imwrite(out_path, labels, compression="zlib")

        napari.run()


def main():
    skip_proofread = True
    for split in [1, 2, 3]:
        proofread_split(split, skip_proofread=skip_proofread)


if __name__ == "__main__":
    main()
