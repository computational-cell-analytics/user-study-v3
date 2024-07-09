import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import napari

from skimage.measure import label


def check_reference_images():
    images = sorted(glob("data/ground_truth/images/*.tif"))
    masks_v1 = "data/ground_truth/masks_v1"
    masks_v2 = "data/ground_truth/masks_v2"
    results = "data/ground_truth/masks"
    os.makedirs(results, exist_ok=True)

    print("Number of annotated images:", len(images))

    for im_path in images:
        name = Path(im_path).stem

        v1_path = os.path.join(masks_v1, f"{name}.tif")
        mask = imageio.imread(v1_path)
        load_path = v1_path

        v2_path = os.path.join(masks_v2, f"{name}_cp_masks.png")
        if os.path.exists(v2_path):
            mask = imageio.imread(v2_path)
            load_path = v2_path

        result_path = os.path.join(results, f"{name}.tif")
        if os.path.exists(result_path):
            mask = imageio.imread(result_path)
            load_path = result_path

        image = imageio.imread(im_path)

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(mask)
        v.title = load_path

        @v.bind_key("s")
        def _save(viewer):
            mask = viewer.layers["mask"].data
            mask = label(mask)
            print("Saving mask to", result_path)
            imageio.imwrite(result_path, mask, compression="zlib")

        napari.run()


def check_new_images():
    images = sorted(glob("data/for_annotation/all_images/*.tif"))

    print("Number of new images:", len(images))
    for im_path in images:
        image = imageio.imread(im_path)

        v = napari.Viewer()
        v.add_image(image)
        napari.run()


def main():
    check_reference_images()
    # check_new_images()


if __name__ == "__main__":
    main()
