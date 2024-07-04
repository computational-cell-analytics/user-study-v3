from glob import glob

import imageio.v3 as imageio
import napari


def check_reference_images():
    images = sorted(glob("data/reference_data/images/*.tif"))
    masks = sorted(glob("data/reference_data/masks/*.tif"))

    print("Number of annotated images:", len(images))

    for im_path, mask_path in zip(images, masks):
        image = imageio.imread(im_path)
        mask = imageio.imread(mask_path)

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(mask)
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
