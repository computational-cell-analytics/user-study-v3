import argparse
import os
from glob import glob

import imageio.v3 as imageio
import napari

from skimage.measure import label

IMAGE_ROOT = "./data/for_annotation"
ANNOTATION_ROOT = "./data/annotations"
NUM_IMAGES = {
    "v1": 3, "v2": 3, "v3": 3
}
SPLITS = {
    "v1": "split1", "v2": "split1", "v3": "split1",
}


def check_data():
    versions = sorted(glob(os.path.join(ANNOTATION_ROOT, "v*")))
    for version_folder in versions:
        version = os.path.split(version_folder)[1]
        annotations = sorted(glob(os.path.join(version_folder, "*")))
        print("Version", version, "status:")
        for annotation_folder in annotations:
            name = os.path.split(annotation_folder)[1]
            is_complete = len(glob(os.path.join(annotation_folder, "*.tif"))) == NUM_IMAGES[version]
            print(name, ":", "complete" if is_complete else "INCOMPLETE!")


def check_annotations(versions):
    if versions is None:
        versions = sorted(glob(os.path.join(ANNOTATION_ROOT, "v*")))
    else:
        versions = [
            os.path.join(os.path.join(ANNOTATION_ROOT, f"v{version}")) for version in versions
        ]

    for version_folder in versions:
        version = os.path.split(version_folder)[1]
        image_folder = os.path.join(IMAGE_ROOT, SPLITS[version])
        print("Checking version:", version)
        annotations = sorted(glob(os.path.join(version_folder, "*")))

        images = sorted(glob(os.path.join(image_folder, "*.tif")))

        for path in images:
            fname = os.path.split(path)[1]
            image = imageio.imread(path)

            labels = {}
            for annotation_folder in annotations:
                name = os.path.split(annotation_folder)[1]
                labels[name] = label(imageio.imread(os.path.join(annotation_folder, fname)))

            v = napari.Viewer()
            v.add_image(image)
            for name, lab in labels.items():
                v.add_labels(lab, name=name)
            v.title = f"{version}:{fname}"
            napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--versions", default=None, type=int, nargs="+")

    args = parser.parse_args()
    versions = args.versions

    # check_data()
    check_annotations(versions)


if __name__ == "__main__":
    main()
