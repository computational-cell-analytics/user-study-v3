import argparse
import os
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np

from skimage.measure import label

IMAGE_ROOT = "./data/for_annotation"
ANNOTATION_ROOT = "./data/annotations"
NUM_IMAGES = {
    "v1": 3, "v2": 3, "v3": 3, "v4": 3,
    "v5": 6, "v6": 6, "v7": 6, "v8": 6
}
SPLITS = {
    "v1": "split1", "v2": "split1", "v3": "split1", "v4": "split1",
    "v5": "split2", "v6": "split2", "v7": "split3", "v8": "split3",
}


def check_data(annotators, versions):
    if versions is None:
        versions = sorted(glob(os.path.join(ANNOTATION_ROOT, "v*")))
    else:
        versions = [
            os.path.join(os.path.join(ANNOTATION_ROOT, f"v{version}")) for version in versions
        ]

    for version_folder in versions:
        version = os.path.split(version_folder)[1]
        annotations = sorted(glob(os.path.join(version_folder, "*")))
        print("Version", version, "status:")
        for annotation_folder in annotations:
            name = os.path.split(annotation_folder)[1]
            if annotators is not None and name not in annotators:
                continue

            if version in ("v4", "v6", "v8"):
                is_complete = len(glob(os.path.join(annotation_folder, "*.npy"))) == NUM_IMAGES[version]
            else:
                is_complete = len(glob(os.path.join(annotation_folder, "*.tif"))) == NUM_IMAGES[version]
            print(name, ":", "complete" if is_complete else "INCOMPLETE!")


def check_annotations(annotators, versions):
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
                if annotators is not None and name not in annotators:
                    continue

                if version in ("v4", "v6", "v8"):
                    annotation_file = os.path.join(annotation_folder, fname.replace(".tif", "_seg.npy"))
                    segmentation = np.load(annotation_file, allow_pickle=True).item()["masks"]
                else:
                    segmentation = imageio.imread(os.path.join(annotation_folder, fname))
                segmentation = label(segmentation)
                labels[name] = segmentation

            v = napari.Viewer()
            v.add_image(image)
            for name, lab in labels.items():
                v.add_labels(lab, name=name)
            v.title = f"{version}:{fname}"
            napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--versions", default=None, type=int, nargs="+")
    parser.add_argument("-a", "--annotators", default=None, nargs="+")
    parser.add_argument("-s", "--show", action="store_true")

    args = parser.parse_args()
    versions = args.versions
    annotators = args.annotators

    check_data(annotators=annotators, versions=versions)
    if args.show:
        check_annotations(annotators=annotators, versions=versions)


if __name__ == "__main__":
    main()
