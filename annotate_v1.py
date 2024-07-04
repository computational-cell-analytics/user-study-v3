import os
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np


def main():
    images = sorted(glob("data/for_annotation/split1/*.tif"))
    assert len(images) == 3, "Wrong number of images. Please check the data!"

    for im in images:
        image = imageio.imread(im)
        annotations = np.zeros(image.shape, dtype="uint16")

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(annotations)
        v.title = f"Image Name: {os.path.basename(im)}"
        napari.run()


if __name__ == "__main__":
    main()
