import json
import os
import subprocess
from glob import glob

import numpy as np
from natsort import natsorted


def run_cp(input_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "python",
        "-m",
        "cellpose",
        "--verbose",
        "--use_gpu",
        "--dir",
        input_dir,
        "--pretrained_model",
        model_path,
        "--chan",
        "0",
        "--chan2",
        "0",
        "--diameter",
        "0",
        "--no_interp",
        "--save_tif",
        "--savedir",
        output_dir,
        "--no_npy",
    ]

    # Run the command
    subprocess.run(command, check=True)


def analyze_results(input_dir, output_dir, check_visually):
    import imageio.v3 as imageio
    from skimage.measure import regionprops

    segmentations = natsorted(glob(os.path.join(output_dir, "*.tif")))

    if check_visually:
        import napari

        images = natsorted(glob(os.path.join(input_dir, "*.tif")))
        assert len(images) == len(segmentations)

        for im_file, seg_file in zip(images, segmentations):
            im = imageio.imread(im_file)
            seg = imageio.imread(seg_file)

            v = napari.Viewer()
            v.add_image(im)
            v.add_labels(seg)
            napari.run()

    size_dict = {}
    for seg_file in segmentations:
        seg = imageio.imread(seg_file)
        props = regionprops(seg)
        sizes = np.array([prop.area for prop in props])
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        size_dict[os.path.basename(seg_file)] = (mean_size, std_size)

    save_path = os.path.join(output_dir, "sizes.json")
    with open(save_path, "w") as f:
        json.dump(size_dict, f)


def main():
    input_dir = "data/for_annotation/all_images"
    output_dir = "data/for_annotation/predictions/cellpose_initial"

    # run_cp(input_dir=input_dir, output_dir=output_dir, model_path="models/cellpose_initial")
    analyze_results(input_dir, output_dir, check_visually=False)


if __name__ == "__main__":
    main()
