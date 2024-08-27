import os
import imageio.v3 as imageio

import h5py
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation


def export():
    image_folder = "/scratch-emmy/projects/nim00007/user-study/data/ground_truth/images"
    pred_folder = "/scratch-emmy/projects/nim00007/user-study/predictions/test_images"

    image_name = "VID532_H12_1_11d16h00m.tif"

    out_folder = "./images/for_figure"
    os.makedirs(out_folder, exist_ok=True)

    im = imageio.imread(os.path.join(image_folder, image_name))
    label_path = os.path.join(pred_folder, image_name.replace(".tif", ".h5"))

    with h5py.File(label_path, "r") as f:
        seg_vanilla = f["sam/vit_b"][:]
        seg_ft = f["sam/v7/marei"][:]

    imageio.imwrite(os.path.join(out_folder, "image.tif"), im, compression="zlib")
    imageio.imwrite(os.path.join(out_folder, "seg_vanilla.tif"), seg_vanilla, compression="zlib")
    imageio.imwrite(os.path.join(out_folder, "seg_ft.tif"), seg_ft, compression="zlib")


def _to_boundaries(seg, dilation=0):
    boundaries = find_boundaries(seg)
    if dilation > 0:
        boundaries = binary_dilation(boundaries, iterations=dilation)
    return boundaries


def make_figure():
    import napari

    im = imageio.imread("images/for_figure/image.tif")
    seg_vanilla = imageio.imread("images/for_figure/seg_vanilla.tif")
    bd_vanilla = _to_boundaries(seg_vanilla)

    seg_ft = imageio.imread("images/for_figure/seg_ft.tif")
    bd_ft = _to_boundaries(seg_ft)

    v = napari.Viewer()
    v.add_image(im)

    v.add_labels(seg_vanilla)
    v.add_labels(bd_vanilla)

    v.add_labels(seg_ft)
    v.add_labels(bd_ft)

    napari.run()


# export()
make_figure()
