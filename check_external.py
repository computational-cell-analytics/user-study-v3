import os
import imageio.v3 as imageio
import napari


def check_orgasegment():
    from torch_em.data.datasets.light_microscopy.orgasegment import _get_data_paths

    print("Checking orga segment")

    im_paths, label_paths = _get_data_paths("./data/datasets/orgasegment", download=True, split="train")
    print("Number of image paths:", len(im_paths))

    for ii in range(5):
        im = imageio.imread(im_paths[ii])
        mask = imageio.imread(label_paths[ii])

        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(mask)
        napari.run()


def check_organoidnet():
    from torch_em.data.datasets.light_microscopy.organoidnet import _get_data_paths

    print("Checking organoidnet")

    im_paths, label_paths = _get_data_paths("./data/datasets/organoidnet", download=True, split="Test")
    print("Number of image paths:", len(im_paths))

    for ii in range(10):
        im = imageio.imread(im_paths[ii])
        mask = imageio.imread(label_paths[ii])

        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(mask)
        v.title = os.path.basename(im_paths[ii])
        napari.run()


def main():
    # check_orgasegment()
    check_organoidnet()


if __name__ == "__main__":
    main()
