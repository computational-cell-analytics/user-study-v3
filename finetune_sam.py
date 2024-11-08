# Finetune a SAM model on the annotated data from v5
import argparse
import os

from glob import glob


from micro_sam.training import train_sam_for_configuration, default_sam_dataset
from torch_em.segmentation import get_data_loader

# For local finetuning.
# DATA_ROOT = "data"

# For finetuning on the cluster.
# DATA_ROOT = "/scratch-emmy/projects/nim00007/user-study/data"
DATA_ROOT = "/scratch1/projects/cca/data/user-study/data"


def get_paths(name):
    image_folder = os.path.join(DATA_ROOT, "for_annotation/split2")
    label_folder = os.path.join(DATA_ROOT, f"annotations/v5/{name}")

    images = sorted(glob(os.path.join(image_folder, "*.tif")))
    labels = sorted(glob(os.path.join(label_folder, "*.tif")))
    assert len(images) == len(labels) == 6

    # We use im4 as val.
    train_images = images[:4] + images[-1:]
    train_labels = labels[:4] + labels[-1:]

    val_images = images[4:5]
    val_labels = labels[4:5]

    return train_images, train_labels, val_images, val_labels


def run_training(name):
    with_segmentation_decoder = True

    train_images, train_labels, val_images, val_labels = get_paths(name)

    num_workers = 0
    patch_shape = (1024, 1024)

    train_ds = default_sam_dataset(
        raw_paths=train_images, raw_key=None,
        label_paths=train_labels, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        n_samples=50,
    )
    train_loader = get_data_loader(train_ds, shuffle=True, batch_size=1, num_workers=num_workers)

    val_ds = default_sam_dataset(
        raw_paths=val_images, raw_key=None,
        label_paths=val_labels, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        is_train=False, n_samples=4,
    )
    val_loader = get_data_loader(val_ds, shuffle=True, batch_size=1, num_workers=num_workers)

    print("Start finetuning for", name, "!!")
    train_sam_for_configuration(
        name="organoid_model", configuration="CPU",
        train_loader=train_loader, val_loader=val_loader,
        with_segmentation_decoder=with_segmentation_decoder,
        save_root=f"./models/{name}", device="cpu",
        model_type="vit_b_lm"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    run_training(args.name)


if __name__ == "__main__":
    main()
