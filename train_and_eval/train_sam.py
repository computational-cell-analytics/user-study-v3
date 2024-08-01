import argparse
import os

from micro_sam.training import train_sam_for_configuration, default_sam_dataset
from torch_em.segmentation import get_data_loader

from common import get_train_and_val_images, MODEL_ROOT


def run_training(name):
    with_segmentation_decoder = True

    train_images, train_labels, val_images, val_labels = get_train_and_val_images(name)

    num_workers = 8
    patch_shape = (1024, 1024)

    train_ds = default_sam_dataset(
        raw_paths=train_images, raw_key=None,
        label_paths=train_labels, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        n_samples=250,
    )
    train_loader = get_data_loader(train_ds, shuffle=True, batch_size=2, num_workers=num_workers)

    val_ds = default_sam_dataset(
        raw_paths=val_images, raw_key=None,
        label_paths=val_labels, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        is_train=False, n_samples=20,
    )
    val_loader = get_data_loader(val_ds, shuffle=True, batch_size=2, num_workers=num_workers)

    save_root = os.path.join(MODEL_ROOT, f"micro-sam/v7/{name}")
    train_sam_for_configuration(
        name="organoid_model", configuration="A100",
        train_loader=train_loader, val_loader=val_loader,
        with_segmentation_decoder=with_segmentation_decoder,
        save_root=save_root, device="cuda",
        model_type="vit_b_lm",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    run_training(args.name)


if __name__ == "__main__":
    main()
