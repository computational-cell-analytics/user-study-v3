import argparse
import os

import torch_em

from torch_em.model import UNet2d
from torch_em.data.datasets import get_organoidnet_loader, get_organoidnet_dataset


# ROOT = "../data"
ROOT = "/scratch-emmy/projects/nim00007/user-study/data"


def _get_orga_loader(batch_size, label_transform):
    path = os.path.join(ROOT, "datasets", "orga", "v1")
    patch_shape = (1, 512, 512)

    train_images = os.path.join(path, "train", "images")
    train_labels = os.path.join(path, "train", "masks")
    assert os.path.exists(train_images)
    assert os.path.exists(train_labels)
    train_loader = torch_em.default_segmentation_loader(
        raw_paths=train_images, label_paths=train_labels,
        raw_key="*.tif", label_key="*.tif",
        patch_shape=patch_shape, ndim=2,
        label_transform=label_transform, batch_size=batch_size,
        is_seg_dataset=True, shuffle=True, num_workers=6,
        n_samples=500,
    )

    val_images = os.path.join(path, "val", "images")
    val_labels = os.path.join(path, "val", "masks")
    assert os.path.exists(val_images)
    assert os.path.exists(val_labels)
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=val_images, label_paths=val_labels,
        raw_key="*.tif", label_key="*.tif",
        patch_shape=patch_shape, ndim=2,
        label_transform=label_transform, batch_size=batch_size,
        is_seg_dataset=True, shuffle=True, num_workers=6,
        n_samples=40,
    )

    return train_loader, val_loader


def _get_organoidnet_loader(batch_size, label_transform):
    path = os.path.join(ROOT, "datasets", "organoidnet")
    patch_shape = (512, 512)

    train_loader = get_organoidnet_loader(
        path, split="Training", patch_shape=patch_shape, batch_size=batch_size, label_transform=label_transform,
        shuffle=True, num_workers=6,
    )
    val_loader = get_organoidnet_loader(
        path, split="Validation", patch_shape=patch_shape, batch_size=batch_size, label_transform=label_transform,
        shuffle=True, num_workers=6,
    )

    return train_loader, val_loader


def _get_combined_loader(batch_size, label_transform):
    path_organoidnet = os.path.join(ROOT, "datasets", "organoidnet")
    path_orga = os.path.join(ROOT, "datasets", "orga", "v1")
    patch_shape = (512, 512)

    train_images = os.path.join(path_orga, "train", "images")
    train_labels = os.path.join(path_orga, "train", "masks")
    train_ds = [
        get_organoidnet_dataset(
            path_organoidnet, split="Training", patch_shape=patch_shape, label_transform=label_transform
        ),
        torch_em.default_segmentation_dataset(
            raw_paths=train_images, label_paths=train_labels, raw_key="*.tif", label_key="*.tif",
            patch_shape=(1,) + patch_shape, ndim=2,
            label_transform=label_transform, is_seg_dataset=True, n_samples=500,
        )
    ]
    train_ds = torch_em.data.ConcatDataset(*train_ds)
    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, num_workers=6, shuffle=True)

    val_images = os.path.join(path_orga, "val", "images")
    val_labels = os.path.join(path_orga, "val", "masks")
    val_ds = [
        get_organoidnet_dataset(
            path_organoidnet, split="Validation", patch_shape=patch_shape, label_transform=label_transform
        ),
        torch_em.default_segmentation_dataset(
            raw_paths=val_images, label_paths=val_labels, raw_key="*.tif", label_key="*.tif",
            patch_shape=(1,) + patch_shape, ndim=2,
            label_transform=label_transform, is_seg_dataset=True, n_samples=40,
        )
    ]
    val_ds = torch_em.data.ConcatDataset(*val_ds)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=batch_size, num_workers=6, shuffle=True)

    return train_loader, val_loader


def get_loaders(ds_name, batch_size, label_transform):
    if ds_name == "orga/v1":
        return _get_orga_loader(batch_size, label_transform)
    elif ds_name == "combined":
        return _get_combined_loader(batch_size, label_transform)
    else:
        return _get_organoidnet_loader(batch_size, label_transform)


def train_unet(ds_name):
    assert ds_name in ("orga/v1", "organoidnet", "combined")
    model = UNet2d(in_channels=1, out_channels=3, initial_features=64, final_activation="Sigmoid")

    batch_size = 4

    label_trafo = torch_em.transform.label.PerObjectDistanceTransform()
    train_loader, val_loader = get_loaders(ds_name, batch_size=batch_size, label_transform=label_trafo)
    loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)

    name = f"distance_unet/{ds_name}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
    )
    trainer.fit(iterations=int(5e4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name")
    args = parser.parse_args()
    train_unet(args.ds_name)


if __name__ == "__main__":
    main()
