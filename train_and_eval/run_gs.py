import argparse
import os

import imageio.v3 as imageio
import torch

from micro_sam.instance_segmentation import get_predictor_and_decoder, get_amg
from micro_sam.training import default_sam_dataset
from micro_sam.evaluation import instance_segmentation as grid_search
from torch_em.segmentation import get_data_loader

from common import get_train_and_val_images, MODEL_ROOT


def _run_grid_search(model_type, ckpt_root, val_loader):
    ckpt_path = os.path.join(ckpt_root, "best.pt")

    grid_search_values = grid_search.default_grid_search_values_instance_segmentation_with_decoder()
    predictor, decoder = get_predictor_and_decoder(model_type, ckpt_path)
    segmenter = get_amg(predictor, decoder=decoder, is_tiled=False)

    # Create images for the grid search from the val loader.
    data_root = os.path.join(ckpt_root, "grid_search")
    os.makedirs(data_root, exist_ok=True)
    image_paths, gt_paths = [], []
    for i, (x, y) in enumerate(val_loader):
        im_path = os.path.join(data_root, f"im{i}.tif")
        imageio.imwrite(im_path, x[0, 0].numpy())
        image_paths.append(im_path)

        gt_path = os.path.join(data_root, f"gt{i}.tif")
        imageio.imwrite(gt_path, y[0, 0].numpy())
        gt_paths.append(gt_path)

    print("Start instance segmentation grid search.")
    grid_search.run_instance_segmentation_grid_search(
        segmenter, grid_search_values, image_paths, gt_paths, result_dir=data_root, embedding_dir=None,
    )
    gs_result = grid_search.evaluate_instance_segmentation_grid_search(data_root, list(grid_search_values.keys()))

    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt["grid_search"] = gs_result
    torch.save(ckpt, ckpt_path)


def run_grid_search(name, version):
    # Skip the ones that already have a grid-search result.
    checkpoint_root = os.path.join(MODEL_ROOT, "micro-sam", f"v{version}", name, "checkpoints", "organoid_model")
    checkpoint = os.path.join(checkpoint_root, "best.pt")
    state = torch.load(checkpoint, weights_only=False)
    if "grid_search" in state:
        print("Grid search already performed for", checkpoint)
        return

    _, _, val_images, val_labels = get_train_and_val_images(name)

    val_ds = default_sam_dataset(
        raw_paths=val_images, raw_key=None,
        label_paths=val_labels, label_key=None,
        patch_shape=(1024, 1024), with_segmentation_decoder=True,
        is_train=False, n_samples=20,
    )
    val_loader = get_data_loader(val_ds, shuffle=True, batch_size=2, num_workers=4)

    _run_grid_search(model_type="vit_b", ckpt_root=checkpoint_root, val_loader=val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("-v", "--version", type=int, default=7)
    args = parser.parse_args()
    run_grid_search(args.name, args.version)


if __name__ == "__main__":
    main()
