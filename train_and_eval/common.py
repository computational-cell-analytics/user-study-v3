import os
from functools import partial
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from tqdm import tqdm

# Change this for cluster
DATA_ROOT = "../data"
MODEL_ROOT = ""


def get_test_split_folders():
    image_folder = os.path.join(DATA_ROOT, "ground_truth/images")
    label_folder = os.path.join(DATA_ROOT, "ground_truth/masks")
    return image_folder, label_folder


def get_organoidnet_folders():
    image_folder = os.path.join(DATA_ROOT, "datasets/organoidnet/Test/Images")
    label_folder = os.path.join(DATA_ROOT, "datasets/organoidnet/Test/Masks")
    return image_folder, label_folder


def get_all_cellpose_models():
    pass


def get_all_sam_models():
    pass


def _evaluate(image_folder, label_folder, seg_function, verbose=True):
    images = glob(os.path.join(image_folder, "*.tif"))
    labels = glob(os.path.join(label_folder, "*.tif"))
    assert len(images) == len(labels)
    assert len(images) > 0

    msa, sa50, sa75 = [], [], []
    for image_path, label_path in tqdm(
        zip(images, labels), total=len(images), disable=not verbose,
        desc=f"Evaluate {image_folder}"
    ):
        image = imageio.imread(image_path)
        segmentation = seg_function(image)

        gt = imageio.imread(label_path)
        this_msa, scores = mean_segmentation_accuracy(segmentation, gt, return_accuracies=True)

        msa.append(this_msa)
        sa50.append(scores[0])
        sa75.append(scores[5])

    results = pd.DataFrame({
        "msa": [np.mean(msa)],
        "sa50": [np.mean(sa50)],
        "sa75": [np.mean(sa75)],
    })
    return results


def segment_cp(image, model):
    masks = model.eval(image, channels=[0, 0])[0]
    return masks


def segment_sam(image, model):
    pass


# TODO
def evaluate_sam(image_folder, label_folder, model_path, use_ais):
    if use_ais:
        model = ""
    else:
        model = ""
    segment = partial(segment_sam, model=model)
    return _evaluate(image_folder, label_folder, segment)


def evaluate_cellpose(image_folder, label_folder, model_path):
    import torch
    from cellpose import models

    use_gpu = torch.cuda.is_available()
    if model_path is None:  # We evaluate cyto2 as default model
        model = models.Cellpose(gpu=use_gpu, model_type="cyto2")
    else:
        assert os.path.exists(model_path)
        raise NotImplementedError

    segment = partial(segment_cp, model=model)
    return _evaluate(image_folder, label_folder, segment)
