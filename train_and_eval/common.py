import os
from functools import partial
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from tqdm import tqdm

ANNOTATORS = ["anwai", "caro", "constantin", "luca", "marei"]
# Change this for cluster
DATA_ROOT = "/scratch-emmy/projects/nim00007/user-study/data"
MODEL_ROOT = "/scratch-emmy/projects/nim00007/user-study/models"


def get_train_and_val_images(name, for_cellpose=False):
    # Get the annotated data from split 2.
    image_folder = os.path.join(DATA_ROOT, "for_annotation/split2")
    label_folder = os.path.join(
        DATA_ROOT, "annotations", "v6" if for_cellpose else "v5", f"{name}"
    )

    images = sorted(glob(os.path.join(image_folder, "*.tif")))
    if for_cellpose:
        images = [path for path in images if not path.endswith("_flows.tif")]

    labels = sorted(glob(os.path.join(label_folder, "*.npy" if for_cellpose else "*.tif")))
    assert len(images) == len(labels) == 6

    # We use im4 as val.
    train_images = images[:4] + images[-1:]
    train_labels = labels[:4] + labels[-1:]

    val_images = images[4:5]
    val_labels = labels[4:5]

    # Get the annotated data from split 3.
    image_folder = os.path.join(DATA_ROOT, "for_annotation/split3")
    label_folder = os.path.join(
        DATA_ROOT, "annotations", "v8" if for_cellpose else "v7", f"{name}"
    )

    images = sorted(glob(os.path.join(image_folder, "*.tif")))
    labels = sorted(glob(os.path.join(label_folder, "*.npy" if for_cellpose else "*.tif")))
    assert len(images) == len(labels) == 6

    train_images += images[:-1]
    train_labels += labels[:-1]

    val_images += images[-1:]
    val_labels += labels[-1:]

    return train_images, train_labels, val_images, val_labels


def get_test_split_folders():
    image_folder = os.path.join(DATA_ROOT, "ground_truth/images")
    label_folder = os.path.join(DATA_ROOT, "ground_truth/masks")
    return image_folder, label_folder


def get_organoidnet_folders():
    image_folder = os.path.join(DATA_ROOT, "datasets/organoidnet/Test/Images")
    label_folder = os.path.join(DATA_ROOT, "datasets/organoidnet/Test/Masks")
    return image_folder, label_folder


def get_all_cellpose_models():
    cp_models = {}

    v6_root = os.path.join(MODEL_ROOT, "cellpose", "v6")
    for ann in ANNOTATORS:
        pattern = os.path.join(v6_root, ann, "CP*")
        models = glob(pattern)
        assert len(models) == 1, f"{ann} : {pattern} ; {models}"
        model_path = models[0]
        cp_models[f"v6/{ann}"] = model_path

    v8_root = os.path.join(MODEL_ROOT, "cellpose", "v8")
    for ann in ANNOTATORS:
        pattern = os.path.join(v8_root, ann, "models", "cellpose*")
        models = glob(pattern)
        assert len(models) == 1, f"{ann} : {pattern} ; {models}"
        model_path = models[0]
        cp_models[f"v8/{ann}"] = model_path

    return cp_models


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


def segment_cp(image, model, diameter):
    masks = model.eval(image, channels=[0, 0], diameter=diameter)[0]
    return masks


def segment_sam(image, model):
    pass


def evaluate_sam(image_folder, label_folder, model_type, model_path, use_ais):
    import micro_sam

    if use_ais:
        if model_path is None:
            pass  # TODO
        else:
            predictor, decoder = micro_sam.instance_semgentation.get_predictor_and_decoder(model_type, model_path)
        model = micro_sam.instance_segmentation.get_amg(predictor, decoder=decoder)
    else:
        predictor = micro_sam.util.get_sam_model(model_type=model_type, checkpoint_path=model_path)
        model = micro_sam.instance_segmentation.get_amg(predictor)
    segment = partial(segment_sam, model=model)
    return _evaluate(image_folder, label_folder, segment)


def evaluate_cellpose(image_folder, label_folder, model_path):
    import torch
    from cellpose import models

    use_gpu = torch.cuda.is_available()
    if model_path is None:  # We evaluate cyto2 as default model.
        model = models.Cellpose(gpu=use_gpu, model_type="cyto2")
        diameter = None
    else:
        assert os.path.exists(model_path)
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
        diameter = model.diam_labels

    segment = partial(segment_cp, model=model, diameter=diameter)
    return _evaluate(image_folder, label_folder, segment)
