import argparse

import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import torch_em

from elf.evaluation import mean_segmentation_accuracy
from torch_em.util.prediction import predict_with_padding
from torch_em.util.segmentation import watershed_from_center_and_boundary_distances

# ROOT = "../data"
ROOT = "/scratch-emmy/projects/nim00007/user-study/data"


def run_segmentation(model, image):
    input_ = torch_em.transform.raw.standardize(image)
    pred = predict_with_padding(model, input_, (16, 16), "cuda")
    pred = pred.squeeze(0)
    fg, cdist, bdist = pred
    seg = watershed_from_center_and_boundary_distances(cdist, bdist, fg, min_size=100)
    return seg


def evaluate_organoidnet(model_name):
    print("Evaluating", model_name, "...")

    image_folder = os.path.join(ROOT, "datasets", "organoidnet", "Test", "Images")
    label_folder = os.path.join(ROOT, "datasets", "organoidnet", "Test", "Masks")

    images = sorted(glob(os.path.join(image_folder, "*.tif")))
    labels = sorted(glob(os.path.join(label_folder, "*.tif")))

    model_path = f"checkpoints/distance_unet/{model_name}"
    model = torch_em.util.load_model(model_path)

    msas, sa50s, sa75s = [], [], []
    for im_path, label_path in zip(images, labels):
        image = imageio.imread(im_path)
        image = image.mean(axis=-1)
        segmentation = run_segmentation(model, image)
        gt = imageio.imread(label_path)
        msa, accs = mean_segmentation_accuracy(segmentation, gt, return_accuracies=True)
        sa50, sa75 = accs[0], accs[5]
        msas.append(msa)
        sa50s.append(sa50)
        sa75s.append(sa75)

    msa = np.mean(msas)
    sa50 = np.mean(sa50s)
    sa75 = np.mean(sa75s)

    results = pd.DataFrame({
        "model": [model_name], "msa": [msa], "sa50": [sa50], "sa75": [sa75],
    })
    return results


def evaluate_all_models():
    models = ("organoidnet", "orga/v1", "combined")
    results = []
    for model in models:
        result = evaluate_organoidnet(model)
        results.append(result)
    results = pd.concat(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name")
    args = parser.parse_args()

    if args.model_name is None:
        result = evaluate_all_models()
    else:
        result = evaluate_organoidnet(args.model_name)
    print(result)


# Results:
#        model       msa      sa50      sa75
#  organoidnet  0.262385  0.504349  0.253957
#      orga/v1  0.054521  0.136979  0.037297
#     combined  0.253246  0.496628  0.236607
if __name__ == "__main__":
    main()
