import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy


def evaluate_cp(gt_folder, res_folder):
    gt_paths = sorted(glob(os.path.join(gt_folder, "*.tif")))
    res_paths = sorted(glob(os.path.join(res_folder, "*.npy")))
    assert len(gt_paths) == len(res_paths)

    msa, sa50, sa75 = [], [], []
    for gt_path, res_path in zip(gt_paths, res_paths):

        gt = imageio.imread(gt_path)
        seg = np.load(res_path, allow_pickle=True).item()["masks"]
        this_msa, scores = mean_segmentation_accuracy(seg, gt, return_accuracies=True)

        msa.append(this_msa)
        sa50.append(scores[0])
        sa75.append(scores[5])

    results = pd.DataFrame({
        "msa": [np.mean(msa)],
        "sa50": [np.mean(sa50)],
        "sa75": [np.mean(sa75)],
    })
    return results


# run a sanity check on cellpose predictions done via the cellpose gui
# with the v6/constantin model.
def main():
    gt_folder = "/home/pape/Work/my_projects/sam-projects/new-user-study/data/ground_truth/masks"

    cp_folder = "/home/pape/Work/my_projects/sam-projects/new-user-study/data/ground_truth/cp_no_calibration"
    res = evaluate_cp(gt_folder, cp_folder)
    print("Results without per-image calibration:")
    print(res)

    cp_folder = "/home/pape/Work/my_projects/sam-projects/new-user-study/data/ground_truth/cp_calibration"
    res = evaluate_cp(gt_folder, cp_folder)
    print("Results WITH per-image calibration:")
    print(res)


if __name__ == "__main__":
    main()
