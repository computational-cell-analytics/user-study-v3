import argparse
import os

import pandas as pd
from common import get_test_split_folders, evaluate_cellpose, get_all_cellpose_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-a", "--evaluate_all", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()

    if args.evaluate_all:
        models = {"cyto2": None}
        models.update(get_all_cellpose_models())
    else:
        models = {"cyto2": None} if args.model_path is None else {os.path.basename(args.model_path): args.model_path}

    image_folder, label_folder = get_test_split_folders()

    if args.save:
        prediction_root = "/scratch-emmy/projects/nim00007/user-study/predictions/test_images"
    else:
        prediction_root = None

    results = []
    for name, model_path in models.items():
        print("Evaluating", name)
        result = evaluate_cellpose(
            image_folder, label_folder, model_path,
            prediction_root=prediction_root, prediction_name=f"cellpose/{name}"
        )
        result.insert(loc=0, column="Model", value=[name])
        results.append(result)
    results = pd.concat(results)

    print(results)
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
