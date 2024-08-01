import argparse
import os

import pandas as pd
from common import get_all_cellpose_models, evaluate_cellpose, get_organoidnet_folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-a", "--evaluate_all", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    if args.evaluate_all:
        models = {"cyto2": None}
        models.update(get_all_cellpose_models())
    else:
        models = {"cyto2": None} if args.model_path is None else {os.path.basename(args.model_path): args.model_path}

    image_folder, label_folder = get_organoidnet_folders()

    results = []
    for name, model_path in models.items():
        print("Evaluating", name)
        result = evaluate_cellpose(image_folder, label_folder, model_path)
        result.insert(loc=0, column="Model", value=[name])
        results.append(result)
    results = pd.concat(results)

    print(results)
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
