import os
import argparse

from common import get_test_split_folders, evaluate_cellpose, get_all_cellpose_models


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

    image_folder, label_folder = get_test_split_folders()

    for name, model_path in models.items():
        print("Visualizing", name)
        evaluate_cellpose(image_folder, label_folder, model_path, visualize=True)


if __name__ == "__main__":
    main()
