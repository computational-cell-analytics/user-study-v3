import argparse
import os

import pandas as pd
from common import get_test_split_folders, get_all_sam_models, evaluate_sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-a", "--evaluate_all", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    if args.evaluate_all:
        models = {"vit_b": None, "vit_b_lm": None}
        use_ais = {"vit_b": False, "vit_b_lm": True}
        custom_models, use_ais_ = get_all_sam_models()
        models.update(custom_models)
        use_ais.update(use_ais_)
    else:
        model_name = os.path.basename(args.model_path) if args.model_path else None
        models = {"vit_b_lm": None} if args.model_path is None else {model_name: args.model_path}
        use_ais = {"vit_b_lm": True} if args.model_path is None else {model_name: True}

    image_folder, label_folder = get_test_split_folders()

    results = []
    for name, model_path in models.items():
        print("Evaluating", name)
        result = evaluate_sam(image_folder, label_folder, model_path=model_path, use_ais=use_ais[name])
        result.insert(loc=0, column="Model", value=[name])
        results.append(result)
    results = pd.concat(results)

    print(results)
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
