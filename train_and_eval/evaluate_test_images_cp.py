import argparse
from common import get_test_split_folders, evaluate_cellpose


# TODO enable eval for all cellpose models
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    args = parser.parse_args()

    image_folder, label_folder = get_test_split_folders()
    results = evaluate_cellpose(image_folder, label_folder, args.model_path)

    print(results)


if __name__ == "__main__":
    main()
