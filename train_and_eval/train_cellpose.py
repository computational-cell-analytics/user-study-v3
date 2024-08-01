import os
import argparse

from cellpose import train, core, io, models

from common import get_train_and_val_images, MODEL_ROOT


def run_training(name):
    """Reference: https://github.com/anwai98/tukra/blob/master/tukra/training/cellpose2.py
    """
    train_images, train_labels, val_images, val_labels = get_train_and_val_images(name, for_cellpose=True)

    # Starting the logger which logs the training across epochs.
    io.logger_setup()

    # The choice of model initialized with pretrained "cyto2"
    model = models.CellposeModel(gpu=core.use_gpu(), model_type="cyto2")

    save_root = os.path.join(MODEL_ROOT, f"cellpose/v8/{name}")
    train.train_seg(
        net=model.net,
        train_files=train_images,
        train_labels_files=train_labels,
        test_files=val_images,
        test_labels_files=val_labels,
        channels=[0, 0],
        save_path=save_root,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    run_training(args.name)


if __name__ == "__main__":
    main()
