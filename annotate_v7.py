import argparse
import os

from micro_sam.sam_annotator import image_folder_annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    input_folder = "data/for_annotation/split3"
    output_folder = f"data/annotations/v7/{args.name}"

    embedding_path = "./embeddings/v7"
    checkpoint = f"models/{args.name}/checkpoints/organoid_model/best.pt"
    # Check for alternate model location.
    if not os.path.exists(checkpoint):
        checkpoint = f"models/micro-sam/v5/{args.name}/checkpoints/organoid_model/best.pt"

    image_folder_annotator(
        input_folder, output_folder, pattern="*.tif",
        model_type="vit_b", embedding_path=embedding_path,
        precompute_amg_state=True, checkpoint_path=checkpoint
    )


if __name__ == "__main__":
    main()
