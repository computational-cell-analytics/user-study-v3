import argparse

from micro_sam.sam_annotator import image_folder_annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    input_folder = "data/for_annotation/split1"
    output_folder = f"data/annotations/v7/{args.name}"

    embedding_path = "./embeddings/v7"
    checkpoint = f"models/{args.name}/checkpoints/organoid_model/latest.pt"

    image_folder_annotator(
        input_folder, output_folder, pattern="*.tif",
        model_type="vit_b", embedding_path=embedding_path,
        precompute_amg_state=True, checkpoint_path=checkpoint
    )


if __name__ == "__main__":
    main()
