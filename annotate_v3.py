import argparse

from micro_sam.sam_annotator import image_folder_annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    input_folder = "data/for_annotation/split1"
    output_folder = f"data/annotations/v3/{args.name}"

    embedding_path = "./embeddings/v3"

    image_folder_annotator(
        input_folder, output_folder, pattern="*.tif",
        model_type="vit_b_lm", embedding_path=embedding_path,
        precompute_amg_state=True,
    )


if __name__ == "__main__":
    main()
