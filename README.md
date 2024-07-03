# Updated micro-sam user study


## User study strategy:

- Cluster images into three categories: large, medium, small organoids, based on initial segmentation.
- Create 3 different splits, each containing at least one image from each of the category
    - Split 1: 3 images, to be annotated with the following approaches:
        - I: Manual annotation
        - II: Segmentation and correction in CellPose (cyto2)
        - III: Segmentation and correction with micro_sam (vit_b)
        - IV: Segmentation and correction with micro_sam (vit_b_lm)
    - Split 2: 6 images, to be annotated with the following approaches:
        - V: CellPose HIL (starting from cyto2)
        - VI: micro_sam (same approach as III or IV, what worked better)
    - Split 3: 6 images, to be annotated with the following
        - VII: CellPose segmentation + correction (based on model after V)
        - VIII: micro_sam: segmentation + correction (based on model trained on data annotated in VI)

- Segmentation evaluation:
    - We evaluate CellPose and micro_sam models trained on data from (Split 2) and (Splits 2 and 3) on:
        - On our reference data (some images, with more corrections)
        - On OrganoIDNet data (the 10 images from the test split)

- Time measurements:
    - Measure the annotation times per image.
    - Measure how long processing takes (ideally we add timers to this so that no one has to wait around and actively measure)
        - Embedding and AMG state precomputation for micro-sam
        - Training (CPU + LoRA) for micro_sam
        - Segmentation for CellPose
        - HIL for CellPose


## Data overview

- Annotated reference data:
    - 15 images and masks. 
    - Needs another round of proof-reading!
- Non-annotated data:
    -  25 images without masks.
- External data:
    - OrgaSegment: intestinal organoids. Interesting, but probably too dissimilar from us.
    - OrganoIDNet: from Alves Lab. Further apart from our data then I would have thought. But we can still give it a try.
        - We should see how one of our current networks performs on it.
