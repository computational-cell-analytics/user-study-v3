# Organoid Instance Segmentation

## Our Organoid Instance Segmentation Dataset

This dataset contains 30 images with 6252 annotated organoids. The organoid images were taken by the Grade and Schneider group at UMG.
Annotations were done by Sushmita for a part of the images (based on CellPose segmentations)
and in the micro_sam user study for another part.
The annotations from the latter part are based on consensus annotations across multiple annotators.
The scripts for dataset creation are [here](https://github.com/computational-cell-analytics/user-study-v3/blob/master/data_prep/create_consensus_annotations.py) (consensuns annotations) and [here](https://github.com/computational-cell-analytics/user-study-v3/blob/master/data_prep/create_orga_dataset.py) (creation of the dataset splits).
For shorthand we will call this dataset `orga` below.
You can find the dataset at `/scratch-grete/projects/nim00007/data/pdo/organoid_segmentation_dataset/v2`.

## Evaluation of Organoid Instance Segmentation

We now have access to three different organoid datasets that were imaged with incuyte:
- `orga` dataset (see above)
- `organoidnet` dataset from the Alves group.
- `sartorius` dataset from internal Sartorius annotation efforts.

As a next step we should evaluate the segmentation quality of models trained on these datasets .
The scripts here can be used for model training and evaluation:
- `train_distance_unet.py`: To train a U-Net for distance based segmentation on one or multiple datasets.
- `evaluate_organoidnet.py`: To evaluate segmentation results on the `organoidnet` test split.
- `evaluate_orga_test_split.py`: To evaluate segmentation results on the `orga` test split.

The models trained so far are available here: `/scratch-grete/projects/nim00007/data/pdo/organoid_segmentation_dataset/models`. (Only the model trained on `organoidnet` for now.)

Below are the initial evaluation results for `orga` vs. `organoidnet` dataset.
Note that this was performed with a preliminary version of the `orga` dataset.

| Training Data      | organoidnet (mSA on test set) | orga (mSA on test set) |
|:-------------------|------------------------------:|-----------------------:|
| organoidnet        | 0.50                          | 0.39                   |
| orga               | 0.14                          | 0.63                   |
| organoidnet + orga | 0.50                          | 0.67                   |

See [here](https://docs.google.com/presentation/d/1z6herIjWxrmxhqzY3keSNQIZH-oBl91zZ94IZSQvPVI/edit?usp=sharing) for the presentation with an overview of this initial analysis.

## Further steps / TODOs

- Run evaluation from above with CellPose
- Create sample timeseries with current best method for segmentation and tracking
- Provide this to Alves group
