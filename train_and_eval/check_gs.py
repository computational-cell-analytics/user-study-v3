import os
import warnings
import torch

# model_root = "/scratch-emmy/projects/nim00007/user-study/models/micro-sam"
model_root = "/mnt/lustre-grete/usr/u12086/user-study-cp/models/micro-sam"
annotators = ["anwai", "caro", "constantin", "luca", "marei"]
versions = ["v5", "v7"]

for version in versions:
    for name in annotators:
        checkpoint = os.path.join(model_root, version, name, "checkpoints", "organoid_model", "best.pt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if "grid_search" not in state:
            print("Grid search results are missing for", name, version)

print("done")
