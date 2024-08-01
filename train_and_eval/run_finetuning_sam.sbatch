#!/bin/bash

#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -t 720
#SBATCH -c 10
#SBATCH --mem 128G
#SBATCH --constraint=80gb
#SBATCH -x ggpu212

/scratch-grete/usr/nimcpape/software/mamba/envs/main/bin/python train_sam.py $1
