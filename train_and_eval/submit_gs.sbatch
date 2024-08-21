#!/bin/bash

#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -t 90
#SBATCH -c 6
#SBATCH --mem 128G

/scratch-grete/usr/nimcpape/software/mamba/envs/main/bin/python run_gs.py $1 --version $2
