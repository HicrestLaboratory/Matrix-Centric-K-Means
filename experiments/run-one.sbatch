#!/usr/bin/bash

conda activate rapids-24.04

python3 driver.py  --k 1000  --action cuml-kmeans-infile --infile $1 --maxiters 2
python3 driver.py --k 1000   --action our-kmeans-infile --slurm  --bin gpukmeans --infile $1 --maxiters 2 --fname mtx-kmeans-bulk
