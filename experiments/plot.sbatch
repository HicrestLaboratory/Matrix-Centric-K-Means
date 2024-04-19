#!/usr/bin/bash
#SBATCH --nodes 1
#SBATCH -C gpu
#SBATCH -t 2:00:00
#SBATCH -A m4646_g
#SBATCH -q regular


conda activate rapids-24.04


python3 plot.py --n 1000 --k 10 --dmax 64 --fname shuffle-kmeans --action our-kmeans --slurm
python3 plot.py --n 100000 --k 1000 --dmax 64 --fname shuffle-kmeans --action our-kmeans --slurm
python3 plot.py --n 100000 --k 10000 --dmax 64 --fname shuffle-kmeans --action our-kmeans --slurm

