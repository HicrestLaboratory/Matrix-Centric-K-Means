#!/usr/bin/bash
#SBATCH --nodes 1
#SBATCH -C gpu
#SBATCH -t 10:00:00
#SBATCH -A m4646_g
#SBATCH -q regular


conda activate rapids-24.04

# Scale k 
#python3 driver.py --n 1000000 --k 1000 --d 64  --itervar k --fname $2 --action cuml-kmeans --slurm  --bin $1
#python3 driver.py --n 100000 --k 1000 --d 32  --itervar k --fname $2 --action cuml-kmeans --slurm --bin $1
#python3 driver.py --n 100000 --k 1000 --d 16  --itervar k --fname $2 --action  cuml-kmeans --slurm --bin $1
#python3 driver.py --n 10000 --k 1000 --d 2 --itervar k --fname $2 --action cuml-kmeans --slurm --bin $1 

# Scale n 
#python3 driver.py --n 100000 --k 10000 --d 64  --itervar n --fname $2 --action cuml-kmeans --slurm  --bin $1
python3 driver.py --n 100000 --k 10000 --d 32  --itervar n --fname $2 --action cuml-kmeans --slurm --bin $1
python3 driver.py --n 100000 --k 1000 --d 16  --itervar n --fname $2 --action  cuml-kmeans --slurm --bin $1
python3 driver.py --n 100000 --k 10 --d 2 --itervar n --fname $2 --action cuml-kmeans --slurm --bin $1 

# Scale d 
python3 driver.py --n 1000000 --k 10000 --d 64  --itervar d --fname $2 --action cuml-kmeans --slurm --bin $1
python3 driver.py --n 100000 --k 10000 --d 64  --itervar d --fname $2 --action cuml-kmeans --slurm --bin $1
python3 driver.py --n 100000 --k 1000 --d 64  --itervar d --fname $2 --action  cuml-kmeans --slurm --bin $1
python3 driver.py --n 1000 --k 10 --d 64 --itervar d --fname $2 --action cuml-kmeans --slurm --bin $1 


