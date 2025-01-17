import pickle as pkl
import time
import argparse
import subprocess
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from torch_kmeans import KMeans

from dataclasses import dataclass
from collections import defaultdict
from collections import OrderedDict

n_iters = 10

metadata = {"mtx-kmeans-2":("purple", "x"),
            "shuffle-kmeans":("teal", "o"), 
            "cuml-kmeans":("lime", "v"),
            "mtx-kmeans-spmm":("crimson", "s"),
            "mtx-kmeans-arizona":("orange", "^"),
            "pytorch":("black", "+")}

class Results:

    def __init__(self):
        self.results = []

    @dataclass
    class Result:
        n:int
        k:int
        d:int
        runtime: float
        mem: float
    
    def add_result(self, n, k, d, runtime, mem):
        self.results.append(self.Result(n, k, d, runtime, mem))

    def get_result_data(self, name):
        return list(map(lambda r: r.__dict__[name],  self.results))

    def save(self, fname):
        with open(f"{fname}.pkl", 'wb') as file:
            pkl.dump(self, file)

def run_pytorch_dist(args):
    
    import torch

    pytorch_dist_results = Results()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.dmax:
        d_vals = np.arange(2, args.dmax+1, 2)
    else:
        d_vals = [ args.d ]

    for d in d_vals:

        print(f"Running pytorch distances d={d}...")

        # Generate random data 
        points = torch.tensor(np.random.rand(args.n, d)).to(device)

        centroids = torch.tensor(np.random.rand(args.k, d)).to(device)
        
        total_time = 0
        for _ in range(n_iters):
            stime = time.time()
            torch.cdist(points, centroids)
            etime = time.time()
            total_time += (etime - stime)

        t = total_time / n_iters
        print(f"Total time: {t}")

        pytorch_dist_results.add_result(args.n, args.k, d, t, 0)

    pytorch_dist_results.save(f"./distances-pytorch-n{args.n}-k{args.k}")


def run_torch_kmeans(args):

    import torch

    pytorch_kmeans_results = Results()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Found device {device}")
    
    if args.dmax:
        d_vals = np.arange(2, args.dmax+1, 2)
    else:
        d_vals = [ args.d ]

    for d in d_vals:

        print(f"Running pytorch kmeans d={d}...")

        # Generate random data 
        points = torch.tensor(np.random.rand(1, args.n, d), dtype=torch.float32).to(device)

        model = KMeans(n_clusters=args.k, max_iter=2)

        total_time = 0
        for _ in range(n_iters):
            stime = time.time()
            model.fit(points)
            etime = time.time()
            total_time += (etime - stime)

        t = total_time / n_iters
        print(f"Total time: {t}")

        pytorch_kmeans_results.add_result(args.n, args.k, d, t, 0)

    pytorch_kmeans_results.save(f"./pytorch-kmeans-n{args.n}-k{args.k}")



         

def run_cuml_kmeans(args):
    
    from cuml.cluster import KMeans

    import cudf

    cuml_results = Results()

    if args.dmax:
        d_vals = np.arange(2, args.dmax+1, 2)
    else:
        d_vals = [ args.d ]

    for d in d_vals:

        print(f"Running cuml d={d}...")

        # Generate random data 
        points = np.random.rand(args.n, d)
        d_points = cudf.DataFrame(points)
        
        # Init Kmeans
        kmeans = KMeans(n_clusters=args.k, max_iter=2)

        # Warm up
        kmeans.fit(d_points)
        
        total_time = 0

        # Run Kmeans
        for _ in range(n_iters):
            stime = time.time()
            kmeans.fit(d_points)
            etime = time.time()
            total_time += (etime - stime)
        
        this_time = total_time / n_iters

        print(f"Time: {this_time}s")
        
        cuml_results.add_result(args.n, args.k, d, this_time, 0)

    cuml_results.save(f"./{args.fname}-n{args.n}-k{args.k}")


def run_our_kmeans(args):

    cmd = f"../build/src/bin/gpukmeans -n {args.n} -k {args.k} -m 2 -o test.out -s 1 --runs {n_iters} " 

    # Add srun if using SLURM
    if args.slurm:
        cmd = "srun -G 1 -n 1 " + cmd

    if args.dmax:
        d_vals = np.arange(2, args.dmax+1, 2)
    else:
        d_vals = [ args.d ]
    
    our_results = Results()
    distances_results = Results()
    
    for d in d_vals:

        cmd_curr = cmd + f"-d {d}"
        print(f"Executing {cmd_curr}..")

        result=subprocess.run(cmd_curr, shell=True, capture_output=True, text=True)
        result.check_returncode()

        print(result.stdout)
        pattern = r"GPU_Kmeans: (\d+\.\d+)s"
        match = re.search(pattern, result.stdout)
        time = float(match.group(1))

        
        pattern = r"MEMORY FOOTPRINT: (\d+) MB"
        match = re.search(pattern, result.stdout)
        mem = float(match.group(1))

        pattern = r"time: (\d+\.\d+)"
        match = re.search(pattern, result.stdout)
        dist_runtime = float(match.group(1))
        print(dist_runtime)

        print(f"Time: {time}s")
        
        our_results.add_result(args.n, args.k, d, time, mem)
        distances_results.add_result(args.n, args.k, d, dist_runtime, 0)
    
    our_results.save(f"./{args.fname}-n{args.n}-k{args.k}")


def run_distances(args):

    cmd = f"../build/src/bin/gpukmeans -n {args.n} -k {args.k} -m 2 -o test.out -s 1 --runs {n_iters} " 

    # Add srun if using SLURM
    if args.slurm:
        cmd = "srun -G 1 -n 1 " + cmd

    if args.dmax:
        d_vals = np.arange(2, args.dmax+1, 2)
    else:
        d_vals = [ args.d ]
    
    our_results = Results()
    distances_results = Results()
    
    for d in d_vals:

        cmd_curr = cmd + f"-d {d}"
        print(f"Executing {cmd_curr}..")

        result=subprocess.run(cmd_curr, shell=True, capture_output=True, text=True)
        result.check_returncode()

        print(result.stdout)

        pattern = r"time: (\d+\.\d+)"
        match = re.search(pattern, result.stdout)
        dist_runtime = float(match.group(1))
        
        distances_results.add_result(args.n, args.k, d, dist_runtime, 0)
    
    distances_results.save(f"./distances-{args.fname}-n{args.n}-k{args.k}")



def plot_runtime(args):

    filenames = os.listdir(f"./{args.platform}")
    filenames = list(filter(lambda f: f"-n{args.n}-k{args.k}.pkl" in f and "distances" not in f, filenames))
    filenames = list(map(lambda f: f"./{args.platform}/{f}", filenames))

    data_dict = defaultdict(lambda: []) 

    for filename in filenames:
        
        split = filename.split("-n")[0].split("/") 
        version_name = split[2]
        
        with open(filename, 'rb') as file:
            results = pkl.load(file) 
            data_dict[version_name] = results.get_result_data("runtime")

    
    for version in data_dict.keys():
        plt.plot(np.arange(2, len(data_dict[version])*2+1, 2), data_dict[version], 
                 label=version, markersize=7, marker=metadata[version][1], color=metadata[version][0])

    plt.xlabel("d")
    plt.ylabel("Runtime (s)")
    plt.yscale("log")
    plt.title(f"Runtime of 2 Iterations of K-means (n={args.n} k={args.k})")
    plt.legend()

    plt.savefig(f"./{args.platform}/kmeans-n{args.n}-k{args.k}", bbox_inches='tight')


def plot_mem(args):

    filenames = os.listdir(f"./{args.platform}")
    filenames = list(filter(lambda f: f".pkl" in f, filenames))
    filenames = list(map(lambda f: f"./{args.platform}/{f}", filenames))

    n_params = 3
    param_inds = {"n=1000, k=10, d=64":0,
                  "n=100000, k=1000, d=64":1,
                  "n=100000, k=10000, d=64":2}

    data_dict = defaultdict(lambda: [0]*3)  
    for filename in filenames:
        
        split = filename.split("-n")[0].split("/") 
        
        version_name = split[2]
        if version_name=="cuml-kmeans":
            continue

        params = "-n"+filename.split("-n")[1].split(".pkl")[0]
        n = int(params.split("-n")[1].split("-k")[0])
        k = int(params.split("-k")[1].split(".pkl")[0])
        d = 64
        params = f"n={n}, k={k}, d={d}"
        
        with open(filename, 'rb') as file:
            print(filename)
            results = pkl.load(file)
            data_dict[version_name][param_inds[params]] = (results.get_result_data("mem")[-1])


    
    ind = np.arange(len(data_dict["mtx-kmeans-2"]))
    ind *= 2
    width = 0.35

    offset = width
    i = 0

    for version in data_dict.keys():
        plt.bar(ind - (offset) + i*(offset), data_dict[version], width, label=version,
                 color=metadata[version][0])
        i+=1

    plt.xticks(ind, labels=param_inds, rotation=45)
    plt.ylabel("Memory Footprint (MB)")
    plt.title(f"Memory Footprint of K-means Algorithms")
    plt.legend()

    plt.savefig(f"./{args.platform}/kmeans-mem", bbox_inches='tight')


def plot_distance_runtime(args):
    
    filenames = list(filter(lambda f: f"distances" in f and f"n{args.n}-k{args.k}" in f and ".png" not in f, os.listdir(f"./{args.platform}")))
    
    data_dict = {}

    for filename in filenames:
        
        version = filename.split("distances-")[1].split("-n")[0]

        with open(f"./{args.platform}/{filename}", 'rb') as file:
            results = pkl.load(file) 
            data_dict[version] = results.get_result_data("runtime")

    name_dict = {"mtx-kmeans-spmm":"ours",
                 "mtx-kmeans-arizona":"arizona",
                 "pytorch":"pytorch"}

    for version in data_dict.keys():
        plt.plot(np.arange(len(data_dict[version])), data_dict[version],
                 label=name_dict[version], color=metadata[version][0],
                 marker=metadata[version][1])

    plt.ylabel("Runtime (s)")
    plt.xlabel("d")
    plt.title(f"Runtime of Distances Kernel (n={args.n}, k={args.k})")
    plt.legend()

    plt.savefig(f"./{args.platform}/distances-n{args.n}-k{args.k}", bbox_inches='tight')
            


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--dmax", type=int)
    parser.add_argument("--fname", type=str)
    parser.add_argument("--slurm", action='store_true')
    parser.add_argument("--action", type=str)
    parser.add_argument("--platform", type=str)
    
    args = parser.parse_args()


    if args.action=="plot-runtime":
        plot_runtime(args)
    elif args.action=="plot-mem":
        plot_mem(args)
    elif args.action=="plot-dist":
        plot_distance_runtime(args)
    elif args.action=="cuml-kmeans":
        run_cuml_kmeans(args)
    elif args.action=="sklearn-kmeans":
        run_sklearn_kmeans(args)
    elif args.action=="torch-kmeans":
        run_torch_kmeans(args)
    elif args.action=="our-kmeans":
        run_our_kmeans(args)
    elif args.action=="distances":
        run_distances(args)
    elif args.action=="torch-distances":
        run_pytorch_dist(args)



