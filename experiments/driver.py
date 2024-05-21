import pickle as pkl
import time
import argparse
import subprocess
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import statistics as stats


from dataclasses import dataclass
from collections import defaultdict
from collections import OrderedDict

from trial import Trial


metadata = {"mtx-kmeans-2":("purple", "x"),
            "shuffle-kmeans":("teal", "o"), 
            "cuml-kmeans":("lime", "v"),
            "mtx-kmeans-bulk":("crimson", "s"),
            "mtx-kmeans-norm":("orange", "^"),
            "pytorch":("black", "+")}
font = FontProperties()
font.set_family("monospace")


class KmeansTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "centroids_runtime",
                "memcpy_runtime",
                "d", "k", "n", 
                "mem"]

    def __init__(self):
        super().__init__()


    def compute_time_avg(self, pattern, output):
        match = re.findall(pattern, output)
        avg_time = stats.mean(np.array([float(t) for t in match])) 
        return avg_time


    def parse_output(self, result):

        output = result.stdout
        args = result.args

        print(output)
        
        pattern = r"GPU_Kmeans: (\d+\.\d+)s"
        match = re.search(pattern, output)
        time = float(match.group(1))

        pattern = r"MEMORY FOOTPRINT: (\d+) MB"
        match = re.search(pattern, output)
        mem = float(match.group(1))

        pattern = r"memcpy time: (\d+\.\d+)"
        memcpy_time = self.compute_time_avg(pattern, output)

        pattern = r"clusters_argmin_shfl time: (\d+\.\d+)"
        argmin_time = self.compute_time_avg(pattern, output)

        pattern = r"compute_centroids time: (\d+\.\d+)"
        centroids_time = self.compute_time_avg(pattern, output)

        pattern = r"compute_distances time: (\d+\.\d+)"
        dist_time = self.compute_time_avg(pattern, output)

        pattern = r"-d (\d+)"
        match = re.search(pattern, args)
        d = int(match.group(1))

        pattern = r"-k (\d+)"
        match = re.search(pattern, args)
        k = int(match.group(1))

        pattern = r"-n (\d+)"
        match = re.findall(pattern, args)
        n = int(match[1])

        return {"runtime":time,
                "dist_runtime":dist_time,
                "argmin_runtime":argmin_time,
                "centroids_runtime":centroids_time,
                "memcpy_runtime":memcpy_time,
                "d":d, "k":k, "n":n,
                "mem":mem
                }


class CuMLTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "centroids_runtime",
                "memcpy_runtime",
                "d", "k", "n", 
                "mem"]

    def __init__(self):
        super().__init__()



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
        for _ in range(args.niters):
            stime = time.time()
            torch.cdist(points, centroids)
            etime = time.time()
            total_time += (etime - stime)

        t = total_time / args.niters
        print(f"Total time: {t}")

        pytorch_dist_results.add_result(args.n, args.k, d, t, 0)

    pytorch_dist_results.save(f"./distances-pytorch-n{args.n}-k{args.k}")


def run_torch_kmeans(args):

    import torch
    from torch_kmeans import KMeans

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
        for _ in range(args.niters):
            stime = time.time()
            model.fit(points)
            etime = time.time()
            total_time += (etime - stime)

        t = total_time / args.niters
        print(f"Total time: {t}")

        pytorch_kmeans_results.add_result(args.n, args.k, d, t, 0)

    pytorch_kmeans_results.save(f"./pytorch-kmeans-n{args.n}-k{args.k}")



         

def run_cuml_kmeans(args):
    
    from cuml.cluster import KMeans

    import cudf

    trial_manager = CuMLTrial()

    if args.itervar=="d":
        iter_var = np.arange(2, args.d+1, 2)
        suffix = f"-n{args.n}-k{args.k}"
    elif args.itervar=="n":
        iter_var = np.arange(args.k, args.n+1, 1000)
        suffix = f"-d{args.d}-k{args.k}"
    elif args.itervar=="k":
        iter_var = np.arange(2, args.k+1, 10)
        suffix = f"-n{args.n}-d{args.d}"
    else:
        iter_var = [1]
        suffix = f"-n{args.n}-d{args.d}-k{args.k}"

    for var in iter_var:

        if args.itervar=="d":
            n = args.n
            k = args.k
            d = var
        elif args.itervar=="n":
            k = args.k
            d = args.d
            n = var
        elif args.itervar=="k":
            n = args.n
            d = args.d
            k = var
        else:
            n,d,k=args.n,args.d,args.k

        print(f"Running cuml n={n}, k={k}, d={d}")

        # Generate random data 
        points = np.random.rand(n, d)
        d_points = cudf.DataFrame(points)

        memcpy_time = 0 
        compute_time = 0
        
        # Init Kmeans
        kmeans = KMeans(n_clusters=k, max_iter=2, verbose=6)

        # Warm up
        kmeans.fit(d_points)

        # Run Kmeans
        for i in range(args.niters):

            stime = time.time()
            d_points = cudf.DataFrame(points)
            etime = time.time()

            if i>0:
                memcpy_time += (etime - stime)

            stime = time.time()
            kmeans.fit(d_points)
            etime = time.time()

            if i>0:
                compute_time += (etime - stime)
        
        compute_time = compute_time / (args.niters - 1)
        memcpy_time = memcpy_time / (args.niters - 1)

        print(f"Compute Time: {compute_time}s")
        print(f"Memcpy Time: {memcpy_time}s")

        trial_manager.add_sample({"memcpy_runtime":memcpy_time,
                                  "runtime": compute_time,
                                  "d":d, "n":n, "k":k}
                                 )
        print(trial_manager.df)

    trial_manager.save(f"./{args.fname}{suffix}")


def run_our_kmeans(args):

    cmd = f"../build/src/bin/{args.bin}  -m 2  -s 1 --runs {args.niters} " 

    iter_var = [] 
    if args.itervar=="d":
        iter_var = np.arange(2, args.d+1, 2)
        cmd += f"-n {args.n} "
        cmd += f"-k {args.k} "
        suffix = f"-n{args.n}-k{args.k}"
    elif args.itervar=="n":
        iter_var = np.arange(args.k, args.n+1, 1000)
        cmd += f"-k {args.k} "
        cmd += f"-d {args.d} "
        suffix = f"-d{args.d}-k{args.k}"
    elif args.itervar=="k":
        iter_var = np.arange(2, args.k+1, 100)
        cmd += f"-n {args.n} "
        cmd += f"-d {args.d} "
        suffix = f"-n{args.n}-d{args.d}"

    # Add srun if using SLURM
    if args.slurm:
        cmd = "srun -G 1 -n 1 " + cmd

    trial_manager = KmeansTrial()
    
    for var in iter_var:

        if args.itervar=="d":
            n = args.n
            k = args.k
            d = var
            cmd_curr = cmd + f"-d {var}"
        elif args.itervar=="n":
            k = args.k
            d = args.d
            n = var
            cmd_curr = cmd + f"-n {var}"
        elif args.itervar=="k":
            n = args.n
            d = args.d
            k = var
            cmd_curr = cmd + f"-k {var}"

        print(f"Executing {cmd_curr}..")

        trial_manager.run_trial(cmd_curr)
        print(trial_manager.df)


    trial_manager.save(f"./{args.fname}{suffix}")


def run_distances(args):

    cmd = f"../build/src/bin/gpukmeans -n {args.n} -k {args.k} -m 2  -s 1 --runs {args.niters} " 

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
        print(distances_results.df)
    
    distances_results.save(f"./distances-{args.fname}-n{args.n}-k{args.k}")


def get_version_name(fpath):
    if "cuml" in fpath:
        return "cuml-kmeans"
    elif "bulk" in fpath:
        return "mtx-kmeans-bulk"
    elif "norm" in fpath:
        return "mtx-kmeans-norm"
    elif "shuffle" in fpath:
        return "shuffle-kmeans"


def filter_files(args, filenames):

    if args.itervar=="n":
        filenames = list(filter(lambda f: f"-d{args.d}-k{args.k}.pkl" in f and "distances" not in f, filenames))
    elif args.itervar=="k":
        filenames = list(filter(lambda f: f"-n{args.n}-d{args.d}.pkl" in f and "distances" not in f, filenames))
    elif args.itervar=="d":
        filenames = list(filter(lambda f: f"-n{args.n}-k{args.k}.pkl" in f and "distances" not in f, filenames))
    else:
        raise Exception(f"Invalid itervar: {args.itervar}")

    return filenames


def plot_runtime(args):

    filenames = os.listdir(f"./{args.platform}")
    filenames = filter_files(args, filenames)
    filenames = list(map(lambda f: f"./{args.platform}/{f}", filenames))
    print(filenames)

    data_dict = defaultdict(lambda: []) 

    if args.itervar=="d":
        _markevery=1
    else:
        _markevery=4

    for filename in filenames:
        
        version_name = get_version_name(filename)
        
        with open(filename, 'rb') as file:
            results = pkl.load(file) 
            data_dict[version_name] = results.get_result_data("runtime")
            inds = results.get_result_data(args.itervar)

    hfont = {'fontname':'monospace'}

    for version in data_dict.keys():
        plt.plot(inds, data_dict[version], 
                 label=version, markersize=7, marker=metadata[version][1], color=metadata[version][0],
                 markevery=_markevery)

    plt.xlabel(args.itervar)
    plt.ylabel("Runtime (s)")
    plt.yscale("log")

    if args.itervar=="n":
        title_suffix = f"(d={args.d} k={args.k})"
    elif args.itervar=="k":
        title_suffix = f"(d={args.d} n={args.n})"
    elif args.itervar=="d":
        title_suffix = f"(n={args.n} k={args.k})"
    
    plt.title(f"Runtime of 2 Iterations of K-means {title_suffix}", **hfont)
    plt.legend()

    plt.savefig(f"./{args.platform}/kmeans-{title_suffix}", bbox_inches='tight')


def plot_mem(args):

    filenames = os.listdir(f"./{args.platform}")
    filenames = list(filter(lambda f: "-n1" in f and "-k1" in f and "distance" not in f, filenames))
    filenames = list(map(lambda f: f"./{args.platform}/{f}", filenames))

    print(filenames)

    param_inds = {"n=1000, k=10, d=64":0,
                  "n=100000, k=1000, d=64":1,
                  "n=100000, k=10000, d=64":2,
                  "n=1000000, k=10000, d=64":3}
    n_params = len(param_inds)

    data_dict = defaultdict(lambda: [0]*n_params)  
    for filename in filenames:
        
        
        version_name = get_version_name(filename)

        if version_name=="cuml-kmeans":
            continue


        if version_name=="mtx-kmeans-norm":
            params = "-n"+filename.split("-n")[2].split(".pkl")[0]
        else:
            params = "-n"+filename.split("-n")[1].split(".pkl")[0]

        n = int(params.split("-n")[1].split("-k")[0])

        k = int(params.split("-k")[1].split(".pkl")[0])
        d = 64
        params = f"n={n}, k={k}, d={d}"
        
        with open(filename, 'rb') as file:
            print(filename)
            results = pkl.load(file)
            data_dict[version_name][param_inds[params]] = (results.get_result_data("mem")[-1])


    
    ind = np.arange(len(data_dict["mtx-kmeans-bulk"]))
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
    plt.yscale("log")
    plt.legend()

    plt.savefig(f"./{args.platform}/kmeans-mem", bbox_inches='tight')


def plot_distance_runtime(args):
    
    filenames = list(filter(lambda f: f"distances" in f and f"n{args.n}-k{args.k}" in f and ".png" not in f, os.listdir(f"./{args.platform}")))
    print(filenames)
    
    data_dict = {}

    for filename in filenames:
        
        version = get_version_name(filename)

        with open(f"./{args.platform}/{filename}", 'rb') as file:
            results = pkl.load(file) 
            data_dict[version] = results.get_result_data("runtime")

    name_dict = {"mtx-kmeans-bulk":"bulk",
                 "mtx-kmeans-norm":"norm",
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
    parser.add_argument("--itervar", type=str)
    parser.add_argument("--bin", type=str)
    parser.add_argument("--niters", type=int, default=10)
    parser.add_argument("--maxiters", type=int, default=2)
    
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



