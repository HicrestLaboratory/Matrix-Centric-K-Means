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
import scipy.stats as st



from dataclasses import dataclass
from collections import defaultdict
from collections import OrderedDict

from trial import Trial


metadata = {"mtx-kmeans-2":("purple", "x"),
            "shuffle-kmeans":("teal", "o", "/"), 
            "cuml-kmeans":("lime", "v", "."),
            "mtx-kmeans-bulk":("crimson", "s", "x"),
            "mtx-kmeans-norm":("purple", "x", "o"),
            "raft-kmeans":("teal", "+")}

default_k_vals = [10, 50, 100, 500, 1000, 2000]


class KmeansTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "centroids_runtime",
                "bmult_runtime",
                "iterations",
                "memcpy_runtime",
                "centroid_init_runtime",
                "d", "k", "n", 
                "mem",
                "name",
                "score"]

    def __init__(self):
        super().__init__()


    def compute_time_avg(self, pattern, output, n_trials):
        
        # Avoid warm-up
        if "memcpy" in pattern:
            start_idx = 1
        else:
            start_idx = 2 

        match = re.findall(pattern, output)
        lst = [float(t) for t in match[start_idx:]]
        avg_time = stats.mean(lst)
        stdev = stats.stdev(lst)
        return avg_time,stdev


    def get_iterations(self, output, n_trials, maxiters):

        iters = 0

        for line in output.splitlines():

            if "converged at iteration" in line:
                split_str = line.split("iteration")[1].split("-")[0]
                curr_iters = int(split_str)
                iters += curr_iters

            elif "NOT" in line:
                iters += maxiters

        return iters // n_trials


    def parse_output(self, result, n_trials, maxiters):

        output = result.stdout
        args = result.args

        print(output)
        
        pattern = r"Time: (\d+\.\d+)"
        time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"MEMORY FOOTPRINT: (\d+) MB"
        match = re.search(pattern, output)
        mem = float(match.group(1))

        pattern = r"memcpy time: (\d+\.\d+)"
        memcpy_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"clusters_argmin_shfl time: (\d+\.\d+)"
        argmin_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"compute_centroids time: (\d+\.\d+)"
        centroids_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"compute_distances time: (\d+\.\d+)"
        dist_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"b-mult time: (\d+\.\d+)"
        bmult_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"init_centroids time: (\d+\.\d+)"
        centroid_init_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"Score: -?(\d+\.\d+)"
        match = re.search(pattern, output)
        score = float(match.group(1))

        iters = self.get_iterations(output, n_trials, maxiters)

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
                "centroid_init_runtime": centroid_init_time,
                "d":d, "k":k, "n":n,
                "mem":mem,
                "iterations":iters,
                "bmult_runtime":bmult_time,
                "score":score,
                }




class RaftTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "fused_runtime",
                "centroids_runtime",
                "centroid_init_runtime",
                "d", "k", "n", 
                "iterations",
                "name",
                "score"]

    def __init__(self):
        super().__init__()


    def compute_time_avg(self, pattern, output, n_trials):
        
        # Avoid warm-up
        if "memcpy" in pattern:
            start_idx = 1
        else:
            start_idx = 2 

        match = re.findall(pattern, output)
        lst = [float(t) for t in match[start_idx:]]
        avg_time = stats.mean(lst)
        stdev = stats.stdev(lst)
        return avg_time,stdev


    def parse_output(self, result, n_trials, maxiters):

        output = result.stdout

        print(output)

        pattern = r"n:(\d+) d:(\d+) k:(\d+)"
        matches = re.findall(pattern, output)[0]
        n,d,k = int(matches[0]), int(matches[1]), int(matches[2]) 

        get_time_str = lambda s: re.escape(s) + r": (\d+\.\d+)s"

        #pattern = get_time_str("centroids-update-time")
        #centroids_time = self.compute_time_avg(pattern, output, n_trials)

        #pattern = get_time_str("dist-argmin-time")
        #fused_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = get_time_str("kmeans-time")
        match = re.search(pattern, output)
        kmeans_time = float(match.group(1))

        pattern = r"kmeans-score: (\d+\.\d+)"
        match = re.search(pattern, output)
        score = float(match.group(1))

        pattern = r"kmeans-iterations: (\d+)"
        match = re.search(pattern, output)
        iters = int(match.group(1))

        pattern = get_time_str("init-time")
        match = re.search(pattern, output)
        centroid_init_time = float(match.group(1))

        return {"runtime":kmeans_time,
                "centroid_init_runtime":centroid_init_time,
                "d":d, "k":k, "n":n,
                "score":score,
                "iterations":iters
                }

def run_raft_kmeans(args):

    trial_manager = RaftTrial()

    if args.itervar=="d":
        iter_var = np.arange(2, args.d+1, 2)
        suffix = f"-n{args.n}-k{args.k}"
    elif args.itervar=="n":
        iter_var = np.arange(args.k, args.n+1, 1000)
        suffix = f"-d{args.d}-k{args.k}"
    elif args.itervar=="k":
        iter_var = args.kvals
        suffix = f"-n{args.n}-d{args.d}"

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

        print(f"Running raft with n={n} d={d} k={k}")

        if args.slurm:
            cmd = "srun -n 1 -G 1 "
        else:
            cmd = ""

        check = int(args.check)

        cmd += f"./raft-bench/build/kmeans {n} {d} {k} {args.maxiters} {check} {args.tol} {args.init} "
        if args.infile:
            cmd += f"{args.infile}"
        print(f"Executing {cmd}")

        try:
            trial_manager.run_trial(cmd, args.ntrials, args.maxiters) 
            print(trial_manager.df)
        except Exception as err:
            print(err)

    trial_manager.save(f"{args.fname}{suffix}")






def run_our_kmeans(args):

    cmd = f"../build/src/bin/{args.bin}  -m {args.maxiters}  -s 1 --runs {args.ntrials}  -t {args.tol} -c {int(args.check)} -p {args.pwdist} --init {args.init} " 
    if args.infile:
        cmd += f"-i {args.infile} "

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
        iter_var = args.kvals
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

        trial_manager.run_trial(cmd_curr, args.ntrials, args.maxiters)
        print(trial_manager.df)


    trial_manager.save(f"./{args.fname}{suffix}")


def get_version_name(fpath):
    if "cuml" in fpath:
        return "cuml-kmeans"
    elif "bulk" in fpath:
        return "mtx-kmeans-bulk"
    elif "norm" in fpath:
        return "mtx-kmeans-norm"
    elif "shuffle" in fpath:
        return "shuffle-kmeans"
    elif "raft" in fpath:
        return "raft-kmeans"


def filter_files(args, filenames):

    if args.itervar=="n":
        filenames = list(filter(lambda f: f"-d{args.d}-k{args.k}.csv" in f and "distances" not in f, filenames))
    elif args.itervar=="k":
        filenames = list(filter(lambda f: f"-n{args.n}-d{args.d}.csv" in f and "distances" not in f, filenames))
    elif args.itervar=="d":
        filenames = list(filter(lambda f: f"-n{args.n}-k{args.k}.csv" in f and "distances" not in f, filenames))
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
        _markevery=4
    else:
        _markevery=1

    for filename in filenames:
        
        version_name = get_version_name(filename)
        print(version_name)
        
        with open(filename, 'rb') as file:
            results = pd.read_csv(file) 
            data_dict[version_name] = results["runtime"][0::_markevery]
            print(data_dict[version_name])
            inds = results[args.itervar][0::_markevery]


    width = 0.35
    offset = width
    i = 0

    for version in data_dict.keys():
        plt.plot(inds, data_dict[version], label=version, marker=metadata[version][1],
                 markersize=7,
                 color=metadata[version][0])
        i+=1

    plt.xlabel(args.itervar)
    plt.ylabel("Runtime (s)")
    #plt.yscale("log")

    if args.itervar=="n":
        title_suffix = f"(d={args.d} k={args.k})"
    elif args.itervar=="k":
        title_suffix = f"(d={args.d} n={args.n})"
    elif args.itervar=="d":
        title_suffix = f"(n={args.n} k={args.k})"
    
    plt.title(f"Runtime of 100 Iterations of K-means {title_suffix}")
    plt.legend()

    plt.savefig(f"./{args.platform}/kmeans-{title_suffix}", bbox_inches='tight')
    plt.clf()


def plot_breakdown(args):
    filenames = os.listdir(f"./{args.platform}")
    filenames = filter_files(args, filenames)
    filenames = list(map(lambda f: f"./{args.platform}/{f}", filenames))
    print(filenames)

    data_dict = defaultdict(lambda: []) 

    if args.itervar=="d":
        _markevery=4
    else:
        _markevery=1

    for filename in filenames:
        
        version_name = get_version_name(filename)
        
        with open(filename, 'rb') as file:
            results = pd.read_csv(file) 
            data_dict[version_name] = results
            print(data_dict[version_name])
            inds = results[args.itervar][0::_markevery]


    width = 0.5
    offset = width

    breakdown_colors = {"dist_runtime":"purple",
                          "argmin_runtime":"blue",
                          "memcpy_runtime":"crimson",
                          "centroids_runtime":"peru"}

    if args.itervar=="n":
        title_suffix = f"(d={args.d} k={args.k})"
    elif args.itervar=="k":
        title_suffix = f"(d={args.d} n={args.n})"
    elif args.itervar=="d":
        title_suffix = f"(n={args.n} k={args.k})"



    for version in data_dict.keys():

        if version=="cuml-kmeans":
            continue

        bottom = np.zeros(len(inds))

        for var in ["dist_runtime", "argmin_runtime", "centroids_runtime"]:
            plt.bar(np.arange(len(inds)) - (offset) + (offset), 
                     data_dict[version][var][0::_markevery], width, 
                     label=var,
                     bottom = bottom,
                     color=breakdown_colors[var])
            bottom += data_dict[version][var][0::_markevery]
            plt.xlabel(args.itervar)
            plt.xticks(np.arange(len(inds)), labels=inds, rotation=45)
            plt.ylabel("Runtime (s)")
            #plt.yscale("log")
            plt.title(f"Runtime Breakdown of {version} {title_suffix}")
            plt.legend()

        plt.savefig(f"./{args.platform}/{version}-{title_suffix}-breakdown", bbox_inches='tight')
        plt.clf()




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


def plot_kernel_runtime(args):

    filenames = os.listdir(f"./{args.platform}")
    filenames = filter_files(args, filenames)
    filenames = list(map(lambda f: f"./{args.platform}/{f}", filenames))
    
    print(filenames)
    
    data_dict = {}

    for filename in filenames:
        
        version = get_version_name(filename)

        if version=="cuml-kmeans":
            continue

        with open(f"./{filename}", 'rb') as file:

            results = pd.read_csv(file) 

            # For non-raft, sum runtime of distances and argmin kernel to compare against fused kernel
            if args.kernel=="fused_runtime" and version!="raft-kmeans":
                data_dict[version] = results["dist_runtime"] + results["argmin_runtime"]
            else:
                data_dict[version] = results[args.kernel]

            inds = results[args.itervar]

    name_dict = {"mtx-kmeans-bulk":"bulk",
                 "mtx-kmeans-norm":"norm",
                 "shuffle-kmeans": "shuffle",
                 "raft-kmeans":"raft"}

    for version in name_dict.keys():
        plt.plot(inds, data_dict[version],
                 label=version, color=metadata[version][0],
                 marker=metadata[version][1])

    kernelnames = {"dist_runtime": "Pairwise Distances",
                   "centroids_runtime": "Centroids Update",
                   "argmin_runtime": "Argmin",
                   "fused_runtime": "Fused-Cluster"}

    plt.ylabel("Runtime (s)")
    plt.yscale("log")
    plt.xlabel(args.itervar)
    plt.title(f"Runtime of {kernelnames[args.kernel]} (n={args.n}, d={args.d})")
    plt.legend()

    plt.savefig(f"./{args.platform}/{args.kernel}-n{args.n}-d{args.d}", bbox_inches='tight')
            


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--fname", type=str)
    parser.add_argument("--slurm", action='store_true')
    parser.add_argument("--action", type=str)
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--platform", type=str)
    parser.add_argument("--itervar", type=str)
    parser.add_argument("--bin", type=str)
    parser.add_argument("--ntrials", type=int, default=10)
    parser.add_argument("--maxiters", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.00000001)
    parser.add_argument("--infile", type=str)
    parser.add_argument("--pwdist", type=str)
    parser.add_argument("--init", type=str)
    parser.add_argument("--check", action='store_true' )
    parser.add_argument("--kvals", nargs='+', default=default_k_vals)
    
    args = parser.parse_args()


    if args.action=="plot-runtime":
        plot_breakdown(args)
        plot_runtime(args)
    elif args.action=="plot-mem":
        plot_mem(args)
    elif args.action=="our-kmeans":
        run_our_kmeans(args)
    elif args.action=="raft-kmeans":
        run_raft_kmeans(args)


