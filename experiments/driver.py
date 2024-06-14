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
            "shuffle-kmeans":("teal", "o", "/"), 
            "cuml-kmeans":("lime", "v", "."),
            "mtx-kmeans-bulk":("crimson", "s", "x"),
            "mtx-kmeans-norm":("orange", "^", "o"),
            "raft-kmeans":("black", "+")}
font = FontProperties()
font.set_family("monospace")


class KmeansTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "centroids_runtime",
                "memcpy_runtime",
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
        s = sum([float(t) for t in match[start_idx:]])
        avg_time = s / (n_trials - 1)
        return avg_time


    def parse_output(self, result, n_trials):

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
        memcpy_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"clusters_argmin_shfl time: (\d+\.\d+)"
        argmin_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"compute_centroids time: (\d+\.\d+)"
        centroids_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"compute_distances time: (\d+\.\d+)"
        dist_time = self.compute_time_avg(pattern, output, n_trials)

        pattern = r"Score: (\d+\.\d+)"
        match = re.search(pattern, output)
        score = float(match.group(1))

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
                "mem":mem,
                "score":score,
                }


class CuMLTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "centroids_runtime",
                "memcpy_runtime",
                "d", "k", "n", 
                "mem",
                "name",
                "score"]

    def __init__(self):
        super().__init__()


class RaftTrial(Trial):

    features = ["runtime", 
                "dist_runtime",
                "argmin_runtime",
                "fused_runtime",
                "centroids_runtime",
                "d", "k", "n", 
                "name",
                "score"]

    def __init__(self):
        super().__init__()


    def parse_output(self, result, n_trials):

        output = result.stdout

        print(output)

        pattern = r"n:(\d+) d:(\d+) k:(\d+)"
        matches = re.findall(pattern, output)[0]
        n,d,k = int(matches[0]), int(matches[1]), int(matches[2]) 

        get_time_str = lambda s: re.escape(s) + r": (\d+\.\d+)s"

        pattern = get_time_str("centroids-update-time")
        match = re.search(pattern, output)
        centroids_time = float(match.group(1))

        pattern = get_time_str("kmeans-time")
        match = re.search(pattern, output)
        kmeans_time = float(match.group(1))

        pattern = get_time_str("fused-dist-argmin-time")
        match = re.search(pattern, output)
        fused_time = float(match.group(1))

        pattern = get_time_str("pwdist-time")
        match = re.search(pattern, output)
        pwdist_time = float(match.group(1))

        # Have to use findall otherwise we match with fused-dist-argmin-time :/
        pattern = get_time_str("argmin-time")
        match = re.findall(pattern, output)
        argmin_time = float(match[1])

        pattern = r"kmeans-score: (\d+\.\d+)"
        match = re.search(pattern, output)
        score = float(match.group(1))

        return {"runtime":kmeans_time,
                "dist_runtime":pwdist_time,
                "argmin_runtime":argmin_time,
                "centroids_runtime":centroids_time,
                "fused_runtime":fused_time,
                "d":d, "k":k, "n":n,
                "score":score,
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
        iter_var = np.arange(500, args.k+2, 500)
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

        cmd += f"./raft-bench/build/kmeans {n} {d} {k}"
        print(f"Executing {cmd}")

        try:
            trial_manager.run_trial(cmd, args.ntrials) #TODO: need to actually average this
            print(trial_manager.df)
        except Exception as err:
            print(err)

    trial_manager.save(f"{args.fname}{suffix}")



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
        iter_var = np.arange(2, args.k+1, 100)
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
        if args.infile==None:
            points = np.random.rand(n, d)
        else:
            points = pd.read_csv(args.infile)
            n = points.shape[0]
            d = points.shape[1]
        d_points = cudf.DataFrame(points)

        memcpy_time = 0 
        compute_time = 0
        
        # Init Kmeans
        kmeans = KMeans(n_clusters=k, max_iter=args.maxiters, verbose=6, init="random")

        # Warm up
        kmeans.fit(d_points)

        # Run Kmeans
        for i in range(args.ntrials):

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
        
        compute_time = compute_time / (args.ntrials - 1)
        memcpy_time = memcpy_time / (args.ntrials - 1)
        score = kmeans.score(d_points)

        print(f"Compute Time: {compute_time}s")
        print(f"Memcpy Time: {memcpy_time}s")
        print(f"Score: {score}")

        trial_manager.add_sample({"memcpy_runtime":memcpy_time,
                                  "runtime": compute_time,
                                  "score": score,
                                  "d":d, "n":n, "k":k,
                                  "name":None}
                                 )
        print(trial_manager.df)

    trial_manager.save(f"./{args.fname}{suffix}")


def run_cuml_kmeans_infile(args):
    from cuml.cluster import KMeans

    import cudf

    trial_manager = CuMLTrial()

    points = pd.read_csv(args.infile)

    n = points.shape[0]
    d = points.shape[1]
    k_vec = np.arange(100, args.k, 100) 


    print(f"Running cuml on {args.infile}, n={n} d={d}")

    d_points = cudf.DataFrame(points)

    memcpy_time = 0 
    compute_time = 0
    
    # Init Kmeans
    for k in k_vec:
        kmeans = KMeans(n_clusters=k, max_iter=args.maxiters, verbose=6, init="random")

        # Warm up
        kmeans.fit(d_points)

        # Run Kmeans
        for i in range(args.ntrials):

            stime = time.time()
            d_points = cudf.DataFrame(points)
            etime = time.time()

            if (i > 0):
                memcpy_time += (etime - stime)

            stime = time.time()
            kmeans.fit(d_points)
            etime = time.time()

            if (i > 0):
                compute_time += (etime - stime)

        compute_time /= (args.ntrials - 1)
        memcpy_time /= (args.ntrials - 1)
        score = kmeans.score(d_points)
        
        print(f"Compute Time: {compute_time}s")
        print(f"Memcpy Time: {memcpy_time}s")
        print(f"Score: {score}")

        trial_manager.add_sample({"memcpy_runtime":memcpy_time,
                                  "runtime": compute_time,
                                  "score": score,
                                  "d":d, "n":n, "k":k,
                                  "name":args.infile}
                                 )
        print(trial_manager.df)

    trial_manager.save(f"./{args.infile}-cuml")



def run_our_kmeans(args):

    cmd = f"../build/src/bin/{args.bin}  -m {args.maxiters}  -s 1 --runs {args.ntrials}  -t 0.0001 " 
    if args.infile:
        cmd += f"{args.infile} "

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
        iter_var = np.arange(500, args.k+2, 500)
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

        trial_manager.run_trial(cmd_curr, args.ntrials)
        print(trial_manager.df)


    trial_manager.save(f"./{args.fname}{suffix}")


def run_our_kmeans_infile(args):

    cmd = f"../build/src/bin/{args.bin}  -m {args.maxiters}  -s 1 --runs {args.ntrials}  -t 0.0001 -i {args.infile} " 

    df = pd.read_csv(args.infile)
    n = df.shape[0]
    d = df.shape[1]
    k_vec = np.arange(100, args.k, 100)

    cmd += f"-n {n} -d {d} "

    # Add srun if using SLURM
    if args.slurm:
        cmd = "srun -G 1 -n 1 " + cmd

    trial_manager = KmeansTrial()
    
    for k in k_vec:

        cmd_curr = cmd + f"-k {k} "

        print(f"Executing {cmd_curr}..")

        trial_manager.run_trial(cmd_curr, args.ntrials)
        print(trial_manager.df)


    trial_manager.save(f"./{args.infile}-{args.fname}")



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
    
    plt.title(f"Runtime of 10 Iterations of K-means {title_suffix}")
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
    parser.add_argument("--dmax", type=int)
    parser.add_argument("--fname", type=str)
    parser.add_argument("--slurm", action='store_true')
    parser.add_argument("--action", type=str)
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--platform", type=str)
    parser.add_argument("--itervar", type=str)
    parser.add_argument("--bin", type=str)
    parser.add_argument("--ntrials", type=int, default=10)
    parser.add_argument("--maxiters", type=int, default=10)
    parser.add_argument("--infile", type=str)
    
    args = parser.parse_args()


    if args.action=="plot-runtime":
        plot_breakdown(args)
        plot_runtime(args)
    elif args.action=="plot-mem":
        plot_mem(args)
    elif args.kernel!=None:
        plot_kernel_runtime(args)
    elif args.action=="cuml-kmeans":
        run_cuml_kmeans(args)
    elif args.action=="sklearn-kmeans":
        run_sklearn_kmeans(args)
    elif args.action=="torch-kmeans":
        run_torch_kmeans(args)
    elif args.action=="our-kmeans":
        run_our_kmeans(args)
    elif args.action=="raft-kmeans":
        run_raft_kmeans(args)
    elif args.action=="our-kmeans-infile":
        run_our_kmeans_infile(args)
    elif args.action=="cuml-kmeans-infile":
        run_cuml_kmeans_infile(args)



