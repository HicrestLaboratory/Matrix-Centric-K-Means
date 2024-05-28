#!/usr/bin/bash


../build/src/bin/gpukmeans -n $1 -d $2 -k $3 -m 2 --runs 2 -t 1e-4
./raft-bench/template/build/kmeans $1 $2 $3
