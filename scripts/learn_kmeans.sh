#!/usr/bin/env bash
#$ -wd /home/mliu121/master_project
#$ -V
#$ -N learn_km
#$ -j y -o log/$JOB_NAME-$JOB_ID.out
#$ -M mliu121@jhu.edu
#$ -m e
#$ -l ram_free=100G,mem_free=100G, hostname=b1[123456789]|c0*|c1[123456789]

# Activate dev environments and call programs
conda activate fairseq

feat_dir=$1
split=$2
nshard=$3
km_path=$4
n_cluster=$5

python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1


