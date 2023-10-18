#!/usr/bin/env bash
#$ -wd /home/mliu121/master_project
#$ -V
#$ -N dump_label
#$ -j y -o log/$JOB_NAME-$JOB_ID.out
#$ -M mliu121@jhu.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu
# or, less safely:
#export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
# Activate dev environments and call programs
conda activate fairseq

feat_dir=$1
split=$2
km_path=$3
nshard=$4
rank=$5
lab_dir=$6
python simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}


