#!/usr/bin/env bash
#$ -wd /home/mliu121/master_project
#$ -V
#$ -N dump_features
#$ -j y -o log/$JOB_NAME-$JOB_ID.out
#$ -M mliu121@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu
# or, less safely:
#export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
# Activate dev environments and call programs
conda activate fairseq

tsv_dir=$1
split=$2
ckpt_path=$3
layer=$4
nshard=$5
rank=$6
feat_dir=$7

layer=9
feature_name=hubert_base-l${layer}
feat_dir=features/${feature_name}/librispeech/train-${split}

python simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}


