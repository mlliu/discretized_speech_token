#!/usr/bin/env bash

nshard=10
ckpt_path=save/pretrained/hubert_base_ls960.pt
split=train
tsv_dir=manifest/librispeech/train-960
layer=9
feature_name=hubert_base-l${layer}
feat_dir=features/${feature_name}/librispeech/train-${split}
n_cluster=500
km_path=labels/${feature_name}/c${n_cluster}/km.pkl # /home/mliu121/master_project/labels/hubert_base-l9-k1s1-fp16-ls0.1/c500
lab_dir=labels/${feature_name}/c${n_cluster}
<< features
# 1. dumpy feature, need GPU
for rank in $(seq 0 $((nshard - 1))); do
    echo "dumping $split features"
    # if the feature file exists, skip
    if [ -f ${feat_dir}/${split}_${rank}_${nshard}.npy ] ; then
        echo "${feat_dir}/${split}_${rank}_${nshard}.npy exists"
    else
        qsub scripts/dump_features.sh ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
    fi
done
wait
features
<< kmeans
# 2. train k-means, cpu is enough
qsub scripts/learn_kmeans.sh ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster}
kmeans
# 3. dump pseudo labels, need GPU
km_path=/home/mliu121/master_project/labels/hubert_base-l9-k1s1-fp16-ls0.1/c500/km.pkl
for rank in $(seq 3 $((7 - 1))); do
    echo "dumping $split labels"
    if [ -f ${lab_dir}/${split}_${rank}_${nshard}.km ] ; then
        echo "${lab_dir}/${split}_${rank}_${nshard}.km exists"
    else
        qsub scripts/dump_labels.sh ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
    fi
done
# make sure that the length of the label file is the same as the feature file
for rank in $(seq 0 $((5 - 1))); do
    echo "check the length of ${lab_dir}/${split}_${rank}_${nshard}.km"
    if [ -f ${lab_dir}/${split}_${rank}_${nshard}.km ] ; then
        feat_len=$(wc -l ${feat_dir}/${split}_${rank}_${nshard}.len | awk '{print $1}')
        lab_len=$(wc -l ${lab_dir}/${split}_${rank}_${nshard}.km | awk '{print $1}')
        if [ $feat_len -ne $lab_len ]; then
            echo "length of ${lab_dir}/${split}_${rank}_${nshard}.km is not equal to ${feat_dir}/${split}_${rank}_${nshard}.npy"
            # echo "remove ${lab_dir}/${split}_${rank}_${nshard}.km"
            # rm ${lab_dir}/${split}_${rank}_${nshard}.km
        fi
    fi
done
# 4 merge
#for rank in $(seq 0 $((nshard - 1))); do
#  cat $lab_dir/${split}_${rank}_${nshard}.km
#done > $lab_dir/${split}.km

# 5 Create a dummy dict
#for x in $(seq 0 $((n_clusters - 1))); do
#  echo "$x 1"
#done >> $lab_dir/dict.km.txt



