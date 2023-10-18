#!/usr/bin/env bash
#$ -wd /home/mliu121/master_project
#$ -V
#$ -N extract_hubert_features
#$ -j y -o log/$JOB_NAME-$JOB_ID.out
#$ -M mliu121@jhu.edu
#$ -m e
#$ -l ram_free=100G,mem_free=10G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
#source /home/gqin2/scripts/acquire-gpu
# or, less safely:
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)


# Activate dev environments and call programs
conda activate fairseq

nshard=1

kmean_portion=0.1 # using 10% of the training data to train k-means
# ckpt_path=/persist/git/fairseq-hubert/save/pretrained/hubert_base_ls960.pt
ckpt_path=save/pretrained/hubert_base_ls960.pt
layer=9
pool_k=1 # when set to 1, it means no pooling
pool_s=1
num_clusters=500
echo "ckpt_path: $ckpt_path"
echo "layer: $layer"
echo "num_clusters: $num_clusters"
split=train-${kmean_portion} # train-0.1 split for feature extraction
# pretrain_manifest=manifest/librispeech/train-960/${split}.tsv
pretrain_manifest=manifest/librispeech/train-960/${split}.tsv # train-0.1

feature_name=hubert_base-l${layer}-k${pool_k}s${pool_s}-fp16-ls${kmean_portion}

feature_dir=features/${feature_name}/librispeech/train-${kmean_portion}

<< feature_dump
mkdir -p $feature_dir 


for rank in $(seq 0 $((nshard - 1))); do
    if [ -f ${feature_dir}/${split}_${rank}_${nshard}.npy ] ; then
        echo "${feature_dir}/${split}_${rank}_${nshard}.npy exists"
    else
        python tools/generate_pseudo_language.py dump_hubert_features \
                --ckpt_path $ckpt_path \
                --manifest $pretrain_manifest  --rank $rank --nshard $nshard --feat_dir $feature_dir \
                --layer $layer \
                --pool_k $pool_k --pool_s $pool_s --fp16 True &
    fi
done
wait
feature_dump


label_dir=labels/${feature_name}/c${num_clusters}
cluster_model_path=${label_dir}/km.pkl

mkdir -p $label_dir

echo $cluster_model_path
if [ -f $cluster_model_path ]; then
    echo "using existing k-means $cluster_model_path"
else
    echo "training k-means model with $num_clusters clusters"
    python tools/learn_kmeans.py $feature_dir train-${kmean_portion} $nshard $cluster_model_path $num_clusters --percent 1
fi



#for split in valid train; do
for split in train; do
    echo "dumping $split labels"
    [ -s ${label_dir}/${split}.km ] || rm -f ${label_dir}/${split}.km
    if ! [ -f ${label_dir}/${split}.km ]; then
        for rank in $(seq 0 $((nshard - 1))); do
            python tools/generate_pseudo_language.py dump_hubert_clusters \
                    --ckpt_path $ckpt_path \
                    --km_path $cluster_model_path \
                    --manifest manifest/librispeech/train-960/${split}.tsv \
                    --rank $rank --nshard $nshard --lab_dir $label_dir \
                    --layer $layer \
                    --pool_k $pool_k --pool_s $pool_s --fp16 True &

        done
        wait

        echo "merging $split shards"
        for rank in $(seq 0 $((nshard - 1))); do
        cat $label_dir/${split}_${rank}_${nshard}.km
        done > $label_dir/${split}.km
    fi
done

echo "create dict"
python tools/create_dict.py ${label_dir}/train.km ${label_dir}/dict.km.txt


echo "done"

