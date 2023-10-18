## Prepare training data manifest

1. First, install the `soundfile` library:
```shell script
pip install soundfile
```
2. Please set `LIBRISPEECH_PATH` to your librispeech folder which contains three subfolders `train-clean-100`, `train-clean-360`, `train-other-500`.
```sh
export LIBRISPEECH_PATH=/export/corpora5/LibriSpeech
mkdir -p manifest/librispeech/train-960
python -m fairseq.examples.wav2vec.wav2vec_manifest $LIBRISPEECH_PATH  --dest manifest/librispeech/train-960 --ext flac --valid-percent 0 --path-must-contain train
```
$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a
separately pre-processed manifest file.

1. what is the format of the manifest file?

`*.tsv` files contains a list of audio, where each line is the root, and
following lines are the subpath for each audio:
```
<root-dir>
<audio-path-1> # of frame
<audio-path-2> # of frame
...
```
## HUBERT feature
To extract features from the `${layer}`-th transformer layer of a trained
HUBERT model saved at `${ckpt_path}`, run:
```sh
python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
```
Features would also be saved at `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`.

- if out-of-memory, decrease the chunk size with `--max_chunk`

## Train k-means model and get cluster indices.
Please make sure that you have download pre-trained hubert-base checkpoint at `HUBERT_PATH`.
Notably, this step requires a GPU for feature extraction and 64GB main memory for k-means training.
Extracting HuBERT features takes about 15 minutes, training k-means may take about an hour, dumping the cluster ids of the whole Librispeech 960h data takes more than two hours.
```sh
HUBERT_PATH="save/pretrained/hubert_base_ls960.pt"
mkdir -p save/pretrained
if ! [ -f $HUBERT_PATH ]; then
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -O $HUBERT_PATH
fi
bash scripts/extract-hubert-features.sh $HUBERT_PATH 9 500

qsub scripts/extract-hubert-features.sh
```
where 9, 2, 2, 500 means that we use the 9-th layer of HuBERT, kernel size 2 and stride size 2 for average pooling, and 500 custers in k-means.

2

## K-means clustering
To fit a k-means model with `${n_clusters}` clusters on 10% of the `${split}` data, run
```sh
python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1
```
This saves the k-means model to `${km_path}`.

- set `--precent -1` to use all data
- more kmeans options can be found with `-h` flag


## K-means application
To apply a trained k-means model `${km_path}` to obtain labels for `${split}`, run
```sh
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
```
This would extract labels for the `${rank}`-th shard out of `${nshard}` shards
and dump them to `${lab_dir}/${split}_${rank}_${shard}.km`


Finally, merge shards for `${split}` by running
```sh
for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
```


## Create a dummy dict
To create a dummy dictionary, run
```sh
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt
```

