#!/bin/bash

set -x
#source ../../venvast/bin/activate    virtual environment created in conda
export TORCH_HOME=../../pretrained_models

model=ast
dataset=Rana_Draytonii # replace with your dataset name
set=balanced # replace with your training set type (full/balanced)
imagenetpretrain=True
audioset_pretrain=True

# Please adjust all these parameters based on your data and task
lr=1e-5
epoch=10
tr_data=/content/drive/MyDrive/Rana_Draytonii/Rana7/train_data.json # replace with your training data path
te_data=/content/drive/MyDrive/Rana_Draytonii/Rana7/val_data.json # replace with your evaluation data path
freqm=48
timem=200
mixup=0.5
fstride=10
tstride=10
batch_size=4
dataset_mean=-4.27 # same as Audioset
dataset_std=4.57 # same as Audioset
audio_length=1000 # 10 seconds
noise=False
metrics=acc
loss=BCE
warmup=True
wa=True
exp_dir=./exp/test-${set}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-decoupe

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir


python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /content/drive/MyDrive/Rana_Draytonii/Rana7/labels.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--audioset_pretrain $audioset_pretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} \
--wa ${wa}
