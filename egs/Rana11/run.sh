#!/bin/bash

set -x
#source ../../venvast/bin/activate    virtual environment created in conda
export TORCH_HOME=../../pretrained_models

model=ast
dataset=Rana_Draytonii_V5 # replace with your dataset name
set=balanced # replace with your training set type (full/balanced)
imagenetpretrain=True

# Please adjust all these parameters based on your data and task
lr=1e-5
epoch=8
tr_data=./data/train_data_clean.json # replace with your training data path
te_data=./data/val_data.json # replace with your evaluation data path
freqm=48
timem=200
mixup=0.5
fstride=10
tstride=10
batch_size=12
dataset_mean=-4.27 # same as Audioset
dataset_std=4.57 # same as Audioset
audio_length=1000 # 10 seconds
noise=False
metrics=acc
loss=CE
warmup=True
wa=True
exp_dir=./exp/test-${set}-clean-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-decoupe
clean_checkpoint_path=$exp_dir/models/best_audio_model.pth  # Path for saving the model after Phase 1

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir


python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/labels.csv --n_class 3 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} \
--wa ${wa}

# Save the model after Phase 1
cp $exp_dir/models/best_audio_model.pth $clean_checkpoint_path

# Phase 2: Fine-tune on Noisy Data
tr_data=./data/train_data_noisy.json
exp_dir=./exp/test-${set}-noisy-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-decoupe

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

# Fine-tune on noisy data
python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/labels.csv --n_class 3 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} \
--tstride $tstride --fstride $fstride --imagenet_pretrain ${imagenetpretrain} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} \
--wa ${wa} --load_checkpoint ${clean_checkpoint_path}
