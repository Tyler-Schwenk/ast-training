
# AST: Audio Spectrogram Transformer - fine tuned for Rana Draytonii
 - [Introduction](#Introduction)
 - [How it works](#How-it-works)
 - [Limitations](#Limitations)
 - [Ideas for improvement](#Ideas-for-improvement)
 - [Citing](#Citing)  
 - [Getting Started](#Getting-Started)
 - [Contact](#Contact)

## Introduction  

<p align="center"><img src="https://github.com/YuanGongND/ast/blob/master/ast.png?raw=true" alt="Illustration of AST." width="300"/></p>

This repository contains Tyler Schwenk's fork of the official implementation (in PyTorch) of the **Audio Spectrogram Transformer (AST)** proposed in the Interspeech 2021 paper [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) (Yuan Gong, Yu-An Chung, James Glass).  

AST is the first **convolution-free, purely** attention-based model for audio classification which supports variable length input and can be applied to various tasks. 

I have fine tuned the audioset-pretrained model to identify the presence of the endangered Rana Draytonii in audio files taken from the field. These files are recorded near ponds, and can include noise such as other species of frogs or animals, rain, and wind. Despite the noise, the model is able to determine the presence of Rana Draytonii with about 98.77% accuracy, calculated as the proportion of predictions where the highest scoring label matches the key data. Beyond just determining if there are any calls heard in an audio file, my scipt will track when in the file they are heard, as well as pull information from the files metadata and output the information in an Excel file as below:

| Model Name : Version | File Name     | Prediction | Times Heard | Device ID               | Timestamp                  | Temperature | Review Date |
|----------------------|---------------|------------|-------------|-------------------------|----------------------------|-------------|-------------|
| AST_Rana_Draytonii:1.0 | 20221201_190000 | Negative   | N/A         | AudioMoth 249BC30461CBB1E6 | 19:00:00 01/12/2022 (UTC-8) | 9.3C        | 2023-07-22  |
| AST_Rana_Draytonii:1.0 | 20221201_205000 | Negative   | N/A         | AudioMoth 249BC30461CBB1E6 | 20:50:00 01/12/2022 (UTC-8) | 9.1C        | 2023-07-22  |


I have altered the dataloader.py, and created a new model in egs/Rana7. I have also added my preprossesing scripts (Data_Manager.ipynb) and training script (ASTtraining.ipynb) in the folder "Preprocessing". 

The folder "Rana_Draytonii_ML_Model" contains everything needed to run the model, besides my fine tuned weights which can be downloaded here: https://www.dropbox.com/scl/fi/1ohxy38sm9863u2quf14h/best_audio_model1.pth?rlkey=ku3y2z88agn2kyumypjz3vpzj&dl=0





## How it works:
I created a fork from the official AST github repo, and have modified some files to be able to fine tune their model to our task.

I have a data manager in google colab that takes the .wav files and first splits them into 10 second segments, then saves them back to either google drive/locally. 

Then it reduced the files frequency range from 0-25 kHz to 0-3 kHz, which is where rana draytonii calls fall. 

Next it resamples the audio files to be 16 kHz sample rate, which is ideal for most machine learning tasks and required for use with ast. It also converts any stereo files to be mono for uniformity (most are already mono).

The data manager also splits the files into 15% for testing, 15% validation, and 70% training before creating a labels.csv and three .json files to index the files in training. 

ASTtraining google colab will clone the repo, mount google drive and install dependencies, before running the training script.

## Limitations:
Lack of data: currently using about 300 positive and 300 negative samples that are 10 seconds each. ~10 minutes of data

Noise: Can have very high amounts of noise in the data. Primarily due to chorus frogs, but also things like running water or wind.

Unfocused data: Data comes in as .wav files with frequencies from 0 – 24 kHz, but our frog vocalizes between 0 - 3 kHz, and chorus frogs are around 0.6 - 5 kHz. So a lot of the data is useless, and there is often noise. More focused recordings could help?


## Ideas for improvement:
Gather more data

Look more into what data augmentation techniques are being used and make sure they aren’t harming my data.

Cut off the unused frequencies (above 3 kHz)

Use a totally different model specifically for binary classification and/or frog noises

Polish everything, wrap a good model in an easy to use step by step google colab file that has good documentation. Automate things so you can plug in a path to a folder, and have it return readable locations of where it hears Rana draytonii so human can double check. (original file name and timestamp within 10 seconds)

Perhaps plug this in with data of where/when the files were recorded and provide more information like graphs of calls over time or a map.



## Citing  
The first paper proposes the Audio Spectrogram Transformer while the second paper describes the training pipeline that they applied on AST to achieve the new state-of-the-art on AudioSet.   
```  
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```  
```  
@ARTICLE{gong_psla, 
    author={Gong, Yuan and Chung, Yu-An and Glass, James},  
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
    title={PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation},   
    year={2021}, 
    doi={10.1109/TASLP.2021.3120633}
}
```  
  
## Getting Started  

Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast/ 
python3 -m venv venvast
source venvast/bin/activate
pip install -r requirements.txt 
```
  
Step 2. Test the AST model.

```python
ASTModel(label_dim=527, \
         fstride=10, tstride=10, \
         input_fdim=128, input_tdim=1024, \
         imagenet_pretrain=True, audioset_pretrain=False, \
         model_size='base384')
```  

**Parameters:**\
`label_dim` : The number of classes (default:`527`).\
`fstride`:  The stride of patch spliting on the frequency dimension, for 16\*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6 (used in the paper). (default:`10`)\
`tstride`:  The stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6 (used in the paper). (default:`10`)\
`input_fdim`: The number of frequency bins of the input spectrogram. (default:`128`)\
`input_tdim`: The number of time frames of the input spectrogram. (default:`1024`, i.e., 10.24s)\
`imagenet_pretrain`: If `True`, use ImageNet pretrained model. (default: `True`, we recommend to set it as `True` for all tasks.)\
`audioset_pretrain`: If`True`,  use full AudioSet And ImageNet pretrained model. Currently only support `base384` model with `fstride=tstride=10`. (default: `False`, we recommend to set it as `True` for all tasks except AudioSet.)\
`model_size`: The model size of AST, should be in `[tiny224, small224, base224, base384]` (default: `base384`).

**Input:** Tensor in shape `[batch_size, temporal_frame_num, frequency_bin_num]`. Note: the input spectrogram should be normalized with dataset mean and std, see [here](https://github.com/YuanGongND/ast/blob/102f0477099f83e04f6f2b30a498464b78bbaf46/src/dataloader.py#L191). \
**Output:** Tensor of raw logits (i.e., without Sigmoid) in shape `[batch_size, label_dim]`.

``` 
cd ast/src
python
```  

```python
import os 
import torch
from models import ASTModel 
# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'  
# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins 
test_input = torch.rand([10, input_tdim, 128]) 
# create an AST model
ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True)
test_output = ast_mdl(test_input) 
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes. 
print(test_output.shape)  
```  

We have an one-click, self-contained Google Colab script for (pretrained) AST inference and attention visualization. Please test the model with your own audio at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/ast/blob/master/colab/AST_Inference_Demo.ipynb) by one click (no GPU needed).



 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.

