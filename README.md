# Decoupled Cross-Modal BERT

## README
For a better user experience, we recommend you to directly use github repository
github [link](https://github.com/cvpr2021dcb/cvpr2021dcb)

## Prepare dataset and models

### Dataset
Flickr30K dataset

download the testing features through this [link](https://www.dropbox.com/s/bkgzftnavcub1hs/flickr30k_test_frcnnnew.tar.gz?dl=0) 

download the training features through this [link](https://www.dropbox.com/s/bkgzftnavcub1hs/flickr30k_test_frcnnnew.tar.gz?dl=0)

unzip your downloaded files, and move them to data/f30k_precomp

### BERT Model
download the pretrained bert model provided by HuggingFace through this [link](https://www.dropbox.com/s/a20ufjz3145g80z/pytorch_model.bin?dl=0)

move your downloaded pytorch_model.bin file to ./bert fold

## Run Script
CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 256 --num_epochs=70 --lr_update=30 --learning_rate=.00006


