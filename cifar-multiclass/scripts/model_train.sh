#!/bin/bash

NGPU=0


mkdir data

# baseline
seed=0
#CUDA_VISIBLE_DEVICES=$NGPU python ./train.py --loss 'ce' --seed $seed; #train
#CUDA_VISIBLE_DEVICES=$NGPU python ./get_ood_score.py --loss 'ce' --seed $seed; #inference
#


# our
seed=0
lambda=0.01
lambda_var=0.1
lambda_corr=0.01

CUDA_VISIBLE_DEVICES=$NGPU python ./train.py --loss 'ours' --w1 $lambda --w2 $lambda_var --w3 $lambda_corr --seed $seed; #train
CUDA_VISIBLE_DEVICES=$NGPU python ./get_ood_score.py --loss 'ours' --w1 $lambda --w2 $lambda_var --w3 $lambda_corr --seed $seed; # inference