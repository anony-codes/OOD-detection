#!/bin/bash

trap "kill 0" EXIT  # IMPORANT

echo "start"


for seed in 100
do
  for loss_type in 'ce'
  do
    for aug_type in 'simclr'
    do
     CUDA_VISIBLE_DEVICES=2 python ./main.py --seed $seed --loss $loss_type --aug_type $aug_type;
     CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric baseline --aug_type $aug_type;
     CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric odin --aug_type $aug_type;
     CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric mahalanobis --aug_type $aug_type;
     CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric mahalanobis_ensemble --aug_type $aug_type;
     CUDA_VISIBLE_DEVICES=2 python ./get_features.py --seed $seed --loss $loss_type --aug_type $aug_type;
    done
  done
done
echo "end"


echo "end"
