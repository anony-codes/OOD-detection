#!/bin/bash

trap "kill 0" EXIT  # IMPORANT

echo "start"

for seed in 100
do
  for loss_type in 'final'
  do
    for aug_type in 'simclr'
    do
        for lambda in 0.1
        do
            for lambda_var in 0.1
            do
                for lambda_corr in 0.0001
                do
                CUDA_VISIBLE_DEVICES=2 python ./main.py --seed $seed --loss $loss_type --aug_type $aug_type --lambda_loss_mu $lambda --lambda_loss_var $lambda_var --lambda_loss_corr $lambda_corr;
                CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric baseline --aug_type $aug_type --lambda_loss_mu $lambda --lambda_loss_var $lambda_var --lambda_loss_corr $lambda_corr;
                CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric mahalanobis --aug_type $aug_type --lambda_loss_mu $lambda --lambda_loss_var $lambda_var --lambda_loss_corr $lambda_corr;
                CUDA_VISIBLE_DEVICES=2 python ./test.py --seed $seed --loss $loss_type --metric mahalanobis_ensemble --aug_type $aug_type --lambda_loss_mu $lambda --lambda_loss_var $lambda_var --lambda_loss_corr $lambda_corr;
                CUDA_VISIBLE_DEVICES=2 python ./get_features.py --seed $seed --loss $loss_type --aug_type $aug_type --lambda_loss_mu $lambda --lambda_loss_var $lambda_var --lambda_loss_corr $lambda_corr;
                done
          done
        done
    done
  done
done
echo "end"



