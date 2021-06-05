


LF="regularizer"
CKD="checkpoint" # model checkpoint
VIS="runs" # tensorboard directory
NGPU=0
SAVEDIR="./save" # save_dir
ROOT="data" # data directory

SEED=0 # seed

M=0.1 #weight1
MV=10.0  #weight2
MC=1.0 #weight3

# train
CUDA_VISIBLE_DEVICES=$NGPU python main.py \
    --root $ROOT  \
    --model transformer \
    --seed $SEED \
    --save_example_path SAVEEXAMPLE \
    --encoder_class bert-base-uncased \
    --per_gpu_train_batch_size 128 \
    --per_gpu_eval_batch_size 128 \
    --gradient_accumulation_step 1 \
    --do_train \
    --evaluate_during_training \
    --mixed_precision \
    --dataset product \
    --time_series \
    --n_epoch 10 \
    --metric $LF \
    --lr 1.0e-5 \
    --lambda_var $MV \
    --lambda_mu $M \
    --lambda_corr $MC \
    --pool \
    --scratch \
    --use_custom_vocab\
    --vis_dir $VIS \
    --checkpoint_dir $CKD;

#test
CUDA_VISIBLE_DEVICES=$NGPU python main.py \
    --root $ROOT  \
    --seed $SEED \
    --save_example_path $SAVEDIR\
    --encoder_class bert-base-uncased \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_step 1 \
    --do_test \
    --return_type prob \
    --evaluate_during_training \
    --dataset product \
    --mixed_precision \
    --time_series \
    --n_epoch 5 \
    --lambda_var $MV \
    --lambda_mu $M \
    --lambda_corr $MC \
    --lr 1.0e-3 \
    --pool \
    --model transformer \
    --return_type prob \
    --scratch \
    --use_custom_vocab \
    --checkpoint_dir $CKD \
    --model_path  " " #please specify path
