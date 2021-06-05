LF="basic" # cross entropy
CKD="checkpoint" # model checkpoint
VIS="runs" # tensorboard directory
NGPU=0
SAVEDIR="./save" # save_dir
ROOT="data" # data directory

S=0 # seed

# train
  CUDA_VISIBLE_DEVICES=$NGPU python main.py \
      --root $ROOT \
      --seed $S \
      --model transformer \
      --save_example_path sample \
      --encoder_class bert-base-uncased \
      --per_gpu_train_batch_size 32 \
      --per_gpu_eval_batch_size 32 \
      --gradient_accumulation_step 1 \
      --do_train \
      --evaluate_during_training \
      --dataset product \
      --mixed_precision \
      --time_series \
      --n_epoch 10 \
      --metric $LF \
      --metric $LF \
      --lr 1.0e-5 \
      --pool \
      --use_custom_vocab \
      --vis_dir runs \
      --save_weight \
      --checkpoint_dir $CKD \
      --scratch ;


# test
  CUDA_VISIBLE_DEVICES=$NGPU python main.py \
      --root $ROOT  \
      --seed $S \
      --save_example_path $SAVEDIR \
      --encoder_class bert-base-uncased \
      --per_gpu_train_batch_size 32 \
      --per_gpu_eval_batch_size 32 \
      --gradient_accumulation_step 1 \
      --model transformer \
      --do_test \
      --evaluate_during_training \
      --dataset product \
      --mixed_precision \
      --time_series \
      --n_epoch 5 \
      --metric $LF \
      --lr 5.0e-6 \
      --return_type prob \
      --scratch \
      --pool \
      --use_custom_vocab\
      --model_path  "" #please specify path
