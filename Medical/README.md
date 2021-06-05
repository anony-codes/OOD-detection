## CE Model ##
To train CE model and test it on in-distribution data, use the following command:
!CUDA_VISIBLE_DEVICES=$device python main.py --run_mode 'train' --seed $inp_seed --result_path $path --data_path $pathdata

To get accuracy on OOD datasets, use the following command:
!CUDA_VISIBLE_DEVICES=$device python main.py --run_mode 'test' --seed $inp_seed --result_path $path --data_path $pathdata --variation 'bright'

To test OOD detection performance using MSP. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test.py --metric 'baseline' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata
              
To test OOD detection performance using Mahalanobis. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test.py --metric 'mahalanobis' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata

To test OOD detection performance using Mahalanobis ensemble. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test_mahala_ensemble.py --metric 'mahalanobis_ensemble' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata

To test OOD detection performance using ODIN. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test.py --metric 'odin' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata

## Proposed Model ##
To train model with our proposed loss function and test it on in-distribution dataset, use the following command:
!CUDA_VISIBLE_DEVICES=$device python main.py --run_mode 'train' --seed $inp_seed --result_path $path --data_path $pathdata --w1 1 --w2 0.1 --w3 0.1 --w4 0.1

To get accuracy on OOD datasets, use the following command:
!CUDA_VISIBLE_DEVICES=$device python main.py --run_mode 'test' --seed $inp_seed --result_path $path --data_path $pathdata --variation 'bright' --w1 1 --w2 0.1 --w3 0.1 --w4 0.1


To test OOD detection performance using MSP. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test.py --metric 'baseline' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata
              
To test OOD detection performance using Mahalanobis. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test.py --metric 'mahalanobis' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata

To test OOD detection performance using Mahalanobis ensemble. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test_mahala_ensemble.py --metric 'mahalanobis_ensemble' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata

To test OOD detection performance using ODIN. use the following command:
!CUDA_VISIBLE_DEVICES=$device python test.py --metric 'odin' --seed $inp_seed --flag_adjust --type_adjust 'bright'  --result_path $path --data_path  $pathdata