## Steps to be followed to run commands (when not using seml)

### Model directory
Target home dir for the github repository: '/nfs/homedirs/giri/masterthesis'

This directory contains the runtime-data/* directory where all datasets used will be downloaded and stored.

### Model training commands

#### Adversarial training - dentropy - 0.05 (MNIST + OMNIGLOT)

python -m robust_priornet.setup_priornet --model_arch vgg6 --num_classes 10 --input_size 28 --drop_prob 0.05 --num_channels 1 ./runtime-model-adv-dentropy-05

python -m robust_priornet.train_priornet --gpu {gpu_number} --model_dir ./runtime-model-adv-dentropy-05  --num_epochs 50 --batch_size 64 --lr 0.0001 --weight_decay 0.0 --target_precision 100 --include_adv_samples  --train_stepwise --val_every_steps 100 --min_train_epochs 20000 --patience 100   --adv_training_type ood-detect --adv_attack_type PGD --adv_attack_criteria diff_entropy --adv_epsilon 0.05 --pgd_norm inf --pgd_max_steps 10 ./runtime-data MNIST OMNIGLOT

#### Adversarial training - duncertainty - 0.05 (MNIST + OMNIGLOT)
python -m robust_priornet.setup_priornet --model_arch vgg6 --num_classes 10 --input_size 28 --drop_prob 0.05 --num_channels 1 ./runtime-model-adv-duncertainty-05

python -m robust_priornet.train_priornet --gpu {gpu_number} --model_dir ./runtime-model-adv-duncertainty-05  --num_epochs 50 --batch_size 64 --lr 0.0001 --weight_decay 0.0 --target_precision 100 --include_adv_samples  --train_stepwise --val_every_steps 100 --min_train_epochs 20000 --patience 100   --adv_training_type ood-detect --adv_attack_type PGD --adv_attack_criteria mutual_info --adv_epsilon 0.05 --pgd_norm inf --pgd_max_steps 10 ./runtime-data MNIST OMNIGLOT

#### Adversarial training - dentropy - 0.1 (MNIST + OMNIGLOT)

python -m robust_priornet.setup_priornet --model_arch vgg6 --num_classes 10 --input_size 28 --drop_prob 0.05 --num_channels 1 ./runtime-model-adv-dentropy-1

python -m robust_priornet.train_priornet --gpu {gpu_number} --model_dir ./runtime-model-adv-dentropy-1  --num_epochs 50 --batch_size 64 --lr 0.0001 --weight_decay 0.0 --target_precision 100 --include_adv_samples  --train_stepwise --val_every_steps 100 --min_train_epochs 20000 --patience 100   --adv_training_type ood-detect --adv_attack_type PGD --adv_attack_criteria diff_entropy --adv_epsilon 0.1 --pgd_norm inf --pgd_max_steps 10 ./runtime-data MNIST OMNIGLOT

#### Adversarial training - duncertainty - 0.05 (MNIST + OMNIGLOT)
python -m robust_priornet.setup_priornet --model_arch vgg6 --num_classes 10 --input_size 28 --drop_prob 0.05 --num_channels 1 ./runtime-model-adv-duncertainty-1

python -m robust_priornet.train_priornet --gpu {gpu_number} --model_dir ./runtime-model-adv-duncertainty-1  --num_epochs 50 --batch_size 64 --lr 0.0001 --weight_decay 0.0 --target_precision 100 --include_adv_samples  --train_stepwise --val_every_steps 100 --min_train_epochs 20000 --patience 100   --adv_training_type ood-detect --adv_attack_type PGD --adv_attack_criteria mutual_info --adv_epsilon 0.1 --pgd_norm inf --pgd_max_steps 10 ./runtime-data MNIST OMNIGLOT

### Attack commands

#### Mutual Info PGD attacks (OMNIGLOT)
python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/normal --max_steps 10 --threshold 0.3525 --ood_dataset OMNIGLOT ./runtime-data MNIST ./MNIST_OMNIGLOT_models/normal/attack-pgd-mutual_info-ood-detect-omniglot

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/adv_conf --max_steps 10 --threshold 0.3713 --ood_dataset OMNIGLOT ./runtime-data MNIST ./MNIST_OMNIGLOT_models/adv_conf/attack-pgd-mutual_info-ood-detect-omniglot

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/adv_dentropy --max_steps 10 --threshold 0.3838 --ood_dataset OMNIGLOT ./runtime-data MNIST ./MNIST_OMNIGLOT_models/adv_dentropy/attack-pgd-mutual_info-ood-detect-omniglot

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/adv_duncertainty --max_steps 10 --threshold 0.3657 --ood_dataset OMNIGLOT ./runtime-data MNIST ./MNIST_OMNIGLOT_models/adv_duncertainty/attack-pgd-mutual_info-ood-detect-omniglot

#### Mutual Info PGD atatcks (CIFAR10)
python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/normal --max_steps 10 --threshold 0.3525 --ood_dataset CIFAR10 ./runtime-data MNIST ./MNIST_OMNIGLOT_models/normal/attack-pgd-mutual_info-ood-detect-cifar10

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/adv_conf --max_steps 10 --threshold 0.3713 --ood_dataset CIFAR10 ./runtime-data MNIST ./MNIST_OMNIGLOT_models/adv_conf/attack-pgd-mutual_info-ood-detect-cifar10

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/adv_dentropy --max_steps 10 --threshold 0.3838 --ood_dataset CIFAR10 ./runtime-data MNIST ./MNIST_OMNIGLOT_models/adv_dentropy/attack-pgd-mutual_info-ood-detect-cifar10

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 64 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/adv_duncertainty --max_steps 10 --threshold 0.3657 --ood_dataset CIFAR10 ./runtime-data MNIST ./MNIST_OMNIGLOT_models/adv_duncertainty/attack-pgd-mutual_info-ood-detect-cifar10


#### OOD-detect attacks on RPN-10 model

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 32 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria diff_entropy --norm inf --model_dir ./MNIST_OMNIGLOT_models/rpn --max_steps 10 --threshold -12.864 --ood_dataset OMNIGLOT ./runtime-data MNIST ./MNIST_OMNIGLOT_models/rpn/attack-pgd-diff_entropy-ood-detect-omniglot

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 32 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria diff_entropy --norm inf --model_dir ./MNIST_OMNIGLOT_models/rpn --max_steps 10 --threshold -12.864 --ood_dataset CIFAR10 ./runtime-data MNIST ./MNIST_OMNIGLOT_models/rpn/attack-pgd-diff_entropy-ood-detect-cifar10

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 32 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/rpn --max_steps 10 --threshold 0.3711 --ood_dataset OMNIGLOT ./runtime-data MNIST ./MNIST_OMNIGLOT_models/rpn/attack-pgd-mutual_info-ood-detect-omniglot

python -m robust_priornet.attack_priornet --gpu {gpu_number} --batch_size 32 --epsilon 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type ood-detect --attack_strategy PGD --attack_criteria mutual_info --norm inf --model_dir ./MNIST_OMNIGLOT_models/rpn --max_steps 10 --threshold 0.3711 --ood_dataset CIFAR10 ./runtime-data MNIST ./MNIST_OMNIGLOT_models/rpn/attack-pgd-mutual_info-ood-detect-cifar10