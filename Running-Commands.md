## Steps to be followed to run commands (when not using seml)

### Model directory
Target home dir for the github repository: '/nfs/homedirs/giri/masterthesis'

This directory contains the runtime-data/* directory where all datasets used will be downloaded and stored.

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
