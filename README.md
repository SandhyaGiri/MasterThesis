# Analysing the Robustness of Dirichlet Prior Networks (DPNs)

## Dirichlet Prior Networks

Paper: https://arxiv.org/abs/1802.10501  
Github repo of the paper: https://github.com/KaosEngineer/PriorNetworks-OLD

## Local dev environment setup
1. Create a virtualenv for python and install dependencies
    ```
    $ virtualenv thesis -p python3.7 
    $ source ./thesis/bin/activate
    $ pip install -r ./requirements.txt
    ```
2. Setup the DPN model

    Sample command from the project directory:
    ```
    python -m robust_priornet.setup_priornet --model_arch vgg6 --num_classes 10 --input_size 28 --drop_prob 0.05 --num_channels 1 /Users/sandhyagiri/git/masterthesis/runtime-model-conv
    ```
    For more options see: [here](robust_priornet/setup_priornet.py) for command line arguments.

3. Train the DPN model (normally)

    Sample command from the project directory:
    ```
    python -m robust_priornet.train_priornet --gpu 0 --model_dir ./runtime-model-vgg16-40epochs-svhn --num_epochs 40 --batch_size 128 --lr 0.0007500000000000001 --weight_decay 0.0 --use_cyclic_lr --add_ce_loss --grad_clip_value 10.0 --train_stepwise --val_every_steps 100 --ce_weight 1.0 --gamma 1.0 --min_train_epochs 15000 --patience 20 --cyclic_lr_pct_start 0.375 --optimizer SGD --target_precision 1000 ./runtime-data SVHN CIFAR10
    ```
    For more training options see: [here](robust_priornet/train_priornet.py).

4. Attack the DPN model

    Sample command from the project directory:
    ```
    python -m robust_priornet.attack_priornet --gpu 0 --batch_size 64 --epsilon 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --attack_type misclassify --attack_strategy PGD --attack_criteria confidence --target_precision 100 --norm 2 --model_dir ./runtime-model-vgg6-50epochs-mnist-adgen-robust-conf-ccat2-lr0.0001-precision100-g1.0-ce0.0 --max_steps 100 --threshold 0.0 --ood_dataset OMNIGLOT ./runtime-data MNIST ./runtime-model-vgg6-50epochs-mnist-adgen-robust-conf-ccat2-lr0.0001-precision100-g1.0-ce0.0/attack-PGD-confidence-misclassify-OMNIGLOT
    ```
    For more attack options see: [here](robust_priornet/attack_priornet.py)

5. Train the DPN model (adversarially)
    
    Sample command from the project directory:
    ```
    python -m robust_priornet.train_priornet --gpu 2 --model_dir ./runtime-model-vgg6-50epochs-mnist-adgen-robust-conf-ccat2-lr0.0001-precision100-g1.0-ce0.0 --num_epochs 50 --batch_size 64 --lr 0.0001 --weight_decay 0.0 --target_precision 100 --include_adv_samples    --train_stepwise --val_every_steps 100 --ce_weight 0.0 --optimizer ADAM --ccat  --gaussian_noise_std_dev 0.05 --min_train_epochs 15000 --patience 15 --grad_clip_value 10.0 --known_threshold_value 0.0 --gamma 1.0 --adv_training_type normal --adv_attack_type PGD --adv_attack_criteria confidence --adv_epsilon 0.3 --pgd_norm 2 --pgd_max_steps 100 --cyclic_lr_pct_start 0.35 ./runtime-data MNIST OMNIGLOT
    ```
    For more adversarial training options see: [here](robust_priornet/train_priornet.py).

## Submitting jobs in cluster

1. Jobs related to model training (normal, adversarial): [seml_config](train_config_seml.yaml)
2. Jobs related to attacks, evaluation, certification: [seml_config](eval_config_seml.yaml)