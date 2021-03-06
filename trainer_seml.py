import logging
import os

import numpy as np

from sacred import Experiment
from seml import database_utils as db_utils
from seml import misc

ex = Experiment()
misc.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(in_domain_dataset, ood_dataset, input_image_size, num_classes, model_arch, rpn_wrapper, rpn_mc_samples, rpn_reduction,
        rpn_sigma, fc_layers, num_epochs, train_stepwise, val_every_steps, min_train_epochs, patience, num_channels, learning_rate,
        use_cyclic_lr, add_ce_loss, cyclic_lr_pct_start, ce_weight, reverse_KL, drop_rate, use_fixed_threshold, known_threshold_value,
        grad_clip_value, weight_decay, target_precision, model_dir, resume_from_ckpt, augment_data, data_dir, lr_decay_milestones,
        batch_size, dataset_size_limit, adv_training, only_out_in_adv, ccat, gaussian_noise_normal, gaussian_noise_std_dev,
        adv_training_type, adv_attack_type, gamma, adv_epsilon, adv_attack_criteria, adv_model_dir, adv_persist_images, optimizer,
        pgd_norm, pgd_max_steps, logdir):

    logging.info('Received the following configuration:')
    logging.info(f'In domain dataset: {in_domain_dataset}, OOD dataset: {ood_dataset}')

    if os.environ.get('SLURM_JOB_GPUS', None) is not None:
        gpu_list = list(map(int, os.environ['SLURM_JOB_GPUS'].split(",")))
        gpu_list = " ".join(map(lambda gpu: "--gpu " + str(gpu), gpu_list))
    else:
        gpu_list = "--gpu -1"

    # needed to be executed before model setup (for torch >=1.5)
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    # modify model_dir with grid search params
    model_dir = f'{model_dir}-lr{learning_rate}-precision{target_precision}-g{gamma}-ce{ce_weight}'
    # set up the model
    fc_layers_list = " ".join(map(lambda x: str(x), fc_layers))
    if rpn_wrapper == 'count':
        rpn = '--rpn_count'
    elif rpn_wrapper == 'simple':
        rpn = '--rpn_simple'
    elif rpn_wrapper == 'normal':
        rpn = '--rpn'
    else:
        rpn = ''
    setup_cmd = f'python -m robust_priornet.setup_priornet --model_arch {model_arch} {rpn} \
        --fc_layers {fc_layers_list} --num_classes {num_classes} --input_size {input_image_size} \
        --rpn_sigma {rpn_sigma} --rpn_num_samples {rpn_mc_samples} --rpn_reduction_method {rpn_reduction} \
        --drop_prob {drop_rate} --num_channels {num_channels} {model_dir}'
    logging.info(f"Setup command being executed: {setup_cmd}")
    os.system(setup_cmd)

    # training the model
    # lr_decay_milestones = " ".join(map(lambda epoch: "--lrc " + str(epoch), lr_decay_milestones))
    resume = '--resume_from_ckpt' if resume_from_ckpt else ''
    augment = '--augment' if augment_data else ''
    use_cyclic_lr = '--use_cyclic_lr' if use_cyclic_lr else ''
    add_ce_loss = '--add_ce_loss' if add_ce_loss else ''
    dataset_limit = f'--dataset_size_limit {dataset_size_limit}' if dataset_size_limit is not None else ''
    use_fixed_threshold = f'--use_fixed_threshold' if use_fixed_threshold else ''
    reverse_kl = '--reverse_KL' if reverse_KL else ''
    train_stepwise = '--train_stepwise' if train_stepwise else ''
    if adv_training:
        adv_model = f'--adv_model_dir {adv_model_dir}' if adv_model_dir != "" else ''
        adv_persist = '--adv_persist_images' if adv_persist_images else ''
        out_in_adv = '--include_only_out_in_adv_samples' if only_out_in_adv else ''
        ccat = '--ccat' if ccat else ''
        gaussian_noise_normal = '--gaussian_noise_normal' if gaussian_noise_normal else ''
        train_cmd = f'python -m robust_priornet.train_priornet {gpu_list} --model_dir {model_dir} \
            --num_epochs {num_epochs} --batch_size {batch_size} --lr {learning_rate} --weight_decay {weight_decay} \
            --target_precision {target_precision} --include_adv_samples {augment} {dataset_limit} \
            {train_stepwise} --val_every_steps {val_every_steps} --ce_weight {ce_weight} \
            --optimizer {optimizer} {ccat} {gaussian_noise_normal} --gaussian_noise_std_dev {gaussian_noise_std_dev} \
            --min_train_epochs {min_train_epochs} --patience {patience} --grad_clip_value {grad_clip_value} \
            {use_fixed_threshold} --known_threshold_value {known_threshold_value} --gamma {gamma} \
            {adv_model} {adv_persist} {resume} {out_in_adv} --adv_training_type {adv_training_type} \
            --adv_attack_type {adv_attack_type} --adv_attack_criteria {adv_attack_criteria} \
            --adv_epsilon {adv_epsilon} --pgd_norm {pgd_norm} --pgd_max_steps {pgd_max_steps} \
            {use_cyclic_lr} {add_ce_loss} {reverse_kl} --cyclic_lr_pct_start {cyclic_lr_pct_start} \
            {data_dir} {in_domain_dataset} {ood_dataset}'
    else:
        train_cmd = f'python -m robust_priornet.train_priornet {gpu_list} --model_dir {model_dir} {dataset_limit} \
            --num_epochs {num_epochs} --batch_size {batch_size} --lr {learning_rate} {resume} {augment} \
            --weight_decay {weight_decay} {use_cyclic_lr} {add_ce_loss} --grad_clip_value {grad_clip_value} \
            {train_stepwise} --val_every_steps {val_every_steps} --ce_weight {ce_weight} --gamma {gamma} \
            --min_train_epochs {min_train_epochs} --patience {patience} {reverse_kl} \
            --cyclic_lr_pct_start {cyclic_lr_pct_start} --optimizer {optimizer} \
            --target_precision {target_precision} {data_dir} {in_domain_dataset} {ood_dataset}'
    logging.info(f"Training command being executed: {train_cmd}")
    os.system(train_cmd)

    # recover the logs from model_dir/log/LOG.txt file and return
    #with open(f'{model_dir}/LOG.txt', 'r') as f:
    #    results = f.read()
    
    # the returned result will be written into the database
    return {}
