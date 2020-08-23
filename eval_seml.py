import logging
import os
from datetime import datetime

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
def run(in_domain_dataset, ood_dataset, model_dir, data_dir, batch_size, use_train_dataset,
        use_val_dataset, dataset_size_limit, target_precision, logdir,
        run_eval, rpn_mc_samples, rpn_reduction, run_attack, epsilon_list, threshold, attack_type,
        attack_strategy, attack_criteria, attack_only_out_dist, attack_norm, max_steps,
        run_certification, certify_task, certify_only_ood, n0, n, sigma, uncertainty_measure,
        uncertainty_measure_threshold):
    """
    Performs both in-domain evaluation, and ood evaluation and the corresponding results
    are stored under eval/ and ood-eval/ inside the model_dir.
    """
    logging.info('Received the following configuration:')
    logging.info(f'In domain dataset: {in_domain_dataset}, OOD dataset: {ood_dataset}')

    os.system('set | grep SLURM | while read line; do echo "# $line"; done')

    # Check that we are training on a sensible GPU
    if os.environ.get('SLURM_JOB_GPUS', None) is not None:
        gpu_list = list(map(int, os.environ['SLURM_JOB_GPUS'].split(",")))
        gpu_list = " ".join(map(lambda gpu: "--gpu " + str(gpu), gpu_list))
    else:
        gpu_list = "--gpu -1"

    if run_eval is True:
        # in-domain evaluation ( + misclassification detection eval, as binary classification task)
        out_dir = os.path.join(model_dir, "eval")
        cmd = f"python -m robust_priornet.eval_priornet {gpu_list} --batch_size {batch_size} \
            --model_dir {model_dir} --task misclassification_detect --result_dir {out_dir} \
            {'--train_dataset' if use_train_dataset else ''} \
            {'--val_dataset' if use_val_dataset else ''} --target_precision {target_precision} \
            --rpn_num_samples {rpn_mc_samples} --rpn_reduction_method {rpn_reduction} \
            {data_dir} {in_domain_dataset} {ood_dataset}"
        logging.info(f"In domain EVAL command being executed: {cmd}")
        os.system(cmd)

        # ood detection evaluation (as a binary classification task)
        out_dir = os.path.join(model_dir, "ood-eval")
        cmd = f"python -m robust_priornet.eval_priornet {gpu_list} --batch_size {batch_size} \
                --model_dir {model_dir} --task ood_detect --result_dir {out_dir} \
                {'--train_dataset' if use_train_dataset else ''} \
                {'--val_dataset' if use_val_dataset else ''} --target_precision {target_precision} \
                --rpn_num_samples {rpn_mc_samples} --rpn_reduction_method {rpn_reduction} \
                {data_dir} {in_domain_dataset} {ood_dataset}"
        logging.info(f"OOD EVAL command being executed: {cmd}")
        os.system(cmd)

    if run_attack is True:
        epsilons = " ".join(map(lambda x: str(x),epsilon_list))
        time = int(datetime.timestamp(datetime.now()))
        out_dir = os.path.join(model_dir, f"attack-{attack_strategy}-{attack_criteria}-{attack_type}-{ood_dataset}-{time}")
        dataset_limit = f'--dataset_size_limit {dataset_size_limit}' if dataset_size_limit is not None else ''
        attack_only_out_dist = f'--attack_only_out_dist' if attack_only_out_dist is True else ''
        attack_cmd = f"python -m robust_priornet.attack_priornet {gpu_list} \
                --batch_size {batch_size} --epsilon {epsilons} \
                --attack_type {attack_type} --attack_strategy {attack_strategy} \
                --attack_criteria {attack_criteria} --target_precision {target_precision} \
                --norm {attack_norm} --model_dir {model_dir} --max_steps {max_steps} \
                --threshold {threshold} --ood_dataset {ood_dataset} \
                {'--train_dataset' if use_train_dataset else ''} {attack_only_out_dist} \
                {'--val_dataset' if use_val_dataset else ''} {dataset_limit}\
                {data_dir} {in_domain_dataset} {out_dir}"
        logging.info(f"{attack_strategy} attack command being executed: {attack_cmd}")
        os.system(attack_cmd)

    if run_certification is True:
        dataset_limit = f'--dataset_size_limit {dataset_size_limit}' if dataset_size_limit is not None else ''
        only_ood = '--only_ood' if certify_only_ood else ''
        time = int(datetime.timestamp(datetime.now()))
        out_dir =  os.path.join(model_dir, f"certify-results-{certify_task}-{uncertainty_measure}-{in_domain_dataset}-{ood_dataset}-{time}")
        certify_cmd = f"python -m robust_priornet.certify_priornet {gpu_list} \
                    --batch_size {batch_size} --model_dir {model_dir} \
                    {'--train_dataset' if use_train_dataset else ''} \
                    {'--val_dataset' if use_val_dataset else ''} {dataset_limit} \
                    --certify_task {certify_task} {only_ood} \
                    --rpn_num_samples {rpn_mc_samples} --rpn_reduction_method {rpn_reduction} \
                    --uncertainty_measure {uncertainty_measure} \
                    --uncertainty_measure_threshold {uncertainty_measure_threshold} \
                    --n0 {n0} --n {n} --sigma {sigma} \
                    {data_dir} {out_dir} {in_domain_dataset} {ood_dataset}"
        logging.info(f"Certification command being executed: {certify_cmd}")
        os.system(certify_cmd)

    # the returned result will be written into the database
    return ''
