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
def run(in_domain_dataset, ood_dataset, model_dir, data_dir, batch_size, logdir,
        run_eval, run_attack, epsilon_list, attack_type, attack_norm):
    """
    Performs both in-domain evaluation, and ood evaluation and the corresponding results
    are stored under eval/ and ood-eval/ inside the model_dir.
    """
    logging.info('Received the following configuration:')
    logging.info(f'In domain dataset: {in_domain_dataset}, OOD dataset: {ood_dataset}')

    os.system('set | grep SLURM | while read line; do echo "# $line"; done')
    cuda_devices = os.environ['SLURM_JOB_GPUS']
    logging.info(f"GPUs assigned to me: {cuda_devices}")

    # Check that we are training on a sensible GPU
    if os.environ.get('SLURM_JOB_GPUS', None) is not None:
        gpu_list = list(map(int, os.environ['SLURM_JOB_GPUS'].split(",")))
        gpu_list = " ".join(map(lambda gpu: "--gpu " + str(gpu), gpu_list))

    if run_eval is True:
        # in-domain evaluation ( + misclassification detection eval, as binary classification task)
        out_dir = os.path.join(model_dir, "eval")
        cmd = f"python -m robust_priornet.eval_priornet {gpu_list} --batch_size {batch_size} \
            --model_dir {model_dir} --task misclassification_detect --result_dir {out_dir} \
            {data_dir} {in_domain_dataset} {ood_dataset}"
        logging.info(f"In domain EVAL command being executed: {cmd}")
        os.system(cmd)

        # ood detection evaluation (as a binary classification task)
        out_dir = os.path.join(model_dir, "ood-eval")
        cmd = f"python -m robust_priornet.eval_priornet {gpu_list} --batch_size {batch_size} \
                --model_dir {model_dir} --task ood_detect {data_dir} \
                {in_domain_dataset} {ood_dataset} {out_dir}"
        logging.info(f"OOD EVAL command being executed: {cmd}")
        os.system(cmd)

    if run_attack is True:
        epsilons = " ".join(map(lambda x: str(x),epsilon_list))
        time = int(datetime.timestamp(datetime.now()))
        out_dir = os.path.join(model_dir, f"{attack_type}-attack-{time}")
        if attack_type == 'FGSM':
            fgsm_cmd = f"python -m robust_priornet.attack_priornet {gpu_list} \
                    --batch_size {batch_size} --epsilon {epsilons} --attack_type {attack_type} \
                    --model_dir {model_dir} {data_dir} {in_domain_dataset} {out_dir}"
            logging.info(f"FGSM attack command being executed: {fgsm_cmd}")
            os.system(fgsm_cmd)
        elif attack_type == "PGD":
            pgd_cmd = f"python -m robust_priornet.attack_priornet {gpu_list} \
                    --batch_size {batch_size} --epsilon {epsilons} --attack_type {attack_type} \
                    --norm {attack_norm} --model_dir {model_dir} \
                    {data_dir} {in_domain_dataset} {out_dir}"
            logging.info(f"PGD attack command being executed: {pgd_cmd}")
            os.system(pgd_cmd)

    # the returned result will be written into the database
    return ''