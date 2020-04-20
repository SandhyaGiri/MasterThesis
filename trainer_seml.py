import logging
import os
from sacred import Experiment
import numpy as np
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
def run(in_domain_dataset, ood_dataset, input_image_size, num_classes, model_arch, fc_layers, num_epochs, num_channels, learning_rate, drop_rate, model_dir, data_dir, lr_decay_milestones, batch_size, logdir):

    logging.info('Received the following configuration:')
    logging.info(f'In domain dataset: {in_domain_dataset}, OOD dataset: {ood_dataset}')

    os.system('set | grep SLURM | while read line; do echo "# $line"; done')
    cuda_devices = os.environ['SLURM_JOB_GPUS']
    logging.info(f"GPUs assigned to me: {cuda_devices}")

    # set up the model
    fc_layers_list = " ".join(map(lambda x: str(x),fc_layers))
    setup_cmd = f'python -m robust_priornet.training.setup_priornet --model_arch {model_arch} \
        --fc_layers {fc_layers_list} --num_classes {num_classes} --input_size {input_image_size} \
        --drop_prob {drop_rate} --num_channels {num_channels} {model_dir}'
    logging.info(f"Setup command being executed: {setup_cmd}")
    os.system(setup_cmd)

    # training the model
    # lr_decay_milestones = " ".join(map(lambda epoch: "--lrc " + str(epoch), lr_decay_milestones))
    train_cmd = f'python -m robust_priornet.training.train_priornet --model_dir {model_dir} \
        --num_epochs {num_epochs} --batch_size {batch_size} --lr {learning_rate} {data_dir} \
            {in_domain_dataset} {ood_dataset}'
    logging.info(f"Training command being executed: {train_cmd}")
    os.system(train_cmd)

    # recover the logs from model_dir/log/LOG.txt file and return
    #with open(f'{model_dir}/LOG.txt', 'r') as f:
    #    results = f.read()
    
    # the returned result will be written into the database
    return {}