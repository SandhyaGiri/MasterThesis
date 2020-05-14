import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from .datasets.adversarial_dataset import AdversarialDataset
from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .losses.dpn_loss import KLDivDirchletDistLoss, PriorNetWeightedLoss
from .training.trainer import PriorNetTrainer
from .utils.common_data import ATTACK_CRITERIA_MAP, OOD_ATTACK_CRITERIA_MAP
from .utils.dataspliter import DataSpliter
from .utils.persistence import persist_image_dataset
from .utils.pytorch import choose_torch_device, load_model

parser = argparse.ArgumentParser(description='Train a Prior Network model (esp Dirichlet prior) using a '
                                             'standard feed forward NN on a Torchvision '
                                             'dataset. Multi task training when both in-domain dataset '
                                             'and out-of-domain dataset are chosen.')
parser.add_argument('data_dir', type=str,
                    help='Absolute or relative path to dir where data resides.')
parser.add_argument('in_domain_dataset', type=str, choices=DatasetEnum._member_map_.keys(),
                    help='Name of the in-domain dataset to be used.')
parser.add_argument('ood_dataset', type=str, choices=DatasetEnum._member_map_.keys(),
                    help='Name of the out-of-domain dataset to be used.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='Absolute directory path where to load the model from.')
parser.add_argument('--target_precision', type=int, default=1e3,
                    help='Indicates the alpha_0 or the precision of the target dirichlet \
                    for in domain samples.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for the optimizer.')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Specifies the number of epochs to train the model.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Specifies the number of samples to be batched while training the model.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specifies the GPU ids to run the script on.')
# adversarial training arguments
parser.add_argument('--include_adv_samples', action='store_true',
                    help='Specifies if adversarial samples should be augmented while training'+
                    ' to create a robust model.')
parser.add_argument('--adv_attack_type', type=str, choices=['FGSM', 'PGD'], default='FGSM',
                    help='Choose the type of attack used to generate adv samples, such that'+
                    ' resulting model is immune to the chosen attack.')
parser.add_argument('--adv_attack_criteria', type=str, choices=ATTACK_CRITERIA_MAP.keys(),
                    required=True,
                    help='Indicates which loss function to use to compute attack gradient'+
                    ' for generating adversarial samples.')
parser.add_argument('--adv_epsilon', type=float, default=0.3,
                    help='Strength of perturbation in range of 0 to 1, ex: 0.25,'+
                    ' to generate adversarial samples.', required=True)
# PGD adversarial training arguments
parser.add_argument('--pgd_norm', type=str, choices=['inf', '2'], default='inf',
                    help='The type of norm ball to restrict the adversarial samples to.' +
                    'Needed only for PGD adversarial training.')
parser.add_argument('--pgd_step_size', type=float, default=0.4,
                    help='The size of the gradient update step, used during each iteration'+
                    ' in PGD attack.')
parser.add_argument('--pgd_max_steps', type=int, default=10,
                    help='The number of the gradient update steps, to be performed for PGD attack.')
def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Model doesn't exist!")

    # move to device if one available
    device = choose_torch_device(args.gpu)
    model, ckpt = load_model(args.model_dir, device=device)
    model.to(device)

    # load the datasets
    vis = TorchVisionDataWrapper()

    # build transforms
    trans = TransformsBuilder()
    # TODO - change mean std to autoamtic calc based on dataset
    mean = (0.5,)
    std = (0.5,)
    trans.add_resize(ckpt['model_params']['n_in'])
    num_channels = ckpt['model_params']['num_channels']
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)

    id_train_set, id_val_set = vis.get_dataset(args.in_domain_dataset,
                                               args.data_dir,
                                               trans.get_transforms(),
                                               None,
                                               'train',
                                               val_ratio=0.1)
    print(f"In domain dataset: Train-{len(id_train_set)}, Val-{len(id_val_set)}")
    if args.include_adv_samples:
        id_train_adv_set = AdversarialDataset(id_train_set, args.adv_attack_type.lower(),
                                              model, args.adv_epsilon,
                                              ATTACK_CRITERIA_MAP[args.adv_attack_criteria],
                                              args.pgd_norm, args.pgd_step_size, args.pgd_max_steps,
                                              batch_size=args.batch_size, device=device)
        print(f"In domain adversarial dataset: Train-{len(id_train_adv_set)}")
        id_train_set = data.ConcatDataset([id_train_set, id_train_adv_set])

    ood_train_set, ood_val_set = vis.get_dataset(args.ood_dataset,
                                                 args.data_dir,
                                                 trans.get_transforms(),
                                                 None,
                                                 'train',
                                                 val_ratio=0.1)
    print(f"OOD domain dataset: Train-{len(ood_train_set)}, Val-{len(ood_val_set)}")
    if args.include_adv_samples:
        ood_train_adv_set = AdversarialDataset(ood_train_set, args.adv_attack_type.lower(),
                                               model, args.adv_epsilon,
                                               OOD_ATTACK_CRITERIA_MAP[args.adv_attack_criteria],
                                               args.pgd_norm, args.pgd_step_size,
                                               args.pgd_max_steps,
                                               batch_size=args.batch_size,
                                               device=device)
        print(f"OOD domain adversarial dataset: Train-{len(ood_train_adv_set)}")
        adv_folder = os.path.join(args.model_dir, "adv-images-ood")
        if not os.path.exists(adv_folder):
            os.makedirs(adv_folder)
        persist_image_dataset(ood_train_adv_set, mean, std, num_channels, adv_folder)
        ood_train_set = data.ConcatDataset([ood_train_set, ood_train_adv_set])

    # make both datasets (id, ood) same size
    if len(ood_val_set) != len(id_val_set):
        final_size = np.min([len(ood_val_set), len(id_val_set)])
        ood_val_set = data.Subset(ood_val_set, np.arange(0, final_size))
        id_val_set = data.Subset(id_val_set, np.arange(0, final_size))

    if len(id_train_set) < len(ood_train_set):
        ratio = np.ceil(float(len(ood_train_set)) / float(len(id_train_set)))
        # duplicate the id_dataset as much as its lesser than ood dataset
        dataset_list = [id_train_set, ] * (int(ratio))
        id_train_set = data.ConcatDataset(dataset_list)
    elif len(id_train_set) > len(ood_train_set):
        ratio = np.ceil(float(len(id_train_set)) / float(len(ood_train_set)))
        dataset_list = [ood_train_set, ] * (int(ratio))
        ood_train_set = data.ConcatDataset(dataset_list)
        if len(ood_train_set) > len(id_train_set):
            ood_train_set = data.Subset(ood_train_set, np.arange(0, len(id_train_set)))

    print(f"(After equalizing) Train dataset length: {len(id_train_set)}")
    print(f"(After equalizing) Validation dataset length: {len(id_val_set)}")

    # loss criteria
    id_loss = KLDivDirchletDistLoss(target_precision=args.target_precision)
    ood_loss = KLDivDirchletDistLoss(target_precision=0.0)
    criterion = PriorNetWeightedLoss([id_loss, ood_loss], weights=[1.0, 1.0])

    # optimizer
    optimizer = optim.Adam
    optimizer_params = {'lr': args.lr,
                        'betas': (0.9, 0.999),
                        'weight_decay': 0.0} # add this for other datasets

    trainer = PriorNetTrainer(model,
                              id_train_set, id_val_set, ood_train_set, ood_val_set,
                              criterion, id_loss, ood_loss, optimizer,
                              optimizer_params=optimizer_params,
                              lr_scheduler=optim.lr_scheduler.ExponentialLR,
                              lr_scheduler_params={'gamma': 0.95},
                              batch_size=args.batch_size, device=device,
                              log_dir=args.model_dir)

    trainer.train(num_epochs=args.num_epochs)
    
if __name__ == "__main__":
    main()
