import argparse
import math
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from .datasets.adversarial_dataset import AdversarialDataset
from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .losses.dpn_loss import KLDivDirchletDistLoss, PriorNetWeightedLoss
from .training.adversarial_trainer import AdversarialPriorNetTrainer
from .training.trainer import PriorNetTrainer
from .utils.common_data import (ATTACK_CRITERIA_MAP,
                                ATTACK_CRITERIA_TO_ENUM_MAP,
                                OOD_ATTACK_CRITERIA_MAP)
from .utils.dataspliter import DataSpliter
from .utils.persistence import persist_image_dataset
from .utils.pytorch import choose_torch_device, load_model

# import torchvision
# from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--augment', action='store_true',
                    help='Indicates whether the dataset should be augmented by flipping, rotations etc.')
parser.add_argument('--target_precision', type=int, default=1e3,
                    help='Indicates the alpha_0 or the precision of the target dirichlet \
                    for in domain samples.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Weight for OOD loss.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for the optimizer.')
parser.add_argument('--use_cyclic_lr', action='store_true',
                    help='Indicates if OneCycleLr scheduler needs to be used for training.')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Specifies the number of epochs to train the model.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Specifies the L2 regularization stength.')
parser.add_argument('--add_ce_loss', action='store_true',
                    help='Specifies whether to use CE loss in addition to KL div PN loss.')
parser.add_argument('--ce_weight', type=float, default=0.5,
                    help='Specifies the weight to be used for CE loss when adding to KL div loss.')
parser.add_argument('--reverse_KL', action='store_true',
                    help='Indicates if distributions need to be reversed while computing the KL div loss.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Specifies the number of samples to be batched while training the model.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specifies the GPU ids to run the script on.')
parser.add_argument('--dataset_size_limit', type=int, default=None,
                    help='Specifies the number of samples to consider in the loaded datasets.')
parser.add_argument('--resume_from_ckpt', action='store_true',
                    help='Indicates if model needs to be laoded from checkpoint.tar for further training.')
parser.add_argument('--grad_clip_value', type=float, default=10.0,
                    help='Specifies the clip value for gradient clipping, to prevent exploding gradients.')
# step wise training arguments
parser.add_argument('--train_stepwise', action='store_true',
                    help='Indicates if the model needs to be trained at step level and not epoch level.')
parser.add_argument('--val_every_steps', type=int, default=100,
                    help='Indicates the step interval after which the validation needs to be done.')
# early stopping arguments
parser.add_argument('--min_train_epochs', type=int, default=10,
                    help='Indicates min num of epochs(steps) to train before val loss monitoring starts for early stopping.')
parser.add_argument('--patience', type=int, default=3,
                    help='Number of epochs(steps) to wait before terminating training, once val loss starts increasing.')
# adversarial training arguments
parser.add_argument('--include_adv_samples', action='store_true',
                    help='Specifies if adversarial samples should be augmented while training'+
                    ' to create a robust model.')
parser.add_argument('--adv_training_type', choices=['normal', 'ood-detect'],
                    default='ood-detect',
                    help='identifies what type of adversarials should the model be trained on.'+
                    ' If normal, model is trained on id adv samples that lead to label misclassification'+
                    ' If ood-detect model is trained on id->ood and ood->id adversarials.')
parser.add_argument('--adv_model_dir', type=str, default=None,
                    help='Absolute directory path where to load the model to be used for generating adv samples.')
parser.add_argument('--adv_attack_type', type=str, choices=['FGSM', 'PGD'], default='FGSM',
                    help='Choose the type of attack used to generate adv samples, such that'+
                    ' resulting model is immune to the chosen attack.')
parser.add_argument('--adv_attack_criteria', type=str, choices=ATTACK_CRITERIA_MAP.keys(),
                    required=False, default='diff_entropy',
                    help='Indicates which loss function to use to compute attack gradient'+
                    ' for generating adversarial samples.')
parser.add_argument('--adv_epsilon', type=float, default=0.3,
                    help='Strength of perturbation in range of 0 to 1, ex: 0.25,'+
                    ' to generate adversarial samples.', required=False)
parser.add_argument('--adv_persist_images', action='store_true',
                    help='Specify this if you want to save the adv images created.')
parser.add_argument('--include_only_out_in_adv_samples', action='store_true',
                    help='Specifies to train the model using only out->in adversarials.')
parser.add_argument('--use_fixed_threshold', action='store_true',
                    help='Specifies whether to use the same threshold to identify adversarials during training.')
parser.add_argument('--known_threshold_value', type=float,
                    help='Fixed threshold value to be used during training.')
# PGD adversarial training arguments
parser.add_argument('--pgd_norm', type=str, choices=['inf', '2'], default='inf',
                    help='The type of norm ball to restrict the adversarial samples to.' +
                    'Needed only for PGD adversarial training.')
parser.add_argument('--pgd_step_size', type=float, default=0.4,
                    help='The size of the gradient update step, used during each iteration'+
                    ' in PGD attack.')
parser.add_argument('--pgd_max_steps', type=int, default=10,
                    help='The number of the gradient update steps, to be performed for PGD attack.')
# cyclic -lr arguments
parser.add_argument('--cyclic_lr_pct_start', type=float, default=0.33,
                    help='Indicates percentage of the cycle the LR needs to increases.')

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter()

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Model doesn't exist!")

    # move to device if one available
    device = choose_torch_device(args.gpu)
    model, ckpt = load_model(args.model_dir, device=device,
                             name='checkpoint.tar' if args.resume_from_ckpt else 'model.tar')
    model.to(device)

    if args.include_adv_samples and args.adv_model_dir is not None:
        print(f"Loaded old model for adversarial sample gen: {args.adv_model_dir}")
        adv_model, adv_ckpt = load_model(args.adv_model_dir, device=device)
        adv_model.to(device)

    # load the datasets
    vis = TorchVisionDataWrapper()

    # build transforms
    trans = TransformsBuilder()
    mean = (0.5,)
    std = (0.5,)
    num_channels = ckpt['model_params']['num_channels']
    trans.add_resize(ckpt['model_params']['n_in'])
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_center_crop(ckpt['model_params']['n_in'])
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    if args.augment:
        trans.add_padding(4)
        trans.add_rotation(15)
        trans.add_random_flipping()
        trans.add_random_crop(ckpt['model_params']['n_in'])
    trans.add_to_tensor()
    # normalize images to range (-1,1) - don't do it for RPN model
    if not ckpt['model_params']['rpn_model']:
        trans.add_normalize(mean, std)

    id_train_set, id_val_set = vis.get_dataset(args.in_domain_dataset,
                                               args.data_dir,
                                               trans.get_transforms(),
                                               None,
                                               'train',
                                               val_ratio=0.1)
    if args.dataset_size_limit is not None:
        id_train_set = DataSpliter.reduceSize(id_train_set, args.dataset_size_limit)
        id_val_set = DataSpliter.reduceSize(id_val_set, args.dataset_size_limit)
    print(f"In domain dataset: Train-{len(id_train_set)}, Val-{len(id_val_set)}")

    # images = torch.tensor(id_train_set[0][0]).unsqueeze(0)
    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('images', grid, 0)
    # writer.add_graph(model, images)
    # writer.close()
    
    ood_train_set, ood_val_set = vis.get_dataset(args.ood_dataset,
                                                 args.data_dir,
                                                 trans.get_transforms(),
                                                 None,
                                                 'train',
                                                 val_ratio=0.1)
    if args.dataset_size_limit is not None:
        ood_train_set = DataSpliter.reduceSize(ood_train_set, args.dataset_size_limit)
        ood_val_set = DataSpliter.reduceSize(ood_val_set, args.dataset_size_limit)
    # ood_train_set = DataSpliter.reduceSize(ood_train_set, 45000)
    print(f"OOD domain dataset: Train-{len(ood_train_set)}, Val-{len(ood_val_set)}")

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
        if len(id_train_set) > len(ood_train_set):
            id_train_set = data.Subset(id_train_set, np.arange(0, len(ood_train_set)))
    elif len(id_train_set) > len(ood_train_set):
        ratio = np.ceil(float(len(id_train_set)) / float(len(ood_train_set)))
        dataset_list = [ood_train_set, ] * (int(ratio))
        ood_train_set = data.ConcatDataset(dataset_list)
        if len(ood_train_set) > len(id_train_set):
            ood_train_set = data.Subset(ood_train_set, np.arange(0, len(id_train_set)))

    print(f"(After equalizing) Train dataset length: {len(id_train_set)}")
    print(f"(After equalizing) Validation dataset length: {len(id_val_set)}")

    # loss criteria
    id_loss = KLDivDirchletDistLoss(target_precision=args.target_precision, reverse_KL=args.reverse_KL)
    ood_loss = KLDivDirchletDistLoss(target_precision=0.0, reverse_KL=args.reverse_KL)
    criterion = PriorNetWeightedLoss([id_loss, ood_loss], weights=[1.0, args.gamma])

    # optimizer
    optimizer = optim.Adam
    optimizer_params = {'lr': args.lr,
                        'betas': (0.9, 0.999),
                        'weight_decay': args.weight_decay} # add this for other datasets

    # lr scheduler
    if args.use_cyclic_lr:
        lr_scheduler = optim.lr_scheduler.OneCycleLR
        lr_scheduler_params = {'max_lr': 10*args.lr,
                               'div_factor': 10,
                               'final_div_factor': args.lr / 1e-6,
                               'epochs': args.num_epochs,
                               'steps_per_epoch': math.ceil(len(id_train_set)/args.batch_size),
                               'pct_start': args.cyclic_lr_pct_start,
                               'anneal_strategy': 'linear',
                               }
    else:
        lr_scheduler = optim.lr_scheduler.ExponentialLR
        lr_scheduler_params = {'gamma': 0.95}

    if args.include_adv_samples:
        trainer = AdversarialPriorNetTrainer(model,
                                             id_train_set, id_val_set,
                                             ood_train_set, ood_val_set,
                                             criterion, id_loss, ood_loss, optimizer,
                                             args.adv_attack_type, args.adv_attack_criteria,
                                             optimizer_params=optimizer_params,
                                             lr_scheduler=lr_scheduler,
                                             lr_scheduler_params=lr_scheduler_params,
                                             add_ce_loss=args.add_ce_loss,
                                             ce_weight=args.ce_weight,
                                             batch_size=args.batch_size, device=device,
                                             min_epochs=args.min_train_epochs, patience=args.patience,
                                             clip_norm=args.grad_clip_value,
                                             log_dir=args.model_dir, attack_params={
                                                 'epsilon': args.adv_epsilon,
                                                 'adv_persist_images': args.adv_persist_images,
                                                 'norm': args.pgd_norm,
                                                 'max_steps': args.pgd_max_steps,
                                                 'step_size': args.pgd_step_size
                                             }, dataset_persistence_params=[
                                                 mean,
                                                 std,
                                                 num_channels
                                             ],
                                             adv_training_type=args.adv_training_type,
                                             uncertainty_measure=
                                             ATTACK_CRITERIA_TO_ENUM_MAP[args.adv_attack_criteria],
                                             only_out_in_adversarials=
                                             args.include_only_out_in_adv_samples,
                                             use_fixed_threshold=args.use_fixed_threshold,
                                             known_threshold_value=args.known_threshold_value)
    else:
        trainer = PriorNetTrainer(model,
                                  id_train_set, id_val_set, ood_train_set, ood_val_set,
                                  criterion, id_loss, ood_loss, optimizer,
                                  optimizer_params=optimizer_params,
                                  lr_scheduler=lr_scheduler,
                                  add_ce_loss=args.add_ce_loss,
                                  ce_weight=args.ce_weight,
                                  lr_scheduler_params=lr_scheduler_params,
                                  batch_size=args.batch_size, device=device,
                                  min_epochs=args.min_train_epochs, patience=args.patience,
                                  clip_norm=args.grad_clip_value,
                                  log_dir=args.model_dir)

    trainer.train(num_epochs=args.num_epochs,
                  resume=args.resume_from_ckpt,
                  ckpt=ckpt,
                  stepwise_train=args.train_stepwise,
                  val_after_steps=args.val_every_steps)

if __name__ == "__main__":
    main()
