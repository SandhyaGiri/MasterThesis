import argparse
import torch
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from ..datasets.torchvision_datasets import DatasetEnum
from ..datasets.torchvision_datasets import TorchVisionDataWrapper
from ..datasets.transforms import TransformsBuilder
from ..losses.dpn_loss import KLDivDirchletDistLoss, PriorNetWeightedLoss
from ..utils.pytorch import load_model
from .trainer import PriorNetTrainer

parser = argparse.ArgumentParser(description='Train a Prior Network model (esp Dirichlet prior) using a '
                                             'standard feed forward NN on a Torchvision '
                                             'dataset. Multi task training when both in-domain dataset '
                                             'and out-of-domain dataset are chosen.')
parser.add_argument('data_dir', type=str,
                    help='Absolute or relative path to dir where data resides.')
parser.add_argument('in_domain_dataset', type=str, choices=DatasetEnum.__dict__.keys(),
                    help='Name of the in-domain dataset to be used.')
parser.add_argument('ood_dataset', type=str, choices=DatasetEnum.__dict__.keys(),
                    help='Name of the out-of-domain dataset to be used.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='Absolute directory path where to load the model from.')
parser.add_argument('--target_precision', type=int, default=1e3,
                    help='Indicates the alpha_0 or the precision of the target dirichlet for in domain samples.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for the optimizer.')  
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Specifies the number of epochs to train the model.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Specifies the number of samples to be batched while training the model.')

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Model doesn't exist!")

    model, ckpt = load_model(args.model_dir)

    # move to device if one available
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(device)}.")
    else:
        print(f"Using CPU device.")
        device = torch.device("cpu")

    model.to(device)

    # load the datasets
    vis = TorchVisionDataWrapper()
    trans = TransformsBuilder()
    trans.add_resize(ckpt['model_params']['n_in'])
    trans.add_to_tensor()
    trans.add_Normalize((0.5,),(0.5,))

    id_train_set, id_val_set = vis.getDataset(args.in_domain_dataset, 
                                                args.data_dir, 
                                                trans.get_transforms(), 
                                                None, 
                                                'train',
                                                val_ratio=0.1)
    print(f"In domain dataset: Train-{len(id_train_set)}, Val-{len(id_val_set)}")
    
    ood_train_set, ood_val_set = vis.getDataset(args.ood_dataset, 
                                                args.data_dir, 
                                                trans.get_transforms(), 
                                                None, 
                                                'train',
                                                val_ratio=0.1)
    print(f"OOD domain dataset: Train-{len(ood_train_set)}, Val-{len(ood_val_set)}")

    # make both datasets (id, ood) same size
    if len(ood_val_set) != len(id_val_set):
        final_size = np.min([len(ood_val_set), len(id_val_set)])
        ood_val_set = data.Subset(ood_val_set, np.arange(0, final_size))
        id_val_set = data.Subset(id_val_set, np.arange(0, final_size))

    if len(id_train_set) < len(ood_train_set):
        ratio = np.ceil(float(len(ood_train_set)) / float(len(id_train_set)))
        dataset_list = [id_train_set, ] * (int(ratio)) ## duplicate the id_dataset as much as its lesser than ood dataset
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
                                criterion, id_loss, ood_loss, optimizer, optimizer_params=optimizer_params,
                                lr_scheduler=optim.lr_scheduler.ExponentialLR,
                                lr_scheduler_params={'gamma': 0.95},
                                batch_size=args.batch_size)

    trainer.train(num_epochs=args.num_epochs)
    
if __name__ == "__main__":
    main()