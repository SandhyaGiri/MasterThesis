import argparse
import torch
import os
from ..models.priornet_mlp import PriorNetMLP
from ..utils.pytorch import save_model

parser = argparse.ArgumentParser(description='Sets up a Prior Network model (esp Dirichlet prior) using the '
                                             'specified model architecture')

parser.add_argument('model_dir', type=str,
                    help='absolute directory path where to save the model.')
parser.add_argument('--model_arch', type=str, choices=['mlp', 'vgg16'],
                    help='architecture of the model to be setup.')
parser.add_argument('--fc_layers','--list', nargs='+', type=int,
                    help='List of fully connected layers each specifying dim of the hidden layer. \
                    Considered only for the MLP architecture.', required=True)
parser.add_argument('--num_classes', type=int, required=True,
                    help='Number of units in the final output layer.')
parser.add_argument('--input_size', type=int, required=True,
                    help='Indicates the size of the input image ex: 28 for 28*28 images.')
parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='Indicates the probability of dropping out hidden units.')

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_params = {}
    if args.model_arch == 'mlp':
        model_params['fc_layers'] = args.fc_layers
        model_params['n_in'] = args.input_size
        model_params['n_out'] = args.num_classes
        model_params['drop_rate'] = args.drop_prob
        model = PriorNetMLP(**model_params)
    elif args.model_arch == 'vgg16':
        pass

    print(model)
    save_model(model, model_params, args.model_dir)

if __name__ == '__main__':
    main()
    