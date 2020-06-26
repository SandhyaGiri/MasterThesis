import argparse
import os

from .models.priornet_conv import vgg_model
from .models.priornet_mlp import PriorNetMLP
from .models.smoothed_priornet_conv import SmoothedPriorNetCount, SmoothedPriorNetSimple, SmoothedPriorNet
from .utils.pytorch import save_model

parser = argparse.ArgumentParser(description='Sets up a Prior Network model ' +
                                 '(esp Dirichlet prior) using the ' +
                                 'specified model architecture')

parser.add_argument('model_dir', type=str,
                    help='absolute directory path where to save the model.')
parser.add_argument('--model_arch', type=str, choices=['mlp', 'vgg6', 'vgg16'],
                    help='architecture of the model to be setup.')
parser.add_argument('--fc_layers', '--list', nargs='+', type=int, required=False,
                    help='List of fully connected layers each specifying dim of the hidden layer. \
                    Considered only for the MLP architecture.')
parser.add_argument('--num_classes', type=int, required=True,
                    help='Number of units in the final output layer.')
parser.add_argument('--input_size', type=int, required=True,
                    help='Indicates the size of the input image ex: 28 for 28*28 images.')
parser.add_argument('--num_channels', type=int, default=3,
                    help='Indicates the number of channels in the input image \
                    ex: 1 for 28*28*1 images.')
parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='Indicates the probability of dropping out hidden units.')
# robust PN params
parser.add_argument('--rpn_count', action='store_true',
                    help='Indicates if the model should be wrapped with a randomized smoothing layer (count vector based).')
parser.add_argument('--rpn_simple', action='store_true',
                    help='Indicates if the model should be wrapped with a randomized smoothing layer (logits summing based).')
parser.add_argument('--rpn', action='store_true',
                    help='Indicates if the model should be wrapped with a randomized smoothing layer (different training and eval behaviors)')
parser.add_argument('--rpn_sigma', type=float, default=0.2,
                    help='Specifies the std deviation of the gaussian dist to be used'+
                    ' for perturbing the input samples.')
parser.add_argument('--rpn_num_samples', type=int, default=1000,
                    help='large number of samples for accurately estimating prob using MC')
parser.add_argument('--rpn_reduction_method', choices=['mean', 'median', 'log_cosh'], default='mean',
                    help='Specifies how to reduce the logits generated from various noisy samples for a single image.')

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_params = {}
    model_params['num_channels'] = args.num_channels
    model_params['n_in'] = args.input_size
    model_params['n_out'] = args.num_classes
    model_params['drop_rate'] = args.drop_prob
    model_params['model_type'] = args.model_arch
    model_params['rpn_model'] = False
    if args.model_arch == 'mlp':
        mean = (0.5,)
        std = (0.5,)
        model_params['fc_layers'] = args.fc_layers
        model = PriorNetMLP(**model_params)
    elif args.model_arch == 'vgg16' or args.model_arch == 'vgg6':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        model = vgg_model(**model_params)
    # additional wrapper
    if args.rpn_count or args.rpn_simple or args.rpn:
        model_params['base_classifier'] = model
        model_params['image_normalization_params'] = {'mean': mean, 'std': std}
        model_params['noise_std_dev'] = args.rpn_sigma
        model_params['num_mc_samples'] = args.rpn_num_samples
        model_params['reduction_method'] = args.rpn_reduction_method
        model_params['rpn_model'] = True
        if args.rpn_simple:
            model = SmoothedPriorNetSimple(**model_params)
        elif args.rpn_count:
            model = SmoothedPriorNetCount(**model_params)
        elif args.rpn:
            model = SmoothedPriorNet(**model_params)

    print(model)
    save_model(model, model_params, args.model_dir)

if __name__ == '__main__':
    main()
