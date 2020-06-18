import argparse
import os

import numpy as np
import torch

from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .eval.misclassification_detection import \
    MisclassificationDetectionEvaluator
from .eval.model_prediction_eval import ClassifierPredictionEvaluator
from .eval.out_of_domain_detection import OutOfDomainDetectionEvaluator
from .eval.uncertainty import UncertaintyEvaluator
from .utils.dataspliter import DataSpliter
from .utils.persistence import persist_image_dataset
from .utils.pytorch import (choose_torch_device, eval_model_on_dataset,
                            load_model)

parser = argparse.ArgumentParser(description='Evaluates a Prior Network model ' +
                                 '(esp Dirichlet prior) for either misclassification ' +
                                 'or out of domain detection tasks')

parser.add_argument('data_dir', type=str,
                    help='Absolute or relative path to dir where data resides.')
parser.add_argument('in_domain_dataset', type=str, choices=DatasetEnum._member_map_.keys(),
                    help='Name of the in-domain dataset to be used.')
parser.add_argument('ood_dataset', type=str, choices=DatasetEnum._member_map_.keys(), default=None,
                    help='Name of the out-of-domain dataset to be used' +
                    '(for ood detection task).')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path of the model to be evaluated.')
parser.add_argument('--result_dir', type=str, default='./',
                    help='absolute directory path where the results of the evaluation' +
                    'should be saved.')
parser.add_argument('--task', choices=['misclassification_detect', 'ood_detect'],
                    help='indicates what evaluation task to be performed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Specifies the number of samples to be batched' +
                    'while evaluating the model.')
parser.add_argument('--train_dataset', action='store_true',
                    help='Whether to evaluate on the training data instead of test data')
parser.add_argument('--val_dataset', action='store_true',
                    help='Whether to evaluate on the val data instead of train/test dataset.')
parser.add_argument('--dataset_size_limit', type=int, default=None,
                    help='Specifies the number of samples to consider in the loaded datasets.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specifies the GPU ids to run the script on.')

def log_model_predictions(logits, probs, labels,
                          uncertainty_measures : dict, prefix_name='model', result_dir='./'):
    np.savetxt(os.path.join(result_dir, f'{prefix_name}_labels.txt'), labels)
    np.savetxt(os.path.join(result_dir, f'{prefix_name}_probs.txt'), probs)
    np.savetxt(os.path.join(result_dir, f'{prefix_name}_logits.txt'), logits)
    for key in uncertainty_measures.keys():
        np.savetxt(os.path.join(result_dir,
                                f'{prefix_name}_{key._value_}.txt'), uncertainty_measures[key])

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Model doesn't exist!")

    if args.task == 'ood_detect' and args.ood_dataset is None:
        raise ValueError("Please provide a OOD dataset for this task to be evaluated.")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # move to device if one available
    device = choose_torch_device(args.gpu)
    model, ckpt = load_model(args.model_dir, device=device)

    # load the datasets
    vis = TorchVisionDataWrapper()

    # build transforms
    trans = TransformsBuilder()
    # TODO - change mean std to automatic calc based on dataset
    mean = (0.5,)
    std = (0.5,)
    num_channels = ckpt['model_params']['num_channels']
    trans.add_resize(ckpt['model_params']['n_in'])
    trans.add_center_crop(ckpt['model_params']['n_in'])
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)

    if args.val_dataset:
        _, id_test_set = vis.get_dataset(args.in_domain_dataset,
                                         args.data_dir,
                                         trans.get_transforms(),
                                         None,
                                         'train',
                                         val_ratio=0.1)
    else:
        id_test_set = vis.get_dataset(args.in_domain_dataset,
                                      args.data_dir,
                                      trans.get_transforms(),
                                      None,
                                      'train' if args.train_dataset else 'test')
    if args.dataset_size_limit is not None:
        id_test_set = DataSpliter.reduceSize(id_test_set, args.dataset_size_limit)

    print(f"In domain dataset: {len(id_test_set)}")

    if args.task == 'ood_detect':
        if args.val_dataset:
            _, ood_test_set = vis.get_dataset(args.ood_dataset,
                                              args.data_dir,
                                              trans.get_transforms(),
                                              None,
                                              'train',
                                              val_ratio=0.1)
        else:
            ood_test_set = vis.get_dataset(args.ood_dataset,
                                           args.data_dir,
                                           trans.get_transforms(),
                                           None,
                                           'train' if args.train_dataset else 'test')
        if args.dataset_size_limit is not None:
            ood_test_set = DataSpliter.reduceSize(ood_test_set, args.dataset_size_limit)
        # persist_image_dataset(ood_test_set, mean, std, num_channels, args.result_dir)
        print(f"OOD domain dataset: {len(ood_test_set)}")

    # Compute model predictions by passing the test set through the model.
    id_logits, id_probs, id_labels = eval_model_on_dataset(model,
                                                           id_test_set,
                                                           args.batch_size,
                                                           device=device)
    # Evaluate uncertainty measures
    id_uncertainties = UncertaintyEvaluator(id_logits).get_all_uncertainties()
    # log results so far
    log_model_predictions(id_logits, id_probs, id_labels, id_uncertainties,
                          prefix_name='id', result_dir=args.result_dir)
    model_accuracy = ClassifierPredictionEvaluator.compute_accuracy(id_probs, id_labels)
    model_nll = ClassifierPredictionEvaluator.compute_nll(id_probs, id_labels)
    with open(os.path.join(args.result_dir, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100 * (1.0 - model_accuracy), 1)} \n')
        f.write(f'NLL: {np.round(model_nll, 3)} \n')

    if args.task == 'ood_detect':
        ood_logits, ood_probs, ood_labels = eval_model_on_dataset(model,
                                                                  ood_test_set,
                                                                  args.batch_size,
                                                                  device=device)
        ood_uncertainties = UncertaintyEvaluator(ood_logits).get_all_uncertainties()
        log_model_predictions(ood_logits, ood_probs, ood_labels, ood_uncertainties,
                              prefix_name='ood', result_dir=args.result_dir)

    if args.task == 'misclassification_detect':
        MisclassificationDetectionEvaluator(id_probs,
                                            id_labels,
                                            id_uncertainties,
                                            args.result_dir).eval()
    elif args.task == 'ood_detect':
        OutOfDomainDetectionEvaluator(id_uncertainties,
                                      ood_uncertainties,
                                      args.result_dir).eval()
    else:
        raise ValueError("Invalid evaluation task.")

if __name__ == "__main__":
    main()
