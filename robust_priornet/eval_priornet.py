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

    model, ckpt = load_model(args.model_dir)

    # move to device if one available
    device = choose_torch_device(args.gpu)
    model.to(device)

    # load the datasets
    vis = TorchVisionDataWrapper()

    # build transforms
    trans = TransformsBuilder()
    # TODO - change mean std to automatic calc based on dataset
    mean = (0.5,)
    std = (0.5,)
    trans.add_resize(ckpt['model_params']['n_in'])
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(ckpt['model_params']['num_channels'])
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)

    id_test_set = vis.get_dataset(args.in_domain_dataset,
                                  args.data_dir,
                                  trans.get_transforms(),
                                  None,
                                  'test')
    print(f"In domain dataset: {len(id_test_set)}")

    if args.task == 'ood_detect':
        ood_test_set = vis.get_dataset(args.ood_dataset,
                                       args.data_dir,
                                       trans.get_transforms(),
                                       None,
                                       'test')
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