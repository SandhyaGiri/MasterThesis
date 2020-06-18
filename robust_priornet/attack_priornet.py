import argparse
import copy
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset

from .datasets.adversarial_dataset import AdversarialDataset
from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .eval.misclassification_detection import \
    MisclassificationDetectionEvaluator
from .eval.model_prediction_eval import ClassifierPredictionEvaluator
from .eval.out_of_domain_detection import OutOfDomainDetectionEvaluator
from .eval.uncertainty import UncertaintyEvaluator, UncertaintyMeasuresEnum
from .losses.attack_loss import AttackCriteria
from .utils.common_data import (ATTACK_CRITERIA_MAP,
                                ATTACK_CRITERIA_TO_ENUM_MAP,
                                OOD_ATTACK_CRITERIA_MAP)
from .utils.dataspliter import DataSpliter
from .utils.persistence import persist_image_dataset
from .utils.pytorch import (choose_torch_device, eval_model_on_dataset,
                            load_model)
from .utils.visualizer import plot_epsilon_curve

matplotlib.use('agg')

parser = argparse.ArgumentParser(description='Constructs an FGSM/PGD attack on test images and \
                    reports the results as epsilon vs adversarial success rate.')
parser.add_argument('data_dir', type=str,
                    help='Path to the directory where data is saved')
parser.add_argument('dataset', choices=DatasetEnum._member_map_.keys(),
                    help='Specify name of dataset to be used for creating adversarial data.')
parser.add_argument('result_dir', type=str,
                    help='Path of directory for saving the attack resultss.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to the saved model exists.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--train_dataset', action='store_true',
                    help='Whether to evaluate on the training data instead of test data')
parser.add_argument('--val_dataset', action='store_true',
                    help='Whether to evaluate on the val data instead of train/test dataset.')
parser.add_argument('--dataset_size_limit', type=int, default=None,
                    help='Specifies the number of samples to consider in the loaded datasets.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specifies the GPU ids to run the script on.')
# attack parameters
parser.add_argument('--attack_type', choices=['misclassify', 'ood-detect'],
                    default='misclassify',
                    help='Choose the type of attack used to generate adversaries.')
parser.add_argument('--attack_strategy', type=str, choices=['FGSM', 'PGD'], default='FGSM',
                    help='Choose the category of attack to be performed.')
parser.add_argument('--attack_criteria', type=str, choices=ATTACK_CRITERIA_MAP.keys(),
                    required=True,
                    help='Indicates which loss function to use to compute attack gradient.')
parser.add_argument('--epsilon', '--list', nargs='+', type=float,
                    help='Strength perturbation in range of 0 to 1, ex: 0.25', required=True)
parser.add_argument('--threshold', type=float, required=False,
                    help='Needed for ood-detect attacks, where threshold is used '+
                    'to make the binary classification between in and out domain samples, '+
                    'which is used to determine a true adversary.')
parser.add_argument('--norm', type=str, choices=['inf', '2'], default='inf',
                    help='The type of norm ball to restrict the adversarial samples to.' +
                    'Needed only for PGD attack.')
parser.add_argument('--step_size', type=float, default=0.4,
                    help='The size of the gradient update step, used during each iteration'+
                    ' in PGD attack.')
parser.add_argument('--max_steps', type=int, default=10,
                    help='The number of the gradient update steps, to be performed for PGD attack.')
parser.add_argument('--ood_dataset', choices=DatasetEnum._member_map_.keys(),
                    help='Dataset to be used for ood-detect attack adversary generation '+
                    'and evaluation of the attack success.')

def _get_ood_success(id_uncertainty, ood_uncertainty, uncertainty_measure, threshold, verbose=False):
    uncertainty_pred = np.concatenate((id_uncertainty, ood_uncertainty), axis=0)
    if uncertainty_measure == UncertaintyMeasuresEnum.CONFIDENCE:
        uncertainty_pred *= -1.0
    id_labels = np.zeros_like(id_uncertainty)
    ood_labels = np.ones_like(ood_uncertainty)
    y_true = np.concatenate((id_labels, ood_labels), axis=0)
    tn, fp, fn, tp = ClassifierPredictionEvaluator.compute_confusion_matrix_entries(
        uncertainty_pred, y_true, threshold=threshold)
    if verbose is True:
        print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp},"+
                f" total_id: {len(id_labels)}, total_ood: {len(ood_labels)}")
    id_success = (fp / len(id_labels)) # in domain -> out domain
    ood_success = (fn / len(ood_labels)) # out domain -> in domain
    return id_success, ood_success

def plot_ood_attack_success(epsilons: list, attack_criteria: UncertaintyMeasuresEnum,
                            thresholds: list, attack_dir: str, result_dir: str, verbose=True):
    adv_success_id = [] # in domain sample gets classified as out domain
    adv_success_ood = [] # out domain sample gets classified as in domain
    for i, epsilon in enumerate(epsilons, 0):
        target_epsilon_dir = os.path.join(attack_dir, f"e{epsilon}-attack")
        id_uncertainty = np.loadtxt(f"{target_epsilon_dir}/eval/{attack_criteria._value_}.txt")
        ood_uncertainty = np.loadtxt(f"{target_epsilon_dir}/ood_eval/{attack_criteria._value_}.txt")
        id_success, ood_success = _get_ood_success(id_uncertainty, ood_uncertainty,
                                                   attack_criteria, thresholds[i],
                                                   verbose=True)
        if verbose is True:
            print(f"epsilon: {epsilon}, threshold: {thresholds[i]}")
            print(f"in->out success: {id_success}")
            print(f"out->in success: {ood_success}")
        # previously we assume all id-samples were classififed as id (label=0) and
        # all ood samples were classified as ood (label=1).
        adv_success_id.append(id_success)
        adv_success_ood.append(ood_success)
    plot_epsilon_curve(epsilons, adv_success_id, result_dir=result_dir, file_name='epsilon-curve_id.png')
    plot_epsilon_curve(epsilons, adv_success_ood, result_dir=result_dir, file_name='epsilon-curve_ood.png')
    return adv_success_id, adv_success_ood

def plot_mis_adv_success(org_eval_dir: str, attack_dir: str, epsilons: list, result_dir: str):
    old_probs = np.loadtxt(os.path.join(org_eval_dir, 'id_probs.txt'))
    labels = np.loadtxt(os.path.join(org_eval_dir, 'id_labels.txt'))
    org_preds = np.argmax(old_probs, axis=1)
    correct_classifications = np.asarray(org_preds == labels, dtype=np.int32)
    correct_classified_indices = np.argwhere(correct_classifications == 1)

    adv_success_rates = []
    for epsilon in epsilons:
        probs = np.loadtxt(os.path.join(attack_dir, f'e{epsilon}-attack', 'eval', 'probs.txt'))
        preds = np.argmax(probs, axis=1)
        misclassifications = np.asarray(preds != labels, dtype=np.int32)
        misclassified_indices = np.argwhere(misclassifications == 1)
        adv_success = len(np.intersect1d(correct_classified_indices, misclassified_indices))
        adv_success_rates.append(adv_success / len(correct_classified_indices))

    plot_epsilon_curve(epsilons, adv_success_rates, result_dir=result_dir)
    return adv_success_rates

def perform_epsilon_attack(model: nn.Module, adv_dataset: Dataset, correct_classified_indices,
                           attack_criteria_enum=None, threshold=0.0, batch_size=128,
                           device=None, result_dir='./', ood_adv_dataset=None):
    logits, probs, labels = eval_model_on_dataset(model, adv_dataset, batch_size, device=device)

    # Save model outputs
    eval_dir = os.path.join(result_dir, 'eval')
    os.makedirs(eval_dir)
    np.savetxt(os.path.join(eval_dir, 'labels.txt'), labels)
    np.savetxt(os.path.join(eval_dir, 'probs.txt'), probs)
    np.savetxt(os.path.join(eval_dir, 'logits.txt'), logits)

    # determine misclassifications under attack
    preds = np.argmax(probs, axis=1)
    misclassifications = np.asarray(preds != labels, dtype=np.int32)
    misclassified_indices = np.argwhere(misclassifications == 1)

    # count as success only those samples that are wrongly classified under attack,
    # while they were correctly classified without attack.
    adv_success = len(np.intersect1d(correct_classified_indices, misclassified_indices))
    adv_success = (adv_success / len(correct_classified_indices))

    # Get dictionary of uncertainties.
    uncertainties = UncertaintyEvaluator(logits).get_all_uncertainties()

    # Save uncertainties
    for key in uncertainties.keys():
        np.savetxt(os.path.join(eval_dir, key._value_ + '.txt'), uncertainties[key])

    # eval model's predictions
    model_accuracy = ClassifierPredictionEvaluator.compute_accuracy(probs, labels)
    model_nll = ClassifierPredictionEvaluator.compute_nll(probs, labels)
    with open(os.path.join(eval_dir, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100 * (1.0 - model_accuracy), 1)} \n')
        f.write(f'NLL: {np.round(model_nll, 3)} \n')

    # eval misclassification detect performance
    MisclassificationDetectionEvaluator(probs, labels, uncertainties, eval_dir).eval()

    # eval ood detect performance
    id_success = None
    ood_success = None
    if ood_adv_dataset is not None:
        ood_logits, ood_probs, _ = eval_model_on_dataset(model,
                                                         ood_adv_dataset,
                                                         batch_size,
                                                         device=device)
        ood_uncertainties = UncertaintyEvaluator(ood_logits).get_all_uncertainties()
        ood_eval_dir = os.path.join(result_dir, 'ood_eval')
        os.makedirs(ood_eval_dir)
        np.savetxt(os.path.join(ood_eval_dir, 'logits.txt'), ood_logits)
        np.savetxt(os.path.join(ood_eval_dir, 'probs.txt'), ood_probs)
        for key in ood_uncertainties.keys():
            np.savetxt(os.path.join(ood_eval_dir, key._value_ + '.txt'), ood_uncertainties[key])
        OutOfDomainDetectionEvaluator(uncertainties, ood_uncertainties, ood_eval_dir).eval()
        id_success, ood_success = _get_ood_success(copy.deepcopy(uncertainties[attack_criteria_enum]),
                                                   copy.deepcopy(ood_uncertainties[attack_criteria_enum]),
                                                   attack_criteria_enum,
                                                   threshold, verbose=True)
    # return aversarial successes
    return {
        'misclassify_success': adv_success,
        'ood_detect_success': (id_success, ood_success)
    }

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Model doesn't exist!")

    if args.attack_type == 'ood-detect' and args.ood_dataset is None:
        raise ValueError("OOD dataset not specified for OOD-Detect attack type.")
    if args.attack_type == 'ood-detect' and args.threshold is None:
        raise ValueError("Threshold not specified for OOD-Detect attack type.")

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
    trans.add_resize(ckpt['model_params']['n_in'])
    trans.add_center_crop(ckpt['model_params']['n_in'])
    num_channels = ckpt['model_params']['num_channels']
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)

    if args.val_dataset:
        _, dataset = vis.get_dataset(args.dataset,
                                     args.data_dir,
                                     trans.get_transforms(),
                                     None,
                                     'train',
                                     val_ratio=0.1)
    else:
        dataset = vis.get_dataset(args.dataset,
                                  args.data_dir,
                                  trans.get_transforms(),
                                  None,
                                  'train' if args.train_dataset else 'test')
    if args.dataset_size_limit is not None:
        dataset = DataSpliter.reduceSize(dataset, args.dataset_size_limit)
    print(f"In domain dataset: {len(dataset)}")
    # save the org images
    id_indices = np.arange(len(dataset))
    id_chosen_indices = random.sample(list(id_indices), min(100, len(id_indices)))
    org_dataset_folder = os.path.join(args.result_dir, "org-images")
    if not os.path.exists(org_dataset_folder):
        os.makedirs(org_dataset_folder)
    np.savetxt(os.path.join(org_dataset_folder, 'img_indices.txt'), id_chosen_indices)
    persist_image_dataset(data.Subset(dataset, id_chosen_indices),
                          mean, std, num_channels, org_dataset_folder)

    # perform original evaluation on the model using unperturbed images
    logits, probs, labels = eval_model_on_dataset(model,
                                                  dataset=dataset,
                                                  device=device,
                                                  batch_size=args.batch_size)
    # determine correct classifications without attack (original non perturbed images)
    org_preds = np.argmax(probs, axis=1)
    correct_classifications = np.asarray(org_preds == labels, dtype=np.int32)
    correct_classified_indices = np.argwhere(correct_classifications == 1)

    # load ood dataset if atatck type is ood-detect.
    ood_dataset = None
    if args.attack_type == 'ood-detect':
        # load the dataset
        if args.val_dataset:
            _, ood_dataset = vis.get_dataset(args.ood_dataset,
                                             args.data_dir,
                                             trans.get_transforms(),
                                             None,
                                             'train',
                                             val_ratio=0.1)
        else:
            ood_dataset = vis.get_dataset(args.ood_dataset,
                                          args.data_dir,
                                          trans.get_transforms(),
                                          None,
                                          'train' if args.train_dataset else 'test')
        if args.dataset_size_limit is not None:
            ood_dataset = DataSpliter.reduceSize(ood_dataset, args.dataset_size_limit)
        print(f"Out domain dataset: {len(ood_dataset)}")
        # save the ood org images
        ood_indices = np.arange(len(ood_dataset))
        ood_chosen_indices = random.sample(list(ood_indices), min(100, len(ood_indices)))
        org_ood_dataset_folder = os.path.join(args.result_dir, "org-images-ood")
        if not os.path.exists(org_ood_dataset_folder):
            os.makedirs(org_ood_dataset_folder)
        np.savetxt(os.path.join(org_ood_dataset_folder, 'img_indices.txt'), ood_chosen_indices)
        persist_image_dataset(data.Subset(ood_dataset, ood_chosen_indices),
                              mean, std, num_channels, org_ood_dataset_folder)

    # perform attacks on the same dataset, using different epsilon values.
    misclass_adv_success = []
    in_out_adv_success = []
    out_in_adv_success = []
    attack_criteria = ATTACK_CRITERIA_MAP[args.attack_criteria]
    ood_attack_criteria = OOD_ATTACK_CRITERIA_MAP[args.attack_criteria]
    adv_success_detect_type = 'normal' if args.attack_type == 'misclassify' else 'ood-detect'
    for epsilon in args.epsilon:
        attack_folder = os.path.join(args.result_dir, f"e{epsilon}-attack")
        out_path = os.path.join(attack_folder, "adv-images")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # create adversarial dataset - in domain samples
        adv_dataset = AdversarialDataset(dataset, args.attack_strategy.lower(), model, epsilon,
                                         attack_criteria, args.norm,
                                         args.step_size, args.max_steps,
                                         args.batch_size, device=device,
                                         only_true_adversaries=False,
                                         adv_success_detect_type=adv_success_detect_type,
                                         uncertainty_measure=
                                         ATTACK_CRITERIA_TO_ENUM_MAP[args.attack_criteria],
                                         uncertainty_threshold=args.threshold)
        print(f"In domain adversarial dataset: {len(adv_dataset)}")
        persist_image_dataset(data.Subset(adv_dataset, id_chosen_indices),
                              mean, std, num_channels, out_path)
        #create adversarial dataset - out domain samples
        ood_adv_dataset = None
        if args.attack_type == 'ood-detect':
            ood_adv_dataset = AdversarialDataset(ood_dataset, args.attack_strategy.lower(),
                                                 model, epsilon, ood_attack_criteria,
                                                 args.norm, args.step_size, args.max_steps,
                                                 args.batch_size, device=device,
                                                 only_true_adversaries=False,
                                                 adv_success_detect_type=adv_success_detect_type,
                                                 ood_dataset=True,
                                                 uncertainty_measure=
                                                 ATTACK_CRITERIA_TO_ENUM_MAP[args.attack_criteria],
                                                 uncertainty_threshold=args.threshold)
            print(f"Out domain adversarial dataset: {len(ood_adv_dataset)}")
            ood_adv_folder = os.path.join(attack_folder, "adv-images-ood")
            if not os.path.exists(ood_adv_folder):
                os.makedirs(ood_adv_folder)
            persist_image_dataset(data.Subset(ood_adv_dataset, ood_chosen_indices),
                                  mean, std, num_channels, ood_adv_folder)
        # assess model perf using adv images
        adv_success = perform_epsilon_attack(model, adv_dataset, correct_classified_indices,
                                             attack_criteria_enum=
                                             ATTACK_CRITERIA_TO_ENUM_MAP[args.attack_criteria],
                                             threshold=args.threshold,
                                             batch_size=args.batch_size, device=device,
                                             result_dir=attack_folder,
                                             ood_adv_dataset=ood_adv_dataset)
        misclass_adv_success.append(adv_success['misclassify_success'])
        if adv_success['ood_detect_success'][0] is not None:
            in_out_adv_success.append(adv_success['ood_detect_success'][0])
            out_in_adv_success.append(adv_success['ood_detect_success'][1])

    # plot the epsilon, adversarial success rate graph (line plot)
    plot_epsilon_curve(args.epsilon, misclass_adv_success, args.result_dir,
                       file_name='epsilon-curve_misclassify.png',
                       title=f'{args.attack_criteria} attack - {args.dataset} - Misclassification success')
    if args.attack_type == "ood-detect":
        plot_epsilon_curve(args.epsilon, in_out_adv_success, args.result_dir,
                           file_name='epsilon-curve_id.png',
                           title=f'{args.attack_criteria} attack - {args.dataset} + {args.ood_dataset} - in->out success')
        plot_epsilon_curve(args.epsilon, out_in_adv_success, args.result_dir,
                           file_name='epsilon-curve_ood.png',
                           title=f'{args.attack_criteria} attack - {args.dataset} + {args.ood_dataset} - out->in success')

if __name__ == '__main__':
    main()
