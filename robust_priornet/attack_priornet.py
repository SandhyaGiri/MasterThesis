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
                                OOD_ATTACK_CRITERIA_MAP,
                                CHOSEN_THRESHOLDS)
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
parser.add_argument('--target_precision', type=int, required=False, default=0,
                    help='Indicates the alpha_0 or the precision of the target dirichlet \
                    for in domain samples. Provide this for precision attacks.')
# attack parameters
parser.add_argument('--attack_type', choices=['misclassify', 'ood-detect'],
                    default='misclassify',
                    help='Choose the type of attack used to generate adversaries.')
parser.add_argument('--attack_strategy', type=str, choices=['FGSM', 'PGD'], default='FGSM',
                    help='Choose the category of attack to be performed.')
parser.add_argument('--attack_criteria', type=str, choices=ATTACK_CRITERIA_MAP.keys(),
                    required=True,
                    help='Indicates which loss function to use to compute attack gradient.')
parser.add_argument('--attack_only_out_dist', action='store_true',
                    help='Indicates if onlu out samples need to be atatcked. Suitable for ood-detect attack type.')
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
parser.add_argument('--target_label', type=str, default='all',
                    help='Indicates the target label for which the attack wants to change the prediction to.'+
                    'To be used for a targeted attack.')

# fixes thresholds for evalutaing adv success!!
def _get_adv_success_criteria(attack_type, attack_criteria, threshold):
    if attack_type == 'misclassify':
        if attack_criteria == 'confidence':
            return {}
        elif attack_criteria == 'alpha_k':
            return {
                'alpha_k': CHOSEN_THRESHOLDS['alpha_k'],
                'precision': CHOSEN_THRESHOLDS['precision']
            }
        elif attack_criteria == 'precision_targeted':
            return {
                'precision': CHOSEN_THRESHOLDS['precision']
            }
    elif attack_type == 'ood-detect':
        if attack_criteria == 'precision':
            return {
                UncertaintyMeasuresEnum.PRECISION: -1 * CHOSEN_THRESHOLDS['precision'],
                #UncertaintyMeasuresEnum.DISTRIBUTIONAL_UNCERTAINTY: CHOSEN_THRESHOLDS['mutual_info'],
                #UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY: CHOSEN_THRESHOLDS['diff_entropy']
            }
        else:
            return {
                ATTACK_CRITERIA_TO_ENUM_MAP[attack_criteria]: threshold
            }
    
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

def _get_ood_success_precision(id_logits, ood_logits, target_precision, precision_threshold_fn, id_valid_indices, ood_valid_indices):
    k = id_logits.shape[1] # num_classes
    id_alpha_0 = np.sum(np.exp(id_logits), axis=1)
    ood_alpha_0 = np.sum(np.exp(ood_logits), axis=1)
    print(f"target_precision: {target_precision}, threshold: {precision_threshold_fn(k, target_precision)}")
    id_success_indices = np.argwhere(id_alpha_0 < precision_threshold_fn(k, target_precision))
    print(f"id_success: {len(id_success_indices)}")
    # from originally correcty classified samples with high precision
    id_success_indices = np.intersect1d(id_success_indices, id_valid_indices)
    print(f"id_success (final): {len(id_success_indices)}")
    ood_success_indices = np.argwhere(ood_alpha_0 >= precision_threshold_fn(k, target_precision))
    print(f"ood_success: {len(ood_success_indices)}")
    # from originally low precision ood samples
    ood_success_indices = np.intersect1d(ood_success_indices, ood_valid_indices)
    print(f"ood_success (final): {len(ood_success_indices)}")
    id_success = 0
    if len(id_valid_indices) > 0:
        id_success = len(id_success_indices)/len(id_valid_indices)
    ood_success = 0
    if len(ood_valid_indices) > 0:
        ood_success = len(ood_success_indices)/len(ood_valid_indices)
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

def _get_mis_adv_success(probs, labels, correct_classified_indices, good_alpha_indices=None):
    preds = np.argmax(probs, axis=1)
    misclassifications = np.asarray(preds != labels, dtype=np.int32)
    misclassified_indices = np.argwhere(misclassifications == 1)
    print(f"Correctly classified (normal) images: {len(correct_classified_indices)}")
    print(f"Misclassified (adv) images: {len(misclassified_indices)}")
    if good_alpha_indices is not None:
        print(f"Adv images with good precision: {len(good_alpha_indices)}")
    
    # count as success only those samples that are wrongly classified under attack,
    # while they were correctly classified without attack.
    adv_success_indices = np.intersect1d(correct_classified_indices, misclassified_indices)
    if good_alpha_indices is not None:
        adv_success_indices = np.intersect1d(adv_success_indices, good_alpha_indices)
    adv_success = len(adv_success_indices)
    print(f"Misclassified (adv) images, which were initially correctly classified: {adv_success}")
    return adv_success/len(correct_classified_indices) if len(correct_classified_indices) > 0 else 0

def plot_mis_adv_success(org_eval_dir: str, attack_dir: str, epsilons: list, result_dir: str):
    old_probs = np.loadtxt(os.path.join(org_eval_dir, 'id_probs.txt'))
    labels = np.loadtxt(os.path.join(org_eval_dir, 'id_labels.txt'))
    org_preds = np.argmax(old_probs, axis=1)
    correct_classifications = np.asarray(org_preds == labels, dtype=np.int32)
    correct_classified_indices = np.argwhere(correct_classifications == 1)

    adv_success_rates = []
    for epsilon in epsilons:
        probs = np.loadtxt(os.path.join(attack_dir, f'e{epsilon}-attack', 'eval', 'probs.txt'))
        adv_success_rates.append(_get_mis_adv_success(probs, labels, correct_classified_indices))

    plot_epsilon_curve(epsilons, adv_success_rates, result_dir=result_dir)
    return adv_success_rates

def perform_epsilon_attack(model: nn.Module,
                           adv_dataset: Dataset,
                           correct_classified_indices,
                           batch_size=128,
                           device=None,
                           result_dir='./',
                           ood_adv_dataset=None,
                           valid_id_indices=[],
                           valid_ood_indices=[]):
    adv_success = None
    uncertainties = None
    if adv_dataset is not None:
        logits, probs, labels = eval_model_on_dataset(model, adv_dataset, batch_size, device=device)

        # Save model outputs
        eval_dir = os.path.join(result_dir, 'eval')
        os.makedirs(eval_dir)
        np.savetxt(os.path.join(eval_dir, 'labels.txt'), labels)
        np.savetxt(os.path.join(eval_dir, 'probs.txt'), probs)
        np.savetxt(os.path.join(eval_dir, 'logits.txt'), logits)

        # determine misclassifications under attack
        adv_indices = adv_dataset.get_adversarial_indices()
        adv_success = np.intersect1d(adv_indices, correct_classified_indices)
        adv_success = len(adv_success) / len(correct_classified_indices)

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
    ood_uncertainties = None
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
        if uncertainties is not None and ood_uncertainties is not None:
            OutOfDomainDetectionEvaluator(uncertainties, ood_uncertainties, ood_eval_dir).eval()
        id_success = None
        if adv_dataset is not None:
            # id_success
            adv_indices = adv_dataset.get_adversarial_indices()
            if len(valid_id_indices) > 0:
                id_success = np.intersect1d(adv_indices, valid_id_indices)
                id_success = len(id_success) / len(valid_id_indices)
            else:
                id_success = adv_indices
                id_success = len(id_success) / logits.shape[0] # over all id samples
        # ood_success
        ood_adv_indices = ood_adv_dataset.get_adversarial_indices()
        if len(valid_ood_indices) > 0:
            ood_success = np.intersect1d(ood_adv_indices, valid_ood_indices)
            ood_success = len(ood_success) / len(valid_ood_indices)
        else:
            ood_success = ood_adv_indices
            ood_success = len(ood_success) / ood_logits.shape[0]
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
    k = ckpt['model_params']['n_out'] # num_classes
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)

    # don't load in domain data when this option is true
    correct_classified_indices = []
    id_valid_indices = []
    if not args.attack_only_out_dist:
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
        # needed for confidence precision attack
        org_preds = np.argmax(probs, axis=1)
        alphas = np.exp(logits)
        alpha_0 = np.sum(alphas, axis=1)
        # determining correct classified indices for misclassify attacks
        correct_classifications = np.asarray(org_preds == labels, dtype=np.int32)
        correct_classified_indices = np.argwhere(correct_classifications == 1)
        if args.attack_criteria == 'precision_targeted':
            good_alpha0_indices = np.argwhere(alpha_0 >= CHOSEN_THRESHOLDS['precision'])
            correct_classified_indices = np.intersect1d(correct_classified_indices, good_alpha0_indices)
        if args.attack_criteria == 'alpha_k':
            good_alphak_indices = np.argwhere(alphas[(np.arange(0, alphas.shape[0]), labels)] >= CHOSEN_THRESHOLDS['alpha_k'])
            correct_classified_indices = np.intersect1d(correct_classified_indices, good_alphak_indices)
        # save correct classified indices
        np.savetxt(os.path.join(args.result_dir, 'id_correct-classfied-indices.txt'), correct_classified_indices)

        # determining valid indices for ood-detect attacks
        # if args.attack_criteria == 'precision':
        #     id_valid_indices = np.argwhere(alpha_0 >= CHOSEN_THRESHOLDS['precision'])
        np.savetxt(os.path.join(args.result_dir, 'id_valid-indices.txt'), id_valid_indices)
    # load ood dataset if atatck type is ood-detect.
    ood_dataset = None
    ood_valid_indices = []
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
        
        # evaluate and find samples with ideal precision
        # if args.attack_criteria == 'precision':
        #     ood_logits, ood_probs, _ = eval_model_on_dataset(model,
        #                                           dataset=ood_dataset,
        #                                           device=device,
        #                                           batch_size=args.batch_size)
        #     alphas = np.exp(ood_logits)
        #     alpha_0 = np.sum(alphas, axis=1)
        #     ood_valid_indices = np.argwhere(alpha_0 < CHOSEN_THRESHOLDS['precision'])
        np.savetxt(os.path.join(args.result_dir, 'ood_valid-indices.txt'), ood_valid_indices)

    # perform attacks on the same dataset, using different epsilon values.
    misclass_adv_success = []
    in_out_adv_success = []
    out_in_adv_success = []
    attack_criteria = ATTACK_CRITERIA_MAP[args.attack_criteria]
    ood_attack_criteria = OOD_ATTACK_CRITERIA_MAP[args.attack_criteria]
    success_detect_criteria = _get_adv_success_criteria(args.attack_type, args.attack_criteria, args.threshold)
    adv_success_detect_type = 'normal' if args.attack_type == 'misclassify' else 'ood-detect'
    for epsilon in args.epsilon:
        attack_folder = os.path.join(args.result_dir, f"e{epsilon}-attack")
        adv_dataset = None
        if not args.attack_only_out_dist:
            out_path = os.path.join(attack_folder, "adv-images")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # create adversarial dataset - in domain samples
            adv_dataset = AdversarialDataset(dataset, args.attack_strategy.lower(), model, epsilon,
                                            attack_criteria, args.norm,
                                            args.step_size, args.max_steps,
                                            args.batch_size, device=device,
                                            only_true_adversaries=False,
                                            use_org_img_as_fallback=False,
                                            adv_success_detect_type=adv_success_detect_type,
                                            success_detect_criteria=success_detect_criteria,
                                            targeted_attack=(args.attack_criteria == 'precision_targeted'),
                                            target_label=args.target_label)
            print(f"In domain adversarial dataset: {len(adv_dataset)}")
            # store in-domain adv indices
            np.savetxt(os.path.join(out_path, 'indices.txt'), adv_dataset.get_adversarial_indices())
            persist_image_dataset(data.Subset(adv_dataset, adv_dataset.get_adversarial_indices()), mean,
                                std, num_channels, out_path)

        #create adversarial dataset - out domain samples
        ood_adv_dataset = None
        if args.attack_type == 'ood-detect':
            ood_adv_dataset = AdversarialDataset(ood_dataset, args.attack_strategy.lower(),
                                                 model, epsilon, ood_attack_criteria,
                                                 args.norm, args.step_size, args.max_steps,
                                                 args.batch_size, device=device,
                                                 only_true_adversaries=False,
                                                 use_org_img_as_fallback=False,
                                                 adv_success_detect_type=adv_success_detect_type,
                                                 ood_dataset=True,
                                                 success_detect_criteria=success_detect_criteria)
            print(f"Out domain adversarial dataset: {len(ood_adv_dataset)}")
            ood_adv_folder = os.path.join(attack_folder, "adv-images-ood")
            if not os.path.exists(ood_adv_folder):
                os.makedirs(ood_adv_folder)
            np.savetxt(os.path.join(ood_adv_folder, 'indices.txt'), ood_adv_dataset.get_adversarial_indices())
            persist_image_dataset(data.Subset(ood_adv_dataset, ood_adv_dataset.get_adversarial_indices()),
                                  mean, std, num_channels, ood_adv_folder)

        # assess model perf using adv images
        adv_success = perform_epsilon_attack(model, adv_dataset, correct_classified_indices,
                                             batch_size=args.batch_size, device=device,
                                             result_dir=attack_folder,
                                             ood_adv_dataset=ood_adv_dataset,
                                             valid_id_indices=id_valid_indices,
                                             valid_ood_indices=ood_valid_indices)
        if adv_success['misclassify_success'] is not None:
            misclass_adv_success.append(adv_success['misclassify_success'])
        if adv_success['ood_detect_success'][0] is not None:
            in_out_adv_success.append(adv_success['ood_detect_success'][0])
        if adv_success['ood_detect_success'][1] is not None:
            out_in_adv_success.append(adv_success['ood_detect_success'][1])
        # log the success lists
        np.savetxt(os.path.join(args.result_dir, 'misclassify_success.txt'), misclass_adv_success)
        np.savetxt(os.path.join(args.result_dir, 'in-out_success.txt'), in_out_adv_success)
        np.savetxt(os.path.join(args.result_dir, 'out-in_success.txt'), out_in_adv_success)
    
    # plot the epsilon, adversarial success rate graph (line plot)
    if not args.attack_only_out_dist:
        plot_epsilon_curve(args.epsilon, misclass_adv_success, args.result_dir,
                        file_name='epsilon-curve_misclassify.png',
                        title=f'{args.attack_criteria} attack - {args.dataset} - Misclassification success')
    if args.attack_type == "ood-detect":
        if not args.attack_only_out_dist:
            plot_epsilon_curve(args.epsilon, in_out_adv_success, args.result_dir,
                            file_name='epsilon-curve_id.png',
                            title=f'{args.attack_criteria} attack - {args.dataset} + {args.ood_dataset} - in->out success')
        plot_epsilon_curve(args.epsilon, out_in_adv_success, args.result_dir,
                           file_name='epsilon-curve_ood.png',
                           title=f'{args.attack_criteria} attack - {args.dataset} + {args.ood_dataset} - out->in success')

if __name__ == '__main__':
    main()
