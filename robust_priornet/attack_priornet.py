import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset

from .datasets.adversarial_dataset import AdversarialDataset
from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .eval.misclassification_detection import \
    MisclassificationDetectionEvaluator
from .eval.model_prediction_eval import ClassifierPredictionEvaluator
from .eval.out_of_domain_detection import OutOfDomainDetectionEvaluator
from .eval.uncertainty import UncertaintyEvaluator
from .losses.attack_loss import AttackCriteria
from .utils.persistence import persist_image_dataset
from .utils.pytorch import (choose_torch_device, eval_model_on_dataset,
                            load_model)
from .utils.dataspliter import DataSpliter

matplotlib.use('agg')

ATTACK_CRITERIA_MAP = {
    'confidence': AttackCriteria.confidence_loss,
    'diff_entropy': AttackCriteria.differential_entropy_loss
}

parser = argparse.ArgumentParser(description='Constructs an FGSM/PGD attack on test images and \
                    reports the results as epsilon vs adversarial success rate.')
parser.add_argument('data_dir', type=str,
                    help='Path to the directory where data is saved')
parser.add_argument('dataset', choices=DatasetEnum._member_map_.keys(),
                    help='Specify name of dataset to be used for creating adversarial data.')
parser.add_argument('result_dir', type=str,
                    help='Path of directory for saving the attack resultss.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--epsilon', '--list', nargs='+', type=float,
                    help='Strength perturbation in range of 0 to 1, ex: 0.25', required=True)
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--train_dataset', action='store_true',
                    help='Whether to evaluate on the training data instead of test data')
parser.add_argument('--attack_type', type=str, choices=['FGSM', 'PGD'], default='FGSM',
                    help='Choose the type of attack to be performed.')
parser.add_argument('--attack_criteria', type=str, choices=ATTACK_CRITERIA_MAP.keys(),
                    required=True,
                    help='Indicates which loss function to use to compute attack gradient.')
parser.add_argument('--norm', type=str, choices=['inf', '2'], default='inf',
                    help='The type of norm ball to restrict the adversarial samples to.' +
                    'Needed only for PGD attack.')
parser.add_argument('--step_size', type=float, default=0.4,
                    help='The size of the gradient update step, used during each iteration'+
                    ' in PGD attack.')
parser.add_argument('--max_steps', type=int, default=10,
                    help='The number of the gradient update steps, to be performed for PGD attack.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specifies the GPU ids to run the script on.')
parser.add_argument('--ood_eval', action='store_true',
                    help='Use this flag to indicate if adversarial images should be used for' +
                    ' ood-detect task along with another ood_dataset.')
parser.add_argument('--ood_dataset', choices=DatasetEnum._member_map_.keys(),
                    help='Dataset to be used for ood-detect evaluation. This dataset will not be' +
                    ' perturbed to get adversarial dataset, just used for eval against an' +
                    ' adversarial in domain dataset.')

def perform_epsilon_attack(model: nn.Module, adv_dataset: Dataset, correct_classified_indices,
                           batch_size=128, device=None, result_dir='./', ood_dataset=None):
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
    if ood_dataset is not None:
        ood_logits, _, _ = eval_model_on_dataset(model,
                                                 ood_dataset,
                                                 batch_size,
                                                 device=device)
        ood_uncertainties = UncertaintyEvaluator(ood_logits).get_all_uncertainties()
        ood_eval_dir = os.path.join(result_dir, 'ood_eval')
        os.makedirs(ood_eval_dir)
        for key in uncertainties.keys():
            np.savetxt(os.path.join(ood_eval_dir, key._value_ + '.txt'), uncertainties[key])
        OutOfDomainDetectionEvaluator(uncertainties, ood_uncertainties, ood_eval_dir).eval()

    # return aversarial successes
    return adv_success

def main():
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise ValueError("Model doesn't exist!")

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
    num_channels = ckpt['model_params']['num_channels']
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)

    dataset = vis.get_dataset(args.dataset,
                              args.data_dir,
                              trans.get_transforms(),
                              None,
                              'train' if args.train_dataset else 'test')

    # dataset = DataSpliter.reduceSize(dataset, 5000)
    org_dataset_folder = os.path.join(args.result_dir, "org-images")
    if not os.path.exists(org_dataset_folder):
        os.makedirs(org_dataset_folder)
    persist_image_dataset(dataset, mean, std, num_channels, org_dataset_folder)

    # perform original evaluation on the model using unperturbed images
    logits, probs, labels = eval_model_on_dataset(model,
                                                  dataset=dataset,
                                                  device=device,
                                                  batch_size=args.batch_size)
    # determine correct classifications without attack (original non perturbed images)
    org_preds = np.argmax(probs, axis=1)
    correct_classifications = np.asarray(org_preds == labels, dtype=np.int32)
    correct_classified_indices = np.argwhere(correct_classifications == 1)

    # load ood dataset if ood-detect eval also needs to be done during adversarial attacks.
    ood_dataset = None
    if args.ood_eval and args.ood_dataset is not None:
        ood_dataset = vis.get_dataset(args.ood_dataset,
                                      args.data_dir,
                                      trans.get_transforms(),
                                      None,
                                      'train' if args.train_dataset else 'test')
        # ood_dataset = DataSpliter.reduceSize(ood_dataset, 5000)

    # perform attacks on the same dataset, using different epsilon values.
    adv_success_rates = []
    attack_criteria = ATTACK_CRITERIA_MAP[args.attack_criteria]
    for epsilon in args.epsilon:
        attack_folder = os.path.join(args.result_dir, f"e{epsilon}-attack")
        out_path = os.path.join(attack_folder, "adv-images")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # create adversarial dataset
        adv_dataset = AdversarialDataset(dataset, args.attack_type.lower(), model, epsilon,
                                         attack_criteria, args.norm,
                                         args.step_size, args.max_steps,
                                         args.batch_size, device=device)
        persist_image_dataset(adv_dataset, mean, std, num_channels, out_path)

        # assess model perf using adv images
        adv_success = perform_epsilon_attack(model, adv_dataset, correct_classified_indices,
                                             batch_size=args.batch_size, device=device,
                                             result_dir=attack_folder, ood_dataset=ood_dataset)
        adv_success_rates.append(adv_success / len(correct_classified_indices))

    # plot the epsilon, adversarial success rate graph (line plot)
    plt.figure(figsize=(5, 5))
    plt.plot(args.epsilon, adv_success_rates, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(np.min(args.epsilon), np.max(args.epsilon), step=0.1))
    plt.title("Adversarial Success Rate vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial Success Rate")
    plt.savefig(os.path.join(args.result_dir, "epsilon-curve.png"))

if __name__ == '__main__':
    main()
