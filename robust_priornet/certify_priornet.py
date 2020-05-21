import argparse
import os
import time

import numpy as np
import torch.utils.data as data

from .certification.randomized_smoothing import RandomizedSmoother
from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .eval.uncertainty import UncertaintyMeasuresEnum
from .utils.dataspliter import DataSpliter
from .utils.pytorch import choose_torch_device, load_model

parser = argparse.ArgumentParser(description='Identifies the robustness gurantee \
                    measures for the ood-detect task')
parser.add_argument('data_dir', type=str,
                    help='Path to the directory where data is saved')
parser.add_argument('result_dir', type=str,
                    help='Path of directory for saving the attack resultss.')
parser.add_argument('id_dataset', choices=DatasetEnum._member_map_.keys(),
                    help='Specify name of in domain dataset to be used for the certification.')
parser.add_argument('ood_dataset', choices=DatasetEnum._member_map_.keys(),
                    help='Specify name of out domain dataset to be used for the certification.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where the saved model exists.')
parser.add_argument('--train_dataset', action='store_true',
                    help='Whether to evaluate on the training data instead of test data')
parser.add_argument('--val_dataset', action='store_true',
                    help='Whether to evaluate on the val data instead of train/test dataset.')
parser.add_argument('--dataset_size_limit', type=int, default=None,
                    help='Specifies the number of samples to consider in the loaded datasets.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specifies the GPU ids to run the script on.')
# certification task type parameters
parser.add_argument('--certify_task', choices=['ood-detect', 'normal'],
                    default='ood-detect',
                    help='Specifies the task for which the model needs to be certified.')
# changing a base classifier into a binary classifer for ood-detect task
parser.add_argument('--uncertainty_measure', choices=UncertaintyMeasuresEnum._member_map_.keys(),
                    required=True,
                    help='Specifies the uncertainty measure to estimate for ood-detect task')
parser.add_argument('--uncertainty_measure_threshold', type=float, required=True,
                    help='Specifies the threshold to be used on uncertainty measure to'+
                    ' predict ood class or in-domain class')
# randomzied smoothing hyper params
parser.add_argument('--sigma', type=float, default=0.2,
                    help='Specifies the std deviation of the gaussian dist to be used'+
                    ' for robust radius certification.')
parser.add_argument('--n0', type=int, default=100,
                    help='small number of samples for estimating prob using MC')
parser.add_argument('--n', type=int, default=100000,
                    help='large number of samples for accurately estimating prob using MC')
parser.add_argument('--alpha', type=float, default=0.0001,
                    help='acceptable error estimate in MC estimate of prob dist')

def certify_classification(rand_smoother, args, device, id_dataset):
    radius_list = []
    pred_labels = []
    expected_labels = []
    for i, (image, label) in enumerate(id_dataset):
        image = image.to(device)
        expected_labels.append(label)
        start = time.time()
        pred, radius = rand_smoother.certify(image, args.n0, args.n, args.alpha, args.batch_size, 'normal')
        end = time.time()
        correct = int(pred == label)
        radius_list.append(radius)
        pred_labels.append(pred)
        with open(os.path.join(args.result_dir, 'results.txt'), 'a') as f:
            f.write(f'Index: {i}, Label: {label}, Prediction: {pred},'+
                    f' Correct: {correct}, TimeTaken: {np.round((end-start), 2)} sec\n')

    # save collected results
    np.savetxt(os.path.join(args.result_dir, 'in_radius.txt'), np.asarray(radius_list))
    np.savetxt(os.path.join(args.result_dir, 'in_preds.txt'), np.asarray(pred_labels))
    np.savetxt(os.path.join(args.result_dir, 'in_labels.txt'), np.asarray(expected_labels))

    # compute avg radius
    avg_radius_all = np.sum(radius_list) / len(id_dataset)
    pred_labels = np.asarray(pred_labels)
    radius_list = np.asarray(radius_list)
    expected_labels = np.asarray(expected_labels)
    avg_radius_correct = np.sum(radius_list[pred_labels == expected_labels]) / len(id_dataset)
    correct_preds = len(np.argwhere(pred_labels == expected_labels))
    total_preds = len(id_dataset)
    with open(os.path.join(args.result_dir, 'radius.txt'), 'a') as f:
        f.write(f'Avg robust radius: {np.round(avg_radius_all, 4)}\n')
        f.write(f'Avg robust radius (only correctly classified samples): {np.round(avg_radius_correct, 4)}\n')
        f.write(f'Accuracy: {correct_preds} correct out of {total_preds}\n')
        f.write(f'Certified accuracy: {np.round(correct_preds/total_preds,4)}')

def certify_ood_detect(rand_smoother, args, device, id_dataset, ood_dataset):
    dataset = data.ConcatDataset((id_dataset, ood_dataset))
    # in domain = label 0, out domain = label 1
    labels = np.concatenate((np.zeros(len(id_dataset), dtype=np.uint8),
                             np.ones(len(ood_dataset), dtype=np.uint8)), axis=0)
    in_radius = []
    out_radius = []
    in_pred_labels = []
    out_pred_labels = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        image = image.to(device)
        label = labels[i]
        start = time.time()
        pred, radius = rand_smoother.certify(image, args.n0, args.n, args.alpha, args.batch_size, 'ood-detect')
        end = time.time()
        correct = int(pred == label)
        if label == 0:
            in_radius.append(radius)
            in_pred_labels.append(pred)
        else:
            out_radius.append(radius)
            out_pred_labels.append(pred)
        with open(os.path.join(args.result_dir, 'results.txt'), 'a') as f:
            f.write(f'Index: {i}, Label: {label}, Prediction: {pred},'+
                    f' Correct: {correct}, TimeTaken: {np.round((end-start), 2)} sec\n')

    # save collected results
    np.savetxt(os.path.join(args.result_dir, 'in_radius.txt'), np.asarray(in_radius))
    np.savetxt(os.path.join(args.result_dir, 'out_radius.txt'), np.asarray(out_radius))
    np.savetxt(os.path.join(args.result_dir, 'in_preds.txt'), np.asarray(in_pred_labels))
    np.savetxt(os.path.join(args.result_dir, 'out_preds.txt'), np.asarray(out_pred_labels))
    
    # compute avg radius
    avg_in_radius_all = np.sum(in_radius) / len(id_dataset)
    in_pred_labels = np.asarray(in_pred_labels)
    in_radius = np.asarray(in_radius)
    avg_in_radius_correct = np.sum(in_radius[in_pred_labels == 0]) / len(np.argwhere(in_pred_labels == 0))
    avg_out_radius_all = np.sum(out_radius) / len(ood_dataset)
    out_pred_labels = np.asarray(out_pred_labels)
    out_radius = np.asarray(out_radius)
    avg_out_radius_correct = np.sum(out_radius[out_pred_labels == 1]) / len(np.argwhere(out_pred_labels == 1))
    avg_radius_correct = (np.sum(in_radius[in_pred_labels == 0]) +
                          np.sum(out_radius[out_pred_labels == 1])) / (
                              len(np.argwhere(in_pred_labels == 0)) +
                              len(np.argwhere(out_pred_labels == 1))
                          )
    correct_preds = len(np.argwhere(in_pred_labels == 0)) + len(np.argwhere(out_pred_labels == 1))
    total_preds = len(in_pred_labels) + len(out_pred_labels)
    with open(os.path.join(args.result_dir, 'radius.txt'), 'a') as f:
        f.write(f'Avg in-domain radius: {np.round(avg_in_radius_all, 4)}\n')
        f.write(f'Avg out-domain radius: {np.round(avg_out_radius_all, 4)}\n')
        f.write(f'Avg in-domain radius (only correctly classified samples): {np.round(avg_in_radius_correct, 4)}\n')
        f.write(f'Avg out-domain radius (only correctly classified samples): {np.round(avg_out_radius_correct, 4)}\n')
        f.write(f'Avg robust radius: {(np.sum(in_radius) + np.sum(out_radius)) / len(dataset)}\n')
        f.write(f'Avg robust radius ((only correctly classified samples)): {np.round(avg_radius_correct, 4)}\n')
        f.write(f'Accuracy: {correct_preds} correct out of {total_preds}\n')
        f.write(f'Certified accuracy: {np.round(correct_preds/total_preds,4)}')

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
    # for certification, load the dataset first without normalization
    trans.add_resize(ckpt['model_params']['n_in'])
    mean = (0.5,)
    std = (0.5,)
    num_channels = ckpt['model_params']['num_channels']
    num_classes =  ckpt['model_params']['n_out']
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()

    if args.val_dataset:
        _, id_dataset = vis.get_dataset(args.id_dataset,
                                        args.data_dir,
                                        trans.get_transforms(),
                                        None,
                                        'train',
                                        val_ratio=0.1)
    else:
        id_dataset = vis.get_dataset(args.id_dataset,
                                     args.data_dir,
                                     trans.get_transforms(),
                                     None,
                                     'train' if args.train_dataset else 'test')
    if args.dataset_size_limit is not None:
        id_dataset = DataSpliter.reduceSize(id_dataset, args.dataset_size_limit)
    print(f"In domain dataset: {len(id_dataset)}")

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

    rand_smoother = RandomizedSmoother(model,
                                       num_classes,
                                       UncertaintyMeasuresEnum.get_enum(args.uncertainty_measure),
                                       args.uncertainty_measure_threshold,
                                       {'mean': mean, 'std': std},
                                       args.sigma)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.certify_task == 'normal':
        certify_classification(rand_smoother, args, device, id_dataset)
    elif args.certify_task == 'ood-detect':
        certify_ood_detect(rand_smoother, args, device, id_dataset, ood_dataset)
if __name__ == "__main__":
    main()
