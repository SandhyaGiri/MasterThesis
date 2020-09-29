import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from ..utils.visualizer import plot_curve
from .uncertainty import UncertaintyEvaluator, UncertaintyMeasuresEnum


class ClassifierPredictionEvaluator:
    """
        Provides methods to evaluate a binary classifier model such as computing
        PR and ROC curves, accuracy etc.
    """
    @staticmethod
    def compute_pr_curve(decision_fn_value, truth_labels, result_dir, file_name):
        """
            Also saves the result in the result_dir provided.
            params
            ------
                decision_fn_value: estimated probability or any other measure used
                                    for decision making in binary classification task.
                                    Usually the model's prediction probability.
                truth_labels: ground truth labels for the given task.
        """
        try:
            precision, recall, thresholds = precision_recall_curve(truth_labels, decision_fn_value)
            aupr = auc(recall, precision)

            np.savetxt(os.path.join(result_dir, file_name + '_recall.txt'), recall)
            np.savetxt(os.path.join(result_dir, file_name + '_precision.txt'), precision)
            np.savetxt(os.path.join(result_dir, file_name + '_pr_thresholds.txt'), thresholds)

            _, axes = plt.subplots(nrows=1, ncols=1)
            plot_curve(recall, precision, axes, x_label='Recall',
                    y_label='Precision',
                    x_lim=(0.0, 1.0), y_lim=(0.0, 1.0),
                    axis_spine_visibility_config=['right', 'top'])
            plt.savefig(os.path.join(result_dir, file_name + '_PR_Curve.png'))
            plt.close()
            return np.round(aupr, 4)
        except ValueError:
            print("PR curve couldn't be plotted because of an exception (ValueError).")
        return 0

    @staticmethod
    def compute_roc_curve(decision_fn_value, truth_labels, result_dir, file_name):
        """
            Also saves the result in the result_dir provided.
            params
            ------
                decision_fn_value: estimated probability or any other measure used
                                    for decision making in binary classification task.
                                    Usually the model's prediction probability.
                truth_labels: ground truth labels for the given task.
        """
        try:
            fpr, tpr, thresholds = roc_curve(truth_labels, decision_fn_value)
            roc_auc = roc_auc_score(truth_labels, decision_fn_value)

            np.savetxt(os.path.join(result_dir, file_name + '_tpr.txt'), tpr)
            np.savetxt(os.path.join(result_dir, file_name + '_fpr.txt'), fpr)
            np.savetxt(os.path.join(result_dir, file_name + '_roc_thresholds.txt'), thresholds)

            _, axes = plt.subplots(nrows=1, ncols=1)
            plot_curve(fpr, tpr, axes, x_label='False Postive Rate (FPR)',
                    y_label='True Positive Rate (TPR)',
                    x_lim=(0.0, 1.0), y_lim=(0.0, 1.0),
                    axis_spine_visibility_config=['right', 'top'])
            plt.savefig(os.path.join(result_dir, file_name + '_ROC_Curve.png'))
            plt.close()
            return np.round(roc_auc, 4)
        except ValueError:
            print("ROC curve couldn't be plotted because of an exception (ValueError).")
        return 0

    @classmethod
    def compute_pr_roc_curves(cls, decision_fn_value, truth_labels, result_dir, file_name):
        """
            Computes both PR and ROC curves for binary classification task, and returns the
            corresponding area under the curves.
            Also saves the result in the result_dir provided.
        """
        aupr = cls.compute_pr_curve(decision_fn_value, truth_labels, result_dir, file_name)
        auroc = cls.compute_roc_curve(decision_fn_value, truth_labels, result_dir, file_name)
        return aupr, auroc

    @staticmethod
    def compute_accuracy(y_probs, y_true, device=None, weights=None):
        """
            Calculates accuracy of model's predictsions, given the output probabilities
            and the truth labels.
        """
        if isinstance(y_probs, np.ndarray):
            y_probs = torch.tensor(y_probs)
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true)

        if weights is None:
            if device is None:
                accuracy = torch.mean((torch.argmax(y_probs, dim=1) == y_true)
                                      .to(dtype=torch.float64))
            else:
                accuracy = torch.mean((torch.argmax(y_probs, dim=1) == y_true)
                                      .to(device, torch.float64))
        else:
            if device is None:
                weights.to(dtype=torch.float64)
                accuracy = torch.mean(
                    weights * (torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float64))
            else:
                weights.to(device=device, dtype=torch.float64)
                accuracy = torch.mean(weights * (torch.argmax(y_probs, dim=1) == y_true)
                                      .to(device=device, dtype=torch.float64))
        return accuracy

    @staticmethod
    def compute_nll(y_probs, y_true, device=None, epsilon=1e-10):
        """
            Returns the negative log likelihood value between model's predictions
            and the ground truth labels.
        """
        if isinstance(y_probs, np.ndarray):
            y_probs = torch.tensor(y_probs)
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true)
        probs_at_truth_labels = y_probs[:, torch.squeeze(y_true)]
        return -1 * torch.mean(torch.log(probs_at_truth_labels + epsilon))

    @staticmethod
    def compute_confusion_matrix_entries(decision_fn_value, y_true, threshold=None):
        """
            Computes the number of true positives, true negatives, false positives and
            false negatives from the given decision_fn_values.

            params
            ------
                decision_fn_value: This could be either prob
                    outputted by the model or any other value used for making classification decision.
                y_true: ground truth labels either 0 or 1.
                threshold: If threshold value is given, decision_fn_values above this threshold
                    are classified as label=1, and those below are classified as label=0.
                    Otherwise, threshold value is chosen (max-min)/2 from the decision_fn_values.

            returns
            -------
                tn, fp, fn, tp (in the same order)
        """
        if threshold is None:
            threshold = (max(decision_fn_value) - min(decision_fn_value)) / 2.0

        y_preds = np.zeros_like(y_true)
        y_preds[np.round(decision_fn_value, 4) >= np.round(threshold, 4)] = 1

        return confusion_matrix(y_true, y_preds).ravel()

class PriorNetClassifierPredictionEvaluator:
    """
        Provides methods to evaluate a priornet classifier model such as computing accuracy etc.
    """
    @staticmethod
    def compute_in_accuracy_from_uncertainty_measures(y_probs, y_true, logits, uncertainty_measures: list(UncertaintyMeasuresEnum),
                                                      thresholds: list):
        num_samples = y_probs.shape[0]
        correct_indices = np.argwhere(np.argmax(y_probs, axis=1) == y_true)
        wrong_indices = np.argwhere(np.argmax(y_probs, axis=1) != y_true)
        valid_indices = None
        for i in range(len(uncertainty_measures)):
            uncertainty_values = UncertaintyEvaluator(logits).get_uncertainty(uncertainty_measures[i],
                                                                            negate_confidence=True).reshape((logits.shape[0],))
            good_indices = np.argwhere(np.round(uncertainty_values, 4) < np.round(thresholds[i], 4))
            if valid_indices is None:
                valid_indices = good_indices
            else:
                valid_indices = np.intersect1d(valid_indices, good_indices)
        # correctly classified samples (with valid criteria)
        correct = np.intersect1d(correct_indices, valid_indices)
        # wrongly classified samples (with valid criteria)
        wrong = np.intersect1d(wrong_indices, valid_indices)
        # rejected samples (invalid criteria)
        reject = np.setdiff1d(np.arange(0, num_samples), valid_indices)

        # normalize the numbers to get percentage values
        correct = len(correct)/num_samples
        wrong = len(wrong)/num_samples
        reject = len(reject)/num_samples
        return (correct, wrong, reject)
    
    @staticmethod
    def compute_out_accuracy_from_uncertainty_measures(y_probs,
                                                       logits,
                                                       uncertainty_measures: list(UncertaintyMeasuresEnum),
                                                       thresholds: list):
        num_samples = y_probs.shape[0]
        valid_indices = None
        preds = np.zeros((logits.shape[0], ), dtype=np.int)
        for i in range(len(uncertainty_measures)):
            uncertainty_values = UncertaintyEvaluator(logits).get_uncertainty(uncertainty_measures[i],
                                                                            negate_confidence=True).reshape((logits.shape[0],))
            good_indices = np.argwhere(np.round(uncertainty_values, 4) >= np.round(thresholds[i], 4))
            if valid_indices is None:
                valid_indices = good_indices
            else:
                valid_indices = np.intersect1d(valid_indices, good_indices)
        preds[valid_indices] = 1 # there are true ood samples
        reject = len(np.argwhere(preds == 1))/num_samples
        problem = len(np.argwhere(preds == 0))/num_samples
        return (problem, reject)
