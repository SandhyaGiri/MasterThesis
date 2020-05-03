import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from ..utils.visualizer import plot_curve


class ClassifierPredictionEvaluator:
    """
    Provides methods to evaluate a binary classifier model such as computing
    PR and ROC curves, accuracy etc.
    """
    @staticmethod
    def compute_pr_curve(decision_fn_value, truth_labels, result_dir, file_name):
        """
        Also saves the result in the result_dir provided.
        Args:
            decision_fn_value: estimated probability or any other measure used
                                for decision making in binary classification task.
                                Usually the model's prediction probability.
            truth_labels: ground truth labels for the given task.
        """
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

    @staticmethod
    def compute_roc_curve(decision_fn_value, truth_labels, result_dir, file_name):
        """
        Also saves the result in the result_dir provided.
        Args:
            decision_fn_value: estimated probability or any other measure used
                                for decision making in binary classification task.
                                Usually the model's prediction probability.
            truth_labels: ground truth labels for the given task.
        """
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

        Args:
            decision_fn_value: This could be either prob
                outputted by the model or any other value used for making classification decision.
            y_true: ground truth labels either 0 or 1.
            threshold: If threshold value is given, decision_fn_values above this threshold
                are classified as label=1, and those below are classified as label=0.
                Otherwise, threshold value is chosen (max-min)/2 from the decision_fn_values.
        
        Returns:
            tn, fp, fn, tp (in the same order)
        """
        if threshold is None:
            threshold = (max(decision_fn_value) - min(decision_fn_value)) / 2.0

        y_preds = np.zeros_like(y_true)
        y_preds[decision_fn_value >= threshold] = 1

        return confusion_matrix(y_true, y_preds).ravel()
