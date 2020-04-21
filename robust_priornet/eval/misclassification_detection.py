import os

import numpy as np
import torch.nn.functional as F

from .model_prediction_eval import ClassifierPredictionEvaluator
from .uncertainty import UncertaintyMeasuresEnum


class MisclassificationDetectionEvaluator:
    
    def __init__(self, probs, truth_labels, uncertainty_measures: dict, result_dir):
        self.probs = probs
        self.truth_labels = truth_labels
        self.uncertainty_measures = uncertainty_measures
        self.result_dir = result_dir

    def get_target_labels(self):
        """
        Generates target labels for the missclassification task as a binary classification
        task.
        label=1 is a missclassified sample.
        label=0 is a correctly classified sample.
        """
        preds = np.argmax(self.probs, axis=1)
        return np.asarray(preds != self.truth_labels, dtype=np.int32)

    def eval(self):
        """
        Uses the uncertainty measure as the model's predictive probability dist and evaluates
        PR and ROC curve for this misclassification task.
        """
        target_truth_labels = self.get_target_labels()
        for key in self.uncertainty_measures.keys():
            # negation needed for confidence, as confidence is indicator of label=0 samples
            # i.e for correct classified samples.
            # But we need scores for label=1 samples i.e misclassified samples
            # to be higher, so we negate.
            decision_fn_value = self.uncertainty_measures[key]
            if key == UncertaintyMeasuresEnum.CONFIDENCE:
                decision_fn_value *= -1.0

            aupr, auroc = ClassifierPredictionEvaluator.compute_pr_roc_curves(
                decision_fn_value, target_truth_labels, self.result_dir, key._value_)

            with open(os.path.join(self.result_dir, 'results.txt'), 'a') as f:
                f.write('AUPR using ' + key._value_ + ": " + str(np.round(aupr * 100.0, 1)) + '\n')
                f.write('AUROC using ' + key._value_ + ": " + str(np.round(auroc * 100.0, 1)) + '\n')
