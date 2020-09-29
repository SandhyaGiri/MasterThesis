import copy
import os

import numpy as np

from .model_prediction_eval import ClassifierPredictionEvaluator
from .uncertainty import UncertaintyMeasuresEnum


class OutOfDomainDetectionEvaluator:

    def __init__(self, id_uncertainty_measures, ood_uncertainty_measures, result_dir):
        self.id_uncertainty_measures = id_uncertainty_measures
        self.ood_uncertainty_measures = ood_uncertainty_measures
        self.result_dir = result_dir

    def get_target_labels(self):
        """
            Generates target labels for the ood detection task as a binary classification
            task.
            label=1 is a out of domain sample.
            label=0 is a in domain sample.
        """
        id_labels = np.zeros_like(self.id_uncertainty_measures[UncertaintyMeasuresEnum.CONFIDENCE])
        ood_labels = np.ones_like(self.ood_uncertainty_measures[UncertaintyMeasuresEnum.CONFIDENCE])
        return np.concatenate((id_labels, ood_labels), axis=0) # row wise concatenation

    def eval(self):
        """
            Uses the combined uncertainty measure (of id and ood) as the model's predictive
            probability dist and evaluates PR and ROC curve for this ood detection task.
        """
        target_truth_labels = self.get_target_labels()
        for key in self.id_uncertainty_measures.keys():
            # deep copy needed as we mutate confidence values later on
            decision_fn_value = np.concatenate((copy.deepcopy(self.id_uncertainty_measures[key]),
                                                copy.deepcopy(self.ood_uncertainty_measures[key])),
                                               axis=0)
            # negation needed for confidence, as confidence is indicator of label=0 samples
            # i.e for correct classified samples.
            # But we need scores for label=1 samples i.e misclassified samples
            # to be higher, so we negate.
            if key == UncertaintyMeasuresEnum.CONFIDENCE or key == UncertaintyMeasuresEnum.PRECISION:
                decision_fn_value *= -1.0

            aupr, auroc = ClassifierPredictionEvaluator.compute_pr_roc_curves(
                decision_fn_value, target_truth_labels, self.result_dir, key._value_)

            with open(os.path.join(self.result_dir, 'results.txt'), 'a') as f:
                f.write('AUPR using ' + key._value_ + ": " +
                        str(np.round(aupr * 100.0, 1)) + '\n')
                f.write('AUROC using ' + key._value_ + ": " +
                        str(np.round(auroc * 100.0, 1)) + '\n')
