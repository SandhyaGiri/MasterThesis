import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm, binom_test

from ..eval.uncertainty import (UncertaintyEvaluatorTorch,
                                UncertaintyMeasuresEnum)


class RandomizedSmoother:
    """
    Based on the paper "Certified Adversarial Robustness via Randomized Smoothing".

    Customized for a binary classifier for the ood-detect task on top of the standard
    dirichlet priornet classifier.

    Performs both in domain data classification for classifying image correctly
    into one of the target classess (multi-class) and in-out classification (binary).
    Tasks: 'normal', 'ood-detect'
    
    IMP: Cannot be used to train the base_classifier, model is set to eval mode before
    computing the predictions.
    """
    # to abstain from making a prediction/certification, this int will be returned.
    ABSTAIN = -1

    def __init__(self, base_classifier: nn.Module,
                 num_classes: int,
                 uncertainty_estimator: UncertaintyMeasuresEnum,
                 decision_threshold: float,
                 image_normalization_params: dict,
                 noise_std_dev: float):
        # num_classes is base classifier's last layer
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.uncertainty_estimator = uncertainty_estimator
        self.threshold = decision_threshold
        self.image_normalization_params = image_normalization_params
        self.noise_std_dev = noise_std_dev

    def predict(self, image: torch.tensor, n: int, alpha: float, batch_size: int):
        # set model in eval mode
        self.base_classifier.eval()
        class_pred_counts = self._get_noisy_samples(image, n, batch_size, 'normal')
        # get top two classes
        top2 = class_pred_counts.argsort()[::-1][:2]
        count1 = class_pred_counts[top2[0]]
        count2 = class_pred_counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return self.ABSTAIN
        else:
            return top2[0]

    def certify(self, image: torch.tensor, n0: int, n: int, alpha: float, batch_size: int,
                task_type: str):
        assert task_type in ['normal', 'ood-detect']
        # set model in eval mode
        self.base_classifier.eval()
        class_pred_counts = self._get_noisy_samples(image, n0, batch_size, task_type)
        # get the class with max counts
        most_prob_class_guess = class_pred_counts.argmax().item()
        # use many samples to estimate the probabilities
        class_pred_counts_strong = self._get_noisy_samples(image, n, batch_size, task_type)
        max_count = class_pred_counts_strong[most_prob_class_guess].item()
        # confidence interval for binomial dist (after n bernoulli trials,
        # with max_count number of times the guessed class was observed.)
        class_prob_lower_bound = proportion_confint(max_count, n, alpha=2 * alpha, method="beta")[0]
        if class_prob_lower_bound < 0.5:
            return self.ABSTAIN, 0.0
        else:
            robust_radius = self.noise_std_dev * norm.ppf(class_prob_lower_bound)
            return most_prob_class_guess, robust_radius

    def _normalize_image(self, inputs: torch.tensor):
        mean = inputs.new_tensor(self.image_normalization_params['mean'])
        mean = mean.repeat(inputs.shape[0], inputs.shape[2],
                           inputs.shape[3], 1) # (batch_size, H, W, 3)
        mean = mean.permute(0, 3, 1, 2)
        std = inputs.new_tensor(self.image_normalization_params['std'])
        std = std.repeat(inputs.shape[0], inputs.shape[2],
                         inputs.shape[3], 1) # (batch_size, H, W, 3)
        std = std.permute(0, 3, 1, 2)
        normalized_inputs = (inputs - mean) / std
        return normalized_inputs

    def _eval_ood_task(self, inputs: torch.tensor):
        """
        normalize input images to range (-1,1) ->
        eval(base_classifer, inputs) -> predict uncertainty_measure based on logits
        -> use the threshold to make the prediction.
        """
        # normalize the input images
        normalized_inputs = self._normalize_image(inputs)
        # eval the inputs on base_classifier
        log_alphas = self.base_classifier(normalized_inputs)
        uncertainty_estimates = UncertaintyEvaluatorTorch(log_alphas).get_uncertainty(
            self.uncertainty_estimator, negate_confidence=True)
        # use threshold value to estimate the correct label, label=1 for ood, label=0 for id
        uncertainty_estimates_numpy = uncertainty_estimates.detach().cpu().numpy()
        preds = np.zeros_like(uncertainty_estimates_numpy)
        preds[np.round(uncertainty_estimates_numpy, 4) >= np.round(self.threshold, 4)] = 1
        return torch.tensor(preds).squeeze()

    def _eval_normal_classification_task(self, inputs: torch.tensor):
        """
        normalize input images to range (-1,1) ->
        eval(base_classifer, inputs) -> predict class label based on max prob
        """
        # normalize the input images
        normalized_inputs = self._normalize_image(inputs)
        # eval the inputs on base_classifier
        log_alphas = self.base_classifier(normalized_inputs)
        probs = torch.nn.functional.softmax(log_alphas, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def _update_count(self, counts: np.ndarray, preds: np.ndarray):
        for i in range(len(counts)):
            counts[i] += len(np.argwhere(preds == i))

    def _get_noisy_samples(self, image: torch.tensor, num_samples: int, batch_size: int,
                           task_type: str):
        num_classes = 2 if task_type == 'ood-detect' else self.num_classes
        with torch.no_grad():
            counts = np.zeros(num_classes, dtype=int)
            for _ in range(int(np.ceil(num_samples / batch_size))):
                this_batch_size = min(batch_size, num_samples)
                num_samples -= this_batch_size

                batch = image.repeat((this_batch_size, 1, 1, 1))
                # sample noise from normal dist
                noise = torch.randn_like(batch) * self.noise_std_dev
                if task_type == 'ood-detect':
                    predictions = self._eval_ood_task(batch + noise)
                elif task_type == 'normal':
                    predictions = self._eval_normal_classification_task(batch + noise)
                preds_numpy = predictions.cpu().numpy()
                self._update_count(counts, preds_numpy)
            return counts
