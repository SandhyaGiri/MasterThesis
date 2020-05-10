import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from ..eval.uncertainty import (UncertaintyEvaluatorTorch,
                                UncertaintyMeasuresEnum)


class RandomizedSmoother:
    """
    Based on the paper "Certified Adversarial Robustness via Randomized Smoothing".

    Customized for a binary classifier for the ood-detect task on top of the standard
    dirichlet priornet classifier.
    """
    # to abstain from making a prediction/certification, this int will be returned.
    ABSTAIN = -1

    def __init__(self, base_classifier: nn.Module,
                 uncertainty_estimator: UncertaintyMeasuresEnum,
                 decision_threshold: float,
                 image_normalization_params: dict,
                 noise_std_dev: float):
        self.base_classifier = base_classifier
        self.uncertainty_estimator = uncertainty_estimator
        self.threhsold = decision_threshold
        self.image_normalization_params = image_normalization_params
        self.noise_std_dev = noise_std_dev

    def predict(self, image: torch.tensor, n: int, alpha: float, batch_size: int):
        pass

    def certify(self, image: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):
        # set model in eval mode
        self.base_classifier.eval()
        class_pred_counts = self._get_noisy_samples(image, n0, batch_size)
        # get the class with max counts
        most_prob_class_guess = class_pred_counts.argmax().item()
        # use many samples to estimate the probabilities
        class_pred_counts_strong = self._get_noisy_samples(image, n, batch_size)
        max_count = class_pred_counts_strong[most_prob_class_guess].item()
        # confidence interval for binomial dist (after n bernoulli trials,
        # with max_count number of times the guessed class was observed.)
        class_prob_lower_bound = proportion_confint(max_count, n, alpha=2 * alpha, method="beta")[0]
        if class_prob_lower_bound < 0.5:
            return self.ABSTAIN, 0.0
        else:
            robust_radius = self.noise_std_dev * norm.ppf(class_prob_lower_bound)
            return most_prob_class_guess, robust_radius

    def _eval_ood_task(self, inputs: torch.tensor):
        """
        normalize input images to range (-1,1) ->
        eval(base_classifer, inputs) -> predict uncertainty_measure based on logits
        -> use the threshold to make the prediction.
        """
        # normalize the input images
        mean = inputs.new_tensor(self.image_normalization_params['mean'])
        mean = mean.repeat(inputs.shape[0], inputs.shape[2],
                           inputs.shape[3], 1) # (batch_size, H, W, 3)
        mean = mean.permute(0,3,1,2)
        std = inputs.new_tensor(self.image_normalization_params['std'])
        std = std.repeat(inputs.shape[0], inputs.shape[2],
                         inputs.shape[3], 1) # (batch_size, H, W, 3)
        std = std.permute(0,3,1,2)
        normalized_inputs = (inputs - mean) / std
        # eval the inputs on base_classifier
        log_alphas = self.base_classifier(normalized_inputs)
        uncertainty_estimates = UncertaintyEvaluatorTorch(log_alphas).get_uncertainty(
            self.uncertainty_estimator, negate_confidence=True)
        # use threhold value to estimate prob(out dist), prob(in dist)
        probs = torch.sigmoid(uncertainty_estimates - self.threhsold)
        preds = torch.zeros_like(probs)
        preds[probs > 0.5] = 1
        return preds

    def _get_noisy_samples(self, image: torch.tensor, num_samples: int, batch_size: int):
        with torch.no_grad():
            counts = np.zeros(2, dtype=int)
            for _ in range(int(np.ceil(num_samples / batch_size))):
                this_batch_size = min(batch_size, num_samples)
                num_samples -= this_batch_size

                batch = image.repeat((this_batch_size, 1, 1, 1))
                # sample noise from normal dist
                noise = torch.randn_like(batch) * self.noise_std_dev
                predictions = self._eval_ood_task(batch + noise)
                preds_numpy = predictions.cpu().numpy()
                counts[0] += len(np.argwhere(preds_numpy == 0)) # in dist
                counts[1] += len(np.argwhere(preds_numpy == 1)) # out dist
            return counts
