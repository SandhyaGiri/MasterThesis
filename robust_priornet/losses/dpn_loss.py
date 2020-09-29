"""
This module contains KL div loss between two dirichlets and a weighted loss to compute the
total loss of a prior network.
"""

from typing import Optional, Iterable
import torch
import torch.nn.functional as F
import numpy as np

class PriorNetWeightedLoss:
    """
        Returns a mixed or linear combination of the losses provided, with weights
        taken directly from the weights.
        The losses provided should be of type : KLDivDirchletDistLoss.
        The final loss is also scaled down by the maximum target_precisions in the losses.
    """
    def __init__(self, losses, weights: Optional[Iterable[float]]):
        assert isinstance(losses, (list, tuple))
        assert isinstance(weights, (list, tuple, np.ndarray))
        assert len(losses) == len(weights)

        self.losses = losses
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.] * len(self.losses)

    def __call__(self, logits, labels):
        return self.forward(logits, labels)

    def forward(self, logits, labels):
        total_loss = []
        target_precision = 0.0
        for i, loss in enumerate(self.losses):
            if loss.target_precision > target_precision:
                target_precision = loss.target_precision
            weighted_loss = (loss(logits[i], labels[i])
                             * self.weights[i])
            total_loss.append(weighted_loss)
        total_loss = torch.stack(total_loss, dim=0)
        # Normalize by target concentration, so that loss  magnitude
        # is constant wrt lr and other losses
        return torch.sum(total_loss) / target_precision


class KLDivDirchletDistLoss:
    """
        Computes KL divergence between two dirichlet distributions.
        Can be applied to any DNN model which returns logits.

        Note that given the logits of a DNN model, exp of this yields the concentration(alpha)
        of the dirichlet distribution outputted by the DNN. From the target labels provided,
        a "desired/expected" target dirichlet distribution's concentration parameters can
        be constructed.
        Ex:
            when target labels are provided, a peaky dirichlet with major concentration on
                target class is expected.
            when target labels are not provided, a flat dirichlet with uniform alphas
                (low value alpha=1) is expected.

        Loss value is then just the KL diveregence between the DNN's dirichlet and the
        expected or target dirichlet distribution.
    """

    def __init__(self, target_precision=1e3, smoothing_factor=1e-2, reverse_KL=False):
        self.target_precision = target_precision
        self.smooothing_factor = smoothing_factor
        self.reverse_KL = reverse_KL

    def __call__(self, logits, labels, reduction='mean'):
        # logits = logits - torch.max(logits, dim=0)[0]
        alphas = torch.exp(logits)
        return self.forward(alphas, labels, reduction=reduction)

    def forward(self, alphas, labels, reduction='mean'):
        loss = self.compute_loss(alphas, labels)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError
    
    @staticmethod
    def compute_kl_div_dirichlets(target_mean, mean, target_precision, precision, epsilon=1e-8):
        """
            Computes KL divergence
            KL( Dir(alpha = target_precision * target_mean) || Dir(beta = precision * mean)
        """
        precision_term = (torch.lgamma(target_precision + epsilon) -
                          torch.lgamma(precision + epsilon))

        alphas = target_precision * target_mean
        betas = precision * mean
        concentration_term = torch.sum(torch.lgamma(betas + epsilon) -
                                       torch.lgamma(alphas + epsilon) +
                                       ((alphas - betas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(target_precision +
                                                                          epsilon)
                                                            )
                                        ), dim=1, keepdim=True)
        kl_div = torch.squeeze(precision_term + concentration_term)
        return kl_div

    def compute_loss(self, alphas, labels: Optional[torch.tensor] = None):
        k = alphas.shape[1] # num_classes
        precision = torch.sum(alphas, dim=1, keepdim=True)
        mean = alphas / precision

        if labels is None:
            # ood sample, set all alphas to 1 to get a flat simplex
            # or precision = num_classes, mean = 1/precision
            target_alphas = torch.ones_like(alphas)
            target_precision = torch.sum(target_alphas, dim=1, keepdim=True)
            target_mean = target_alphas / target_precision
        else:
            # in domain sample
            # this is the epsilon smoothing param in paper
            target_mean = torch.ones_like(alphas) * self.smooothing_factor
            target_mean = torch.clone(target_mean).scatter_(1, labels[:, None],
                                                            1-(k-1) * self.smooothing_factor)
            target_precision = alphas.new_ones((alphas.shape[0], 1)) * self.target_precision
        if self.reverse_KL:
            loss = self.compute_kl_div_dirichlets(mean, target_mean, precision, target_precision)
        else:
            loss = self.compute_kl_div_dirichlets(target_mean, mean, target_precision, precision)
        return loss

class PriorNetWeightedAdvLoss:
    """
        Returns a mixed or linear combination of the losses provided, with weights
        taken directly from the weights.

        The losses provided should be of type : TargetedKLDivDirchletDistLoss.

        The final loss is also scaled down by the maximum target_precisions in the losses.
    """
    def __init__(self, losses, weights: Optional[Iterable[float]]):
        assert isinstance(losses, (list, tuple))
        assert isinstance(weights, (list, tuple, np.ndarray))
        assert len(losses) == len(weights)

        self.losses = losses
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.] * len(self.losses)

    def __call__(self, logits, target_means, target_precisions):
        return self.forward(logits, target_means, target_precisions)

    def forward(self, logits, target_means, target_precisions):
        total_loss = []
        target_precision = 1.0
        for i, loss in enumerate(self.losses):
            # avg_target_precision = torch.mean(target_precisions[i])
            max_target_precision = torch.max(target_precisions[i]).item()
            if max_target_precision > target_precision:
                target_precision = max_target_precision
            weighted_loss = (loss(logits[i], target_means[i], target_precisions[i])
                             * self.weights[i])
            total_loss.append(weighted_loss)
        total_loss = torch.stack(total_loss, dim=0)
        # Normalize by target concentration, so that loss  magnitude
        # is constant wrt lr and other losses
        return torch.sum(total_loss) / target_precision

class TargetedKLDivDirchletDistLoss:
    """
        Similar to KLDivDirchletDistLoss, but a specific target dirichlet dist can be provided
        instead of the target labels (from which a target dirichlet distribution is determined).
    """
    def __init__(self, target_precision=1e3, smoothing_factor=1e-2, reverse_KL=False):
        self.target_precision = target_precision
        self.smooothing_factor = smoothing_factor
        self.reverse_KL = reverse_KL

    def __call__(self, logits, target_mean, target_precision, reduction='mean'):
        # logits = logits - torch.max(logits, dim=0)[0]
        alphas = torch.exp(logits)
        return self.forward(alphas,
                            target_mean,
                            target_precision,
                            reduction=reduction)

    def forward(self, alphas, target_mean, target_precision, reduction='mean'):
        loss = self.compute_loss(alphas,
                                 target_mean,
                                 target_precision)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError

    @staticmethod
    def compute_kl_div_dirichlets(target_mean, mean, target_precision, precision, epsilon=1e-8):
        """
            Computes KL divergence
            KL( Dir(alpha = target_precision * target_mean) || Dir(beta = precision * mean)
        """
        precision_term = (torch.lgamma(target_precision + epsilon) -
                          torch.lgamma(precision + epsilon))

        alphas = target_precision * target_mean
        betas = precision * mean
        concentration_term = torch.sum(torch.lgamma(betas + epsilon) -
                                       torch.lgamma(alphas + epsilon) +
                                       ((alphas - betas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(target_precision +
                                                                          epsilon)
                                                            )
                                        ), dim=1, keepdim=True)
        kl_div = torch.squeeze(precision_term + concentration_term)
        return kl_div

    def compute_loss(self, alphas, target_mean, target_precision):
        precision = torch.sum(alphas, dim=1, keepdim=True)
        mean = alphas / precision
        target_mean = target_mean.to(mean.device)
        target_precision = target_precision.to(precision.device)
        if self.reverse_KL:
            loss = self.compute_kl_div_dirichlets(mean, target_mean, precision, target_precision)
        else:
            loss = self.compute_kl_div_dirichlets(target_mean, mean, target_precision, precision)
        return loss