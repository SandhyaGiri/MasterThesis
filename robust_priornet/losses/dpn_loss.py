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
    Returns a mixed or linear combination of the losses provided, with weights taken directly from the weights.
    The losses provided should be of type : DirichletKLLoss.
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
        # Normalize by target concentration, so that loss  magnitude is constant wrt lr and other losses
        return torch.sum(total_loss) / target_precision


class KLDivDirchletDistLoss:
    """
    Computes KL divergence between two dirichlet distributions. Can be applied to any DNN model which returns logits.
    
    Note that given the logits of a DNN model, exp of this yields the concentration(alpha) of the dirichlet distribution
    outputted by the DNN. From the target labels provided, a "desired/expected" target dirichlet distribution's concentration parameters
    are constructed. 
    Ex: 
        when target labels are provided, a peaky dirichlet with major concentration on target class is expected.
        when target labels are not provided, a flat dirichlet with uniform alphas (low value alpha=1) is expected.

    Loss value is then just the KL diveregence between the DNN's dirichlet and the expected dirichlet distribution.
    """

    def __init__(self, target_precision=1e3, smoothing_factor=1e-2):
        self.target_precision = torch.tensor(target_precision, dtype=torch.float32)
        self.smooothing_factor = smoothing_factor

    def __call__(self, logits, labels, reduction='mean'):
        logits = logits - torch.max(logits, dim=0)[0]
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
        Computes KL divergence KL( Dir(alpha = target_precision * target_mean) || Dir(beta = precision * mean)
        """
        precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)

        alphas = target_precision * target_mean
        betas = precision * mean
        concentration_term = torch.sum(torch.lgamma(betas + epsilon) - torch.lgamma(alphas + epsilon) +
                                        ((alphas - betas) * (torch.digamma(alphas + epsilon) - 
                                                            torch.digamma(target_precision + epsilon))), dim=1, keepdim=True)
        kl_div = torch.squeeze(precision_term + concentration_term)
        return kl_div
    
    def compute_loss(self, alphas, labels: Optional[torch.tensor] = None):
        print("Given dim alphas: ", alphas.shape, " labels: ", labels.shape if labels is not None else '')
        k = alphas.shape[1] # num_classes
        precision = torch.sum(alphas, dim=1, keepdim=True)
        mean = F.softmax(alphas, dim=1)
        
        if labels is None:
            # ood sample, set all alphas to 1 to get a flat simplex or precision = num_classes, mean = 1/precision
            target_alphas = torch.ones_like(alphas)
            target_precision = torch.sum(target_alphas, dim=1, keepdim=True)
            target_mean = target_alphas / target_precision
        else:
            # in domain sample
            target_mean = torch.ones_like(alphas) * self.smooothing_factor # this is the epsilon smoothing param in paper
            target_mean = torch.clone(target_mean).scatter_(1, labels[:, None],
                                                               1-(k-1) * self.smooothing_factor)
            target_precision = torch.ones(alphas.shape[0], 1) * self.target_precision
        loss = self.compute_kl_div_dirichlets(target_mean, mean, target_precision, precision)
        return loss

