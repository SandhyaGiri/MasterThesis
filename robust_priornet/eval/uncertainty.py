from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch
from scipy.special import digamma, gammaln


class UncertaintyMeasuresEnum(Enum):
    """
    Enum class holding uncertainty measures used in prior networks.
    """
    CONFIDENCE = ("confidence",
                  "Max probability of predictive distribution by a model. \
                      Lesser value means more uncertainty.")
    PRECISION = ("precision",
                 "Precision/weight or alpha_0 of the output dirichlet distribution by the PN model. \
                     Lesser value means more uncertainty.")
    TOTAL_UNCERTAINTY = ("total_uncertainty",
                         "Entropy of the expected categorical distribution.")
    EXPECTED_DATA_UNCERTAINTY = ("expected_data_uncertainty",
                                 "Expectation over entropy of the categorical distribution.")
    DISTRIBUTIONAL_UNCERTAINTY = ("distributional_uncertainty",
                                  "Mutual information between labels y and categorical for a DPN.")
    DIFFERENTIAL_ENTROPY = ("differential_entropy",
                            "Clear indicator of a flat dirichlet dist, as this measure is max when \
                                all categorical dist sampled from dirichlet are spread across \
                                the simplex.")

    def __init__(self, name, desc):
        self._value_ = name
        self.desc = desc

    @classmethod
    def get_enum(cls, name):
        return cls._member_map_.get(name)

class BaseUncertaintyEvaluator(ABC):

    @abstractmethod
    def get_confidence(self):
        """Returns max probability predicted by the model"""
        raise NotImplementedError

    @abstractmethod
    def get_precision(self):
        raise NotImplementedError

    @abstractmethod
    def get_total_uncertainty(self):
        raise NotImplementedError

    @abstractmethod
    def get_expected_data_uncertainty(self):
        raise NotImplementedError

    @abstractmethod
    def get_differential_entropy(self):
        raise NotImplementedError

    def get_distributional_uncertainty(self):
        """Returns mutual information"""
        return self.get_total_uncertainty() - self.get_expected_data_uncertainty()

    def get_all_uncertainties(self):
        """Returns a dictionary of all uncertainty measures
        as available in UncertaintyMeasuresEnum."""
        return {
            UncertaintyMeasuresEnum.CONFIDENCE:
                self.get_confidence(),
            UncertaintyMeasuresEnum.TOTAL_UNCERTAINTY:
                self.get_total_uncertainty(),
            UncertaintyMeasuresEnum.EXPECTED_DATA_UNCERTAINTY:
                self.get_expected_data_uncertainty(),
            UncertaintyMeasuresEnum.DISTRIBUTIONAL_UNCERTAINTY:
                self.get_distributional_uncertainty(),
            UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY:
                self.get_differential_entropy()
        }

    def get_uncertainty(self, measure: UncertaintyMeasuresEnum, negate_confidence= False):
        if measure == UncertaintyMeasuresEnum.CONFIDENCE:
            return self.get_confidence() if negate_confidence is False else -1 * self.get_confidence()
        elif measure == UncertaintyMeasuresEnum.PRECISION:
            return self.get_precision() if negate_confidence is False else -1 * self.get_precision()
        elif measure == UncertaintyMeasuresEnum.TOTAL_UNCERTAINTY:
            return self.get_total_uncertainty()
        elif measure == UncertaintyMeasuresEnum.EXPECTED_DATA_UNCERTAINTY:
            return self.get_expected_data_uncertainty()
        elif measure == UncertaintyMeasuresEnum.DISTRIBUTIONAL_UNCERTAINTY:
            return self.get_distributional_uncertainty()
        elif measure == UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY:
            return self.get_differential_entropy()
    
class UncertaintyEvaluator(BaseUncertaintyEvaluator):
    """
    Evaluator class carrying methods which can be used to retrieve uncertainty measures
    as detailed in PriorNet paper.

    Params
    ------
        logits: numpy ndarray or list or tuple
    """
    def __init__(self, logits, epsilon=1e-8):
        super(UncertaintyEvaluator, self).__init__()
        self.logits = np.asarray(logits, dtype=np.float64)
        self.alphas = np.exp(logits)
        self.alpha_0 = np.sum(self.alphas, axis=1, keepdims=True)
        self.probs = self.alphas / self.alpha_0
        self.epsilon = epsilon # used for controlling overflows/underflows

    def get_confidence(self):
        """Returns max probability predicted by the model"""
        return np.max(self.probs, axis=1, keepdims=True)

    def get_precision(self):
        return self.alpha_0

    def get_total_uncertainty(self):
        return -1 * np.sum(self.probs * np.log(self.probs + self.epsilon), axis=1)

    def get_expected_data_uncertainty(self):
        digamma_term = digamma(self.alphas + 1.0) - digamma(self.alpha_0 + 1.0)
        dirichlet_mean = self.alphas / self.alpha_0
        return -1 * np.sum(dirichlet_mean * digamma_term, axis=1)

    def get_differential_entropy(self):
        log_term = gammaln(self.alphas)
        digamma_term = (self.alphas - 1) * (digamma(self.alphas) - digamma(self.alpha_0))
        return np.sum(log_term - digamma_term, axis=1, keepdims=True) - gammaln(self.alpha_0)

class UncertaintyEvaluatorTorch(BaseUncertaintyEvaluator):
    """
    Evaluator class carrying methods which can be used to retrieve uncertainty measures
    as detailed in PriorNet paper. (using torch functions)

    Params
    ------
        logits: a torch tensor
    """
    def __init__(self, logits, epsilon=1e-8):
        super(UncertaintyEvaluatorTorch, self).__init__()
        self.logits = logits
        self.alphas = torch.exp(logits)
        # precision of the predictive posterior dirichlet
        self.alpha_0 = torch.sum(self.alphas, dim=1, keepdim=True)
        self.probs = self.alphas / self.alpha_0
        self.epsilon = epsilon # used for controlling overflows/underflows

    def get_confidence(self):
        return torch.max(self.probs, dim=1, keepdim=True)[0] # return only max values not indices

    def get_precision(self):
        return self.alpha_0

    def get_total_uncertainty(self):
        return -1 * torch.sum(self.probs * torch.log(self.probs + self.epsilon), dim=1)

    def get_expected_data_uncertainty(self):
        digamma_term = torch.digamma(self.alphas + 1.0) - torch.digamma(self.alpha_0 +1.0)
        dirichlet_mean = self.alphas / self.alpha_0
        return -1 * torch.sum(dirichlet_mean * digamma_term, dim=1) # sum across classes

    def get_differential_entropy(self):
        log_term = torch.lgamma(self.alphas)
        digamma_term = (self.alphas - 1) *(torch.digamma(self.alphas) - torch.digamma(self.alpha_0))
        return torch.sum(log_term - digamma_term, dim=1, keepdim=True)  - torch.lgamma(self.alpha_0)
