import torch

from ..eval.uncertainty import UncertaintyEvaluatorTorch


class AttackCriteria:
    """
    Wrapper class on UncertaintyEvaluatorTorch, that provides methods which can be used as
    attack criterion (loss function) during adversarial attacks or adversarial training.
    """
    @staticmethod
    def confidence_loss(outputs, labels):
        return torch.neg(UncertaintyEvaluatorTorch(outputs).get_confidence())

    @staticmethod
    def differential_entropy_loss(outputs, labels):
        return UncertaintyEvaluatorTorch(outputs).get_differential_entropy()

    @staticmethod
    def distributional_uncertainty_loss(outputs, labels):
        return UncertaintyEvaluatorTorch(outputs).get_distributional_uncertainty()

    @staticmethod
    def total_uncertainty_loss(outputs, labels):
        return UncertaintyEvaluatorTorch(outputs).get_total_uncertainty()

    @staticmethod
    def expected_data_uncertainty_loss(outputs, labels):
        return UncertaintyEvaluatorTorch(outputs).get_expected_data_uncertainty()

    @staticmethod
    def ood_confidence_loss(outputs, labels):
        """
        For out of distribution samples, the confidence of the model need to be
        lesser than that of in distribution samples. So confidence itself can be
        considered as a loss function (without negation).
        """
        return UncertaintyEvaluatorTorch(outputs).get_confidence()

    @staticmethod
    def ood_differential_entropy_loss(outputs, labels):
        """
        For out of distribution samples, the differential entropy which is an
        uncertainty measure should be higher than that of in dist samples. So
        as a loss function we minimize the negation of differential entropy.
        """
        return torch.neg(UncertaintyEvaluatorTorch(outputs).get_differential_entropy())
