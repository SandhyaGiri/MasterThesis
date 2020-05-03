import torch

from ..eval.uncertainty import UncertaintyEvaluatorTorch


class AttackCriteria:
    """
    Wrapper class on UncertaintyEvaluatorTorch, that provides methods which can be used as
    attack criterion (loss function) during adversarial attacks or adversarial training.
    """
    @staticmethod
    def confidence_loss(outputs, labels):
        return torch.neg(UncertaintyEvaluatorTorch(outputs).get_confidence()[0])

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
