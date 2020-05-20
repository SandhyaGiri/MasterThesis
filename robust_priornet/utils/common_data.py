from ..losses.attack_loss import AttackCriteria
from ..eval.uncertainty import UncertaintyMeasuresEnum

ATTACK_CRITERIA_MAP = {
    'confidence': AttackCriteria.confidence_loss,
    'diff_entropy': AttackCriteria.differential_entropy_loss,
    'mutual_info': AttackCriteria.distributional_uncertainty_loss,
    'entropy_of_exp': AttackCriteria.total_uncertainty_loss,
    'exp_entropy': AttackCriteria.expected_data_uncertainty_loss
}

OOD_ATTACK_CRITERIA_MAP = {
    'confidence': AttackCriteria.ood_confidence_loss,
    'diff_entropy': AttackCriteria.ood_differential_entropy_loss,
    'mutual_info': AttackCriteria.ood_distributional_uncertainty_loss
}

ATTACK_CRITERIA_TO_ENUM_MAP = {
    'confidence': UncertaintyMeasuresEnum.CONFIDENCE,
    'diff_entropy': UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY,
    'mutual_info': UncertaintyMeasuresEnum.DISTRIBUTIONAL_UNCERTAINTY
}
