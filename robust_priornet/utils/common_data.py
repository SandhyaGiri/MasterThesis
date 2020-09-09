from ..losses.attack_loss import AttackCriteria
from ..eval.uncertainty import UncertaintyMeasuresEnum

ATTACK_CRITERIA_MAP = {
    'confidence': AttackCriteria.confidence_loss,
    'diff_entropy': AttackCriteria.differential_entropy_loss,
    'mutual_info': AttackCriteria.distributional_uncertainty_loss,
    'entropy_of_exp': AttackCriteria.total_uncertainty_loss,
    'exp_entropy': AttackCriteria.expected_data_uncertainty_loss,
    'precision': AttackCriteria.precision_loss,
    'precision_targeted': AttackCriteria.precision_target_loss,
    'alpha_k': AttackCriteria.alpha_k_loss
}

OOD_ATTACK_CRITERIA_MAP = {
    'confidence': AttackCriteria.ood_confidence_loss,
    'diff_entropy': AttackCriteria.ood_differential_entropy_loss,
    'mutual_info': AttackCriteria.ood_distributional_uncertainty_loss,
    'precision': AttackCriteria.ood_precision_loss,
    'precision_targeted': None,
    'alpha_k': None
}

ATTACK_CRITERIA_TO_ENUM_MAP = {
    'confidence': UncertaintyMeasuresEnum.CONFIDENCE,
    'diff_entropy': UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY,
    'mutual_info': UncertaintyMeasuresEnum.DISTRIBUTIONAL_UNCERTAINTY,
    'precision': UncertaintyMeasuresEnum.PRECISION,
    'precision_targeted': 'precision_targeted',
    'alpha_k': 'alpha_k'
}

PRECISION_THRESHOLDS_MAP = {
    'class_relative_strict': lambda k, b: (k-1)/k * b,
    'class_relative_relaxed': lambda k, b: 1/k * b,
    '20': lambda k, b: k+10,
    '50': lambda k, b: k+40
}

# chosen thresholds - will be used in eval, attack scripts
CHOSEN_THRESHOLDS = {
    'precision': 15,
    'alpha_k': 10,
    'diff_entropy': -26.4142,
    'mutual_info': 0.1319
}