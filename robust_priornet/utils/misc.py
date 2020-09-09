import numpy as np
import os
from ..eval.uncertainty import UncertaintyMeasuresEnum, UncertaintyEvaluator
from ..eval.model_prediction_eval import ClassifierPredictionEvaluator
from ..eval.out_of_domain_detection import OutOfDomainDetectionEvaluator

def check_tpr_fpr(eval_dir, measure: UncertaintyMeasuresEnum, threshold):
    id_uncertainty = np.loadtxt(os.path.join(eval_dir,
                                             'id_' + measure._value_ + '.txt'))
    ood_uncertainty = np.loadtxt(os.path.join(eval_dir,
                                              'ood_' + measure._value_ + '.txt'))
    target_labels = np.concatenate((np.zeros_like(id_uncertainty),
                                    np.ones_like(ood_uncertainty)), axis=0)
    decision_fn_value = np.concatenate((id_uncertainty, ood_uncertainty), axis=0)
    if measure == UncertaintyMeasuresEnum.CONFIDENCE or measure == UncertaintyMeasuresEnum.PRECISION:
        decision_fn_value *= -1.0
    tn, fp, fn, tp = ClassifierPredictionEvaluator.compute_confusion_matrix_entries(decision_fn_value, target_labels, threshold)
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}, total_id: {len(id_uncertainty)}, total_ood: {len(ood_uncertainty)}")

def check_tpr_fpr_attack(attack_dir, measure, threshold):
    id_uncertainty = np.loadtxt(os.path.join(attack_dir, 'eval',
                                              measure._value_ + '.txt'))
    ood_uncertainty = np.loadtxt(os.path.join(attack_dir, 'ood_eval',
                                              measure._value_ + '.txt'))
    target_labels = np.concatenate((np.zeros_like(id_uncertainty),
                                    np.ones_like(ood_uncertainty)), axis=0)
    decision_fn_value = np.concatenate((id_uncertainty, ood_uncertainty), axis=0)
    if measure == UncertaintyMeasuresEnum.CONFIDENCE or measure == UncertaintyMeasuresEnum.PRECISION:
        decision_fn_value *= -1.0
    tn, fp, fn, tp = ClassifierPredictionEvaluator.compute_confusion_matrix_entries(decision_fn_value, target_labels, threshold)
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}, total_id: {len(id_uncertainty)}, total_ood: {len(ood_uncertainty)}")


def complete_roc_pr_attack_plots_ood2(ood2_attack_dir, ood1_attack_dir, epsilons):
    """
    Can be used when only ood-ii dataset is attacked using "attack_only_out_dist" flag
    in the attack script.
    """
    for epsilon in epsilons:
        # use the id-logits in eval dir from ood1_attack_dir to calc id uncertainties
        id_logits = np.loadtxt(os.path.join(ood1_attack_dir,
                                           f'e{epsilon}-attack',
                                           'eval',
                                           'logits.txt'))
        id_uncertainties = UncertaintyEvaluator(id_logits).get_all_uncertainties()
        
        target_epsilon_dir = os.path.join(ood2_attack_dir, f'e{epsilon}-attack')
        # store these values inside this eps dir
        eval_dir = os.path.join(target_epsilon_dir, 'eval')
        os.makedirs(eval_dir)
        for key in id_uncertainties.keys():
            np.savetxt(os.path.join(eval_dir, key._value_ + '.txt'), id_uncertainties[key])
            
        # Use ood2 logits to compute ood2 uncertainties
        ood_logits = np.loadtxt(os.path.join(target_epsilon_dir, 'ood_eval', 'logits.txt'))
        ood_uncertainties = UncertaintyEvaluator(ood_logits).get_all_uncertainties()
        
        # use these in-domain uncertainties and current ood2 uncertainties to compute roc, pr curves
        OutOfDomainDetectionEvaluator(id_uncertainties,
                                      ood_uncertainties,
                                      os.path.join(target_epsilon_dir, 'ood_eval')).eval()