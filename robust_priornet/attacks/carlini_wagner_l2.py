import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def _var2numpy(var):
    return var.data.cpu().numpy()

def atanh(x, eps=1e-6):
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def _convert_img_to_tanh(box_min, box_max, x):
    _box_mul = (box_max - box_min) * 0.5
    _box_plus = (box_max + box_min) * 0.5
    return atanh((x - _box_plus) / _box_mul)

def _convert_tanh_to_img(box_min, box_max, x):
    _box_mul = (box_max - box_min) * 0.5
    _box_plus = (box_max + box_min) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus

def _compensate_confidence(logits, targets, targeted_attack: bool, confidence):
    outputs_comp = np.copy(logits)
    row_index = np.arange(targets.shape[0])
    if targeted_attack:
        # for each image $i$:
        # if targeted, `outputs[i, target_onehot]` should be larger than
        # `max(outputs[i, ~target_onehot])` by `self.confidence`
        outputs_comp[row_index, targets] -= confidence
    else:
        # for each image $i$:
        # if not targeted, `max(outputs[i, ~target_onehot]` should be larger
        # than `outputs[i, target_onehot]` (the ground truth image labels)
        # by `self.confidence`
        outputs_comp[row_index, targets] += confidence
    return outputs_comp

def _attack_success(prediction, target, targeted_attack: bool, truth_label):
    if targeted_attack:
        return prediction == target and target != truth_label
    else:
        return prediction != target

def _optimize(model, optimizer, inputs_tanh_var, pert_tanh_var, targets_oh_var,
              c_var, targeted_attack, box_min, box_max, confidence):
    # the adversarial examples in the image space
    # of dimension [B x C x H x W]
    advxs_var = _convert_tanh_to_img(box_min, box_max, inputs_tanh_var + pert_tanh_var)  # type: Variable
    # the perturbed activation before softmax
    pert_outputs_var = model(advxs_var)  # type: Variable
    # the original inputs
    inputs_var = _convert_tanh_to_img(box_min, box_max, inputs_tanh_var)  # type: Variable

    perts_norm_var = torch.pow(advxs_var - inputs_var, 2)
    perts_norm_var = torch.sum(perts_norm_var.view(
            perts_norm_var.size(0), -1), 1)

    # In Carlini's code, `target_activ_var` is called `real`.
    # It should be a Variable of tensor of dimension [B], such that the
    # `target_activ_var[i]` is the final activation (right before softmax)
    # of the $t$th class, where $t$ is the attack target or the image label
    #
    # noinspection PyArgumentList
    target_activ_var = torch.sum(targets_oh_var * pert_outputs_var, 1)
    inf = 1e4  # sadly pytorch does not work with np.inf;
                # 1e4 is also used in Carlini's code
    # In Carlini's code, `maxother_activ_var` is called `other`.
    # It should be a Variable of tensor of dimension [B], such that the
    # `maxother_activ_var[i]` is the maximum final activation of all classes
    # other than class $t$, where $t$ is the attack target or the image
    # label.
    #
    # The assertion here ensures (sufficiently yet not necessarily) the
    # assumption behind the trick to get `maxother_activ_var` holds, that
    # $\max_{i \ne t}{o_i} \ge -\text{_inf}$, where $t$ is the target and
    # $o_i$ the $i$th element along axis=1 of `pert_outputs_var`.
    #
    # noinspection PyArgumentList
    assert (pert_outputs_var.max(1)[0] >= -inf).all(), 'assumption failed'
    # noinspection PyArgumentList
    maxother_activ_var = torch.max(((1 - targets_oh_var) * pert_outputs_var
                                    - targets_oh_var * inf), 1)[0]

    # Compute $f(x')$, where $x'$ is the adversarial example in image space.
    # The result `f_var` should be of dimension [B]
    if targeted_attack:
        # if targeted, optimize to make `target_activ_var` larger than
        # `maxother_activ_var` by `self.confidence`
        #
        # noinspection PyArgumentList
        f_var = torch.clamp(maxother_activ_var - target_activ_var
                            + confidence, min=0.0)
    else:
        # if not targeted, optimize to make `maxother_activ_var` larger than
        # `target_activ_var` (the ground truth image labels) by
        # `self.confidence`
        #
        # noinspection PyArgumentList
        f_var = torch.clamp(target_activ_var - maxother_activ_var
                            + confidence, min=0.0)
    # the total loss of current batch, should be of dimension [1]
    batch_loss_var = torch.sum(perts_norm_var + c_var * f_var)  # type: Variable
    batch_loss_var = Variable(batch_loss_var, requires_grad=True)

    # Do optimization for one step
    optimizer.zero_grad()
    batch_loss_var.backward()
    optimizer.step()

    # Make some records in python/numpy on CPU
    batch_loss = batch_loss_var.item()  # type: float
    pert_norms_np = _var2numpy(perts_norm_var)
    pert_outputs_np = _var2numpy(pert_outputs_var) # logits or the score
    advxs_np = _var2numpy(advxs_var)
    return batch_loss, pert_norms_np, pert_outputs_np, advxs_np
    
def construct_carlini_wagner_l2_attack(model,
                                       inputs,
                                       labels,
                                       epsilon,
                                       criterion=nn.CrossEntropyLoss(),
                                       device=None,
                                       pin_memory: bool = True,
                                       targeted_attack=True,
                                       target_labels = None,
                                       min_confidence=0,
                                       constant_range=(1e-3, 1e10),
                                       binary_search_steps=9,
                                       max_iterations=10,
                                       optimizer_lr=1e-2,
                                       abort_early=True,
                                       early_abortion_tolerance=1e-4,
                                       num_classes=10,
                                       ):
    """
        params
        ------
        epsilon - float
            max strength of the adversarial perturbation
        labels - tensor
            is target label when targeted_attack is true, otherwise it is the
            original truth label.

        References
        ----------
            [1] Nicholas Carlini, David Wagner: "Towards Evaluating the
            Robustness of Neural Networks", https://arxiv.org/abs/1608.04644
            [2] https://github.com/carlini/nn_robust_attacks
            [3] https://github.com/kkew3/pytorch-cw2
    """
    # as our images are normalized in range(-1,1)
    box_min = -1
    box_max = 1
    
    targets_np = labels.clone().cpu().numpy() if not targeted_attack else target_labels.clone().cpu().numpy()
    batch_size = inputs.size(0)
    device = inputs.device
    
    repeat = (binary_search_steps >= 10)

    # constant c
    lower_bounds = torch.zeros(batch_size, device=device) # for each image/input
    upper_bounds = torch.ones(batch_size, device=device) * constant_range[1]
    scale_consts = torch.ones(batch_size, device=device) * constant_range[0]
    scale_consts_np = scale_consts.detach().cpu().numpy()
    upper_bounds_np = upper_bounds.detach().cpu().numpy()
    lower_bounds_np = lower_bounds.detach().cpu().numpy()
    
    # re-parametrization trick (use x to get w)
    inputs_tanh = _convert_img_to_tanh(box_min, box_max, inputs)
    inputs_tanh_variable = Variable(inputs_tanh, requires_grad=False)
    
    # the one-hot encoding of `targets`
    targets_oh = torch.zeros(labels.size() + (num_classes,), device=device)
    targets_oh.scatter_(1, labels.unsqueeze(1), 1.0)
    targets_oh_variable = Variable(targets_oh, requires_grad=False)

    # 
    perturbation_tanh = torch.zeros_like(inputs, device=device)
    perturbation_tanh_variable = Variable(perturbation_tanh, requires_grad=True)
    
    overall_best_l2 = np.ones(batch_size) * np.inf # we want to find a minimum perturbation per l2 norm
    overall_best_l2_ppred = -np.ones(batch_size)
    overall_best_adv_img = inputs.clone().cpu().numpy() # start with given input images
    true_adv_indices = []
    
    optimizer = optim.Adam([perturbation_tanh_variable], lr=optimizer_lr)

    # binary search outer loop
    for outer_step in range(binary_search_steps):
        if repeat and outer_step == binary_search_steps - 1:
            scale_consts = upper_bounds
        scale_consts_variable = Variable(scale_consts, requires_grad=False)
        
        # the minimum L2 norms of perturbations found during optimization
        best_l2 = np.ones(batch_size) * np.inf
        best_l2_ppred = -np.ones(batch_size)

        prev_batch_loss = np.inf  # type: float
        for optim_step in range(max_iterations):
            batch_loss, pert_norms_np, pert_outputs_np, advxs_np = \
                _optimize(model, optimizer, inputs_tanh_variable,
                          perturbation_tanh_variable, targets_oh_variable,
                          scale_consts_variable, targeted_attack, box_min, box_max,
                          min_confidence)
            if optim_step % 10 == 0: print(f'batch [{optim_step}] loss: {batch_loss}')

            if abort_early and not optim_step % (max_iterations // 10):
                if batch_loss > prev_batch_loss * (1 - early_abortion_tolerance):
                    break
                prev_batch_loss = batch_loss

            # update best attack found during optimization
            pert_predictions_np = np.argmax(pert_outputs_np, axis=1)
            comp_pert_predictions_np = np.argmax(
                    _compensate_confidence(pert_outputs_np,
                                           targets_np,
                                           targeted_attack,
                                           min_confidence),
                    axis=1)
            for i in range(batch_size):
                l2 = pert_norms_np[i]
                cppred = comp_pert_predictions_np[i]
                ppred = pert_predictions_np[i]
                tlabel = targets_np[i]
                ax = advxs_np[i]
                if _attack_success(cppred, tlabel, targeted_attack, labels[i]):
                    if i not in true_adv_indices:
                        true_adv_indices.append(i)
                    assert cppred == ppred
                    if l2 < best_l2[i]:
                        best_l2[i] = l2
                        best_l2_ppred[i] = ppred
                    if l2 < overall_best_l2[i]:
                        overall_best_l2[i] = l2
                        overall_best_l2_ppred[i] = ppred
                        overall_best_adv_img[i] = ax
        
        # binary search of `scale_const`
        for i in range(batch_size):
            tlabel = targets_np[i]
            assert best_l2_ppred[i] == -1 or \
                    _attack_success(best_l2_ppred[i], tlabel, targeted_attack, labels[i])
            assert overall_best_l2_ppred[i] == -1 or \
                    _attack_success(overall_best_l2_ppred[i], tlabel, targeted_attack, labels[i])
            if best_l2_ppred[i] != -1:
                # successful; attempt to lower `scale_const` by halving it
                if scale_consts_np[i] < upper_bounds_np[i]:
                    upper_bounds_np[i] = scale_consts_np[i]
                # `upper_bounds_np[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds_np[i] has never been updated by
                # scale_consts_np[i] until
                # `scale_consts_np[i] > 0.1 * c_range[1]`
                if upper_bounds_np[i] < constant_range[1] * 0.1:
                    scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
            else:
                # failure; multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_consts_np[i] > lower_bounds_np[i]:
                    lower_bounds_np[i] = scale_consts_np[i]
                if upper_bounds_np[i] < constant_range[1] * 0.1:
                    scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                else:
                    scale_consts_np[i] *= 10
    overall_best_adv_img = torch.from_numpy(overall_best_adv_img).float()
    return overall_best_adv_img, labels, list(np.sort(true_adv_indices))
