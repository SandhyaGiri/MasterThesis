import torch
from torch import nn
import numpy as np

from ..eval.uncertainty import (UncertaintyEvaluatorTorch,
                                UncertaintyMeasuresEnum)


def _eval_for_adv_success_normal_classify(model, adv_input, label):
    logit = model(adv_input)
    prob = nn.functional.softmax(logit, dim=1)
    pred = torch.max(prob, dim=1)[1] # indices
    return pred.item() != label.item() # adversarial success acheieved

def _eval_for_adv_success_ood_detect(model, adv_input, label, uncertainty_measure: UncertaintyMeasuresEnum, threshold):
    logit = model(adv_input)
    uncertainty_value = UncertaintyEvaluatorTorch(logit).get_uncertainty(uncertainty_measure,
                                                                         negate_confidence=True)
    uncertainty_value = uncertainty_value.detach().cpu().numpy()
    pred = np.zeros((logit.shape[0]))
    pred[np.round(uncertainty_value, 4) >= np.round(threshold, 4)] = 1
    return pred != label.item() # adversarial success acheieved

def _find_adv_single_input(model, input_image, label, epsilon, criterion,
                           device, norm, step_size,
                           max_steps, pin_memory, 
                           only_true_adversaries,
                           success_detect_type,
                           success_detect_args,
                           rel_step_size: float = None):
    """
    Finds adversarial sample for a single input image, using the hyperparameters
    given. (batch_size =1, so input_image must be a 4D tensor)
    """
    adv_input = input_image.clone()
    adv_input.requires_grad = True

    # set model in eval mode
    model.eval()

    epsilon = torch.ones([1, 1, 1, 1]) * epsilon # transform to a 4D tensor

    if device is not None:
        epsilon = epsilon.to(device, non_blocking=pin_memory)

    if rel_step_size is not None:
        step_size = rel_step_size * epsilon

    adv_success_reached = False

    for _ in range(max_steps):
        with torch.enable_grad():
            output = model(adv_input)

            loss = criterion(output, label)
            assert torch.all(torch.isfinite(loss)).item()

            grad_output = torch.ones(loss.shape)
            if device is not None:
                grad_output = grad_output.to(device, non_blocking=pin_memory)

            grad = torch.autograd.grad(loss,
                                       adv_input,
                                       grad_outputs=grad_output,
                                       only_inputs=True)[0]

            # compute the perturbed_image
            # project the perturbed_image back onto the norm-ball
            if norm == 'inf':
                update = step_size * grad.sign()
                perturbed_image = adv_input + update
                perturbed_image = torch.max(
                    torch.min(perturbed_image, input_image + epsilon), input_image - epsilon)
            elif norm == '2':
                update = step_size * grad.sign()
                # l2 norm ball centered around "adv_input" (old image)
                delta = torch.clone(update)
                # as the first dim is batch dimension
                # find norm of the 2D image (all other dimensions)
                norm_value = delta.view(delta.shape[0], -1).norm(p=2, dim=1)
                norm_value = norm_value.view(1, 1, 1, 1) # batch_size is 1
                # mask to indicate if div by norm is necessary or not
                mask = norm_value <= epsilon
                scaling_factor = norm_value
                # update only image deltas whose norm value is more than epsilon
                scaling_factor[mask] = epsilon

                delta = delta * (epsilon / scaling_factor)

                # add corrected delta whose l2 norm is less than epsilon
                perturbed_image = adv_input + delta

            # re-normalize the image to range (-1,1)
            perturbed_image = torch.clamp(perturbed_image, -1, 1)
            adv_input.data = perturbed_image

        # evaluate if adv image results in misclassification
        # if misclassified stop
        with torch.no_grad():
            is_success = False
            if success_detect_type == 'normal':
                is_success = _eval_for_adv_success_normal_classify(model, adv_input, label)
            elif success_detect_type == 'ood-detect':
                # in-domain = label 0, out-domain = label 1
                ood_label = torch.ones_like(label) if success_detect_args['ood_dataset'] else torch.zeros_like(label)
                is_success = _eval_for_adv_success_ood_detect(model, adv_input,
                                                              ood_label,
                                                              success_detect_args['uncertainty_measure'],
                                                              success_detect_args['threshold'])
            if is_success:
                adv_success_reached = True
                break
    return adv_input if not only_true_adversaries or adv_success_reached else None

def construct_pgd_attack(model,
                         inputs,
                         labels,
                         epsilon,
                         criterion=nn.CrossEntropyLoss(),
                         device=None,
                         norm="inf",
                         step_size=0.4,
                         max_steps=10,
                         pin_memory: bool = True,
                         only_true_adversaries: bool = False,
                         success_detect_type: str = 'normal',
                         success_detect_args = {}):
    """
        Constructs adversarial images by doing multiple iterations of criterion maximization
        (maximize loss) to get the adversarial image within a p-norm ball around the
        input image that results in a misclassification.

        Operates on a batch of images (inputs).

        params
        -------
        norm - string
            can be any one of the 'p' norm values given to torch.norm function.
            Used for projecting the perturbed image on to the p-norm ball.
            Possible values: "inf" , "2"
        epsilon - float
            radius of the norm ball around the original image, within which the adversarial
            image constructed will reside.
        step_size - float
            indicates the size of the gradient/gradient sign update to be done at each step.
        max_steps - int
            indicates the maximum steps to perform for chosing the best adversary
            (one with max loss/criterion).
    """
    adv_inputs = []
    adv_labels = []
    for i in range(inputs.shape[0]):
        input_image = torch.unsqueeze(inputs[i], dim=0)
        label = labels[i].view(1,)
        adv_input = _find_adv_single_input(model, input_image, label, epsilon,
                                           criterion, device, norm, step_size,
                                           max_steps, pin_memory,
                                           only_true_adversaries,
                                           success_detect_type,
                                           success_detect_args,
                                           rel_step_size=0.1)
        if adv_input is not None:
            adv_inputs.append(adv_input)
            adv_labels.append(label)
    if len(adv_inputs) == 0:
        return None, None
    return torch.cat(adv_inputs, dim=0), torch.cat(adv_labels, dim=0)
