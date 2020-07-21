import torch
from torch import nn
import numpy as np

from ..utils.common_data import ATTACK_CRITERIA_MAP, OOD_ATTACK_CRITERIA_MAP

def _eval_for_adv_success_classify(model, adv_input, label, target_label, target_precision, precision_threshold_fn):
    logits = model(adv_input)
    alphas = torch.exp(logits)
    k = alphas.shape[1] # num_classes
    alpha_0 = torch.sum(alphas)
    return alpha_0 >= precision_threshold_fn(k, target_precision) and alphas[:,target_label].item() > alphas[:, label.item()].item()

def _get_adv_alpha_k(model, adv_input, target_label):
    logits = model(adv_input)
    alphas = torch.exp(logits)
    return alphas[:, target_label].item()

def _find_adv_single_input(model, input_image, label, epsilon, criterion,
                           device, norm, step_size,
                           max_steps, pin_memory, 
                           only_true_adversaries,
                           target_label,
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

            loss = criterion(output, label, target_label)
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
            is_success = _eval_for_adv_success_classify(model, adv_input,
                                                        label, target_label,
                                                        success_detect_args['target_precision'],
                                                        success_detect_args['precision_threshold_fn'])
            if is_success:
                adv_success_reached = True
                break
    return adv_input if not only_true_adversaries or adv_success_reached else None

def construct_pgd_targeted_attack(model,
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
                                use_org_img_as_fallback: bool = False,
                                success_detect_type='normal',
                                success_detect_args = {},
                                num_classes=10,
                                target_label='all'):
    """
        Targeted attack:
        Constructs adversarial images by doing multiple iterations of criterion maximization
        (maximize loss) for the true class label, and crietrion minimization (minimize loss) 
        for a wrong class label ( != true class label) to get the adversarial image within
        a p-norm ball around the input image that results in a targetted misclassification.

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
        target_label - string or int
            if int, then indicates a particular class label for which loss will be minimized
            if "all", then all class labels other than true label are used to generate classes-1
                adversarial images, which are then filtered by picking the best.
    """
    adv_inputs = []
    adv_labels = []
    for i in range(inputs.shape[0]):
        input_image = torch.unsqueeze(inputs[i], dim=0)
        label = labels[i].view(1,)
        if target_label == "all":
            # generate all k-1 adversarials
            best_adv_input = None
            best_adv_alpha_k = 0
            other_classes = [i for i in np.arange(0, num_classes) if i != label.item()]
            for target in other_classes:
                adv_input = _find_adv_single_input(model, input_image, label, epsilon,
                                                criterion, device, norm, step_size,
                                                max_steps, pin_memory,
                                                only_true_adversaries,
                                                target,
                                                success_detect_args,
                                                rel_step_size=0.1)
                # choose the best adversary
                if adv_input is not None and _get_adv_alpha_k(model, adv_input, target) > best_adv_alpha_k:
                    best_adv_alpha_k = _get_adv_alpha_k(model, adv_input, target)
                    best_adv_input = adv_input
            adv_input = best_adv_input
        else:
            # generate a particular adversary
            adv_input = _find_adv_single_input(model, input_image, label, epsilon,
                                            criterion, device, norm, step_size,
                                            max_steps, pin_memory,
                                            only_true_adversaries,
                                            target_label,
                                            success_detect_args,
                                            rel_step_size=0.1)
        if adv_input is not None:
            adv_inputs.append(adv_input)
            adv_labels.append(label)
        if adv_input is None and use_org_img_as_fallback:
            adv_inputs.append(input_image)
            adv_labels.append(label)

    if len(adv_inputs) == 0:
        return None, None
    return torch.cat(adv_inputs, dim=0), torch.cat(adv_labels, dim=0)