import torch
from torch import nn


def construct_fgsm_attack(model,
                          inputs,
                          labels,
                          epsilon,
                          criterion=nn.CrossEntropyLoss(),
                          device=None,
                          pin_memory: bool = True):
    """
        Constructs adversarial images by single step maximization of the criterion wrt inputs
        and using the sign of the gradient to update the input image to generate the adversary. 

        Operates on a batch of images (inputs). Adversaries generated lie within L_infinity norm
        ball around the original image.

        params
        ------
        epsilon - float
            radius of the norm ball around the original image, within which the adversarial
            image constructed will reside.
    """
    adv_inputs = inputs.clone()
    adv_inputs.requires_grad = True
    model.eval()

    with torch.enable_grad():
        outputs = model(adv_inputs)

        epsilon = torch.ones([outputs.size()[0]]) * epsilon
        epsilon = epsilon.view([epsilon.size()[0], 1, 1, 1])


        if device is not None:
            epsilon = epsilon.to(device, non_blocking=pin_memory)

        loss = criterion(outputs, labels)
        assert torch.all(torch.isfinite(loss)).item()

        grad_outputs = torch.ones(loss.shape)
        if device is not None:
            grad_outputs = grad_outputs.to(device, non_blocking=pin_memory)

        grads = torch.autograd.grad(loss,
                                    adv_inputs,
                                    grad_outputs=grad_outputs,
                                    only_inputs=True)[0]

        update = epsilon * grads.sign()

        perturbed_image = adv_inputs + update
        # re-normalize the image to range (-1,1)
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        adv_inputs.data = perturbed_image

    return adv_inputs, labels
