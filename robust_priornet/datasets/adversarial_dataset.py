from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..attacks.fast_gradient_sign import construct_fgsm_attack
from ..attacks.projected_gradient_descent import construct_pgd_attack


class AdversarialDataset(Dataset):
    """
        Model aware adversarial dataset, that generates adversarial images from the original
        dataset, using gradients of the attack_criterion wrt original inputs.
    """
    def __init__(self, org_dataset: Dataset, attack_type: str,
                 model: nn.Module, epsilon, attack_criterion,
                 norm, step_size, max_steps,
                 batch_size: int = 128, device: Optional[torch.device] = None):
        assert attack_type in ['fgsm', 'pgd']
        # create dataloader
        dataloader = DataLoader(org_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        # Set model in eval mode
        model.eval()

        labels_list = []
        adv_list = []
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                # Get inputs
                inputs, labels = data
                if device is not None:
                    inputs, labels = map(lambda x: x.to(device),
                                         (inputs, labels))
                if attack_type == 'fgsm':
                    adv_inputs = construct_fgsm_attack(model=model, labels=labels, inputs=inputs,
                                                       epsilon=epsilon,
                                                       criterion=attack_criterion, device=device)
                elif attack_type == 'pgd':
                    adv_inputs = construct_pgd_attack(model=model, labels=labels, inputs=inputs,
                                                      epsilon=epsilon,
                                                      criterion=attack_criterion, device=device,
                                                      norm=norm, step_size=step_size,
                                                      max_steps=max_steps)

                labels_list.append(labels)
                adv_list.append(adv_inputs.detach())

        self.labels = torch.cat(labels_list, dim=0).cpu()
        self.adv_inputs = torch.cat(adv_list, dim=0).cpu()

    def __getitem__(self, index):
        return (self.adv_inputs[index], self.labels[index])

    def __len__(self):
        return len(self.labels)
