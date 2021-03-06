from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..attacks.fast_gradient_sign import construct_fgsm_attack
from ..attacks.projected_gradient_descent import construct_pgd_attack
from ..attacks.carlini_wagner_l2 import construct_carlini_wagner_l2_attack
from ..attacks.projected_gradient_descent_targeted import \
    construct_pgd_targeted_attack
from ..eval.uncertainty import UncertaintyMeasuresEnum


class AdversarialDataset(Dataset):
    """
        Model aware adversarial dataset, that generates adversarial images from the original
        dataset, using gradients of the attack_criterion wrt original inputs.
    """
    def __init__(self,
                 org_dataset: Dataset,
                 attack_type: str,
                 model: nn.Module,
                 epsilon,
                 attack_criterion,
                 norm, step_size,
                 max_steps,
                 batch_size: int = 128,
                 device: Optional[torch.device] = None,
                 check_success: bool = True,
                 only_true_adversaries=False,
                 use_org_img_as_fallback=False,
                 targeted_attack=False,
                 adv_success_detect_type: str='normal',
                 ood_dataset: bool = False,
                 success_detect_criteria = {},
                 target_label='all'):
        assert attack_type in ['fgsm', 'pgd', 'cw']
        assert adv_success_detect_type in ['normal', 'ood-detect']

        # create dataloader
        dataloader = DataLoader(org_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        # Set model in eval mode
        model.eval()

        labels_list = []
        adv_list = []
        adv_indices_list = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get inputs (one batch)
                inputs, labels = data
                if device is not None:
                    inputs, labels = map(lambda x: x.to(device),
                                         (inputs, labels))
                if attack_type == 'fgsm':
                    adv_inputs, new_labels = construct_fgsm_attack(model=model,
                                                                   labels=labels,
                                                                   inputs=inputs,
                                                                   epsilon=epsilon,
                                                                   criterion=attack_criterion,
                                                                   device=device)
                elif attack_type == 'pgd':
                    attack_function = construct_pgd_targeted_attack if targeted_attack else construct_pgd_attack
                    adv_inputs, new_labels, adv_indices = attack_function(model=model,
                                                                  labels=labels,
                                                                  inputs=inputs,
                                                                  epsilon=epsilon,
                                                                  criterion=attack_criterion,
                                                                  device=device,
                                                                  norm=norm,
                                                                  step_size=step_size,
                                                                  max_steps=max_steps,
                                                                  check_success=check_success,
                                                                  only_true_adversaries=only_true_adversaries,
                                                                  use_org_img_as_fallback=use_org_img_as_fallback,
                                                                  success_detect_type=adv_success_detect_type,
                                                                  success_detect_args={
                                                                      'ood_dataset': ood_dataset,
                                                                      'criteria': success_detect_criteria
                                                                  },
                                                                  target_label=target_label if target_label == "all" else int(target_label))
                    adv_indices = [index + i * batch_size for index in adv_indices]
                elif attack_type == 'cw': # carlini-wagner
                    # create target label as the second largest logit label.
                    logits = model(inputs)
                    attack_targets = torch.argsort(logits, dim=1)[:, -2] # choose second column from end
                    adv_inputs, new_labels, adv_indices = construct_carlini_wagner_l2_attack(model=model,
                                                                                        labels=labels,
                                                                                        inputs=inputs,
                                                                                        epsilon=epsilon,
                                                                                        criterion=attack_criterion,
                                                                                        device=device,
                                                                                        targeted_attack=True,
                                                                                        target_labels=attack_targets,
                                                                                        max_iterations=max_steps)
                    adv_indices = [index + i * batch_size for index in adv_indices]
                # size of new labels can be lesser than org labels
                # when true adversaries are only returned by attacks
                if adv_inputs is not None:
                    labels_list.append(new_labels)
                    adv_list.append(adv_inputs.detach())
                    adv_indices_list += adv_indices
        if len(adv_list) > 0:
            self.labels = torch.cat(labels_list, dim=0).cpu()
            self.adv_inputs = torch.cat(adv_list, dim=0).cpu()
            self.adv_indices = adv_indices_list # stores true adversary indices
        else:
            self.labels = []
            self.adv_inputs = []

    def __getitem__(self, index):
        return (self.adv_inputs[index], self.labels[index].item())

    def __len__(self):
        return len(self.labels)
    
    def get_adversarial_indices(self):
        """
        Returns indices which resulted in an adversarial image.
        These indices can be indexed directly into orginal dataset provided.
        """
        return self.adv_indices
