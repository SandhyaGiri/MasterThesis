import torch.utils.data as data

from .datasets.adversarial_dataset import AdversarialDataset
from .datasets.torchvision_datasets import DatasetEnum, TorchVisionDataWrapper
from .datasets.transforms import TransformsBuilder
from .utils.common_data import (ATTACK_CRITERIA_MAP,
                                ATTACK_CRITERIA_TO_ENUM_MAP,
                                OOD_ATTACK_CRITERIA_MAP)
from .utils.persistence import persist_image_dataset
from .utils.pytorch import choose_torch_device, load_model


def generate_adversarial_images(dataset_name, ood_dataset: bool, indices, epsilon, attack_criteria, threshold, attack_type, model_dir, data_dir, result_dir):
    """
    Loads the specified dataset and then generates PGD adversarials for the
    chosen indices against the given model.
    Also persists the images.
    """
     # move to device if one available
    device = choose_torch_device(-1) # running locally
    model, ckpt = load_model(model_dir, device=device)

    # load the datasets
    vis = TorchVisionDataWrapper()

    # build transforms
    trans = TransformsBuilder()
    mean = (0.5,)
    std = (0.5,)
    trans.add_resize(ckpt['model_params']['n_in'])
    num_channels = ckpt['model_params']['num_channels']
    if ckpt['model_params']['model_type'].startswith('vgg'):
        trans.add_rgb_channels(num_channels)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trans.add_to_tensor()
    trans.add_normalize(mean, std)
    breakpoint()
    dataset = vis.get_dataset(dataset_name,
                              data_dir,
                              trans.get_transforms(),
                              None,
                              'test')
    # reduce the dataset, by picking only indices given
    final_dataset = data.Subset(dataset, indices)
    
    # gen adv dataset
    adv_success_detect_type = 'normal' if attack_type == 'misclassify' else 'ood-detect'
    if ood_dataset:
        adv_attack_criteria = OOD_ATTACK_CRITERIA_MAP[attack_criteria]
    else:
        adv_attack_criteria = ATTACK_CRITERIA_MAP[attack_criteria]
    print(adv_success_detect_type, adv_attack_criteria, ood_dataset)
    adv_dataset = AdversarialDataset(final_dataset, "pgd", model, epsilon,
                                         adv_attack_criteria, 'inf',
                                         0.4, 10,
                                         64, device=device,
                                         only_true_adversaries=False,
                                         ood_dataset=ood_dataset,
                                         adv_success_detect_type=adv_success_detect_type,
                                         uncertainty_measure=
                                         ATTACK_CRITERIA_TO_ENUM_MAP[attack_criteria],
                                         uncertainty_threshold=threshold)
    persist_image_dataset(adv_dataset, mean, std, num_channels, result_dir)
