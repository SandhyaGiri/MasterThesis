import torch

def construct_target_dirichlets(id_images: torch.tensor,
                                id_labels: torch.tensor,
                                ood_images: torch.tensor,
                                num_classes: int,
                                target_precision: int,
                                smoothing_factor: float = 1e-2):
    """
        Computes normal target dirichlets for id and ood images, as per the Prior Network
        training. Creates a peaky dirichlet dist for in domain samples and a flat dirichlet dist
        for out domain samples.
    """
    id_target_mean, id_target_precision = construct_target_dirichlet_in(id_images,
                                                                        id_labels,
                                                                        target_precision,
                                                                        num_classes,
                                                                        smoothing_factor)
    ood_target_mean, ood_target_precision = construct_target_dirichlet_out(ood_images, num_classes)
    return (id_target_mean, id_target_precision), (ood_target_mean, ood_target_precision)
    
def construct_ccat_adv_target_dirichlets(id_images: torch.tensor,
                                         id_advs: torch.tensor,
                                         id_labels: torch.tensor,
                                         ood_images: torch.tensor,
                                         ood_advs: torch.tensor,
                                         attack_norm: str,
                                         attack_eps: float,
                                         num_classes: int,
                                         target_precision: int,
                                         smoothing_factor: float = 1e-2,
                                         decay_param: int = 20):
    """
        With the given adv images and their corresponding adv images, calculates the distance
        between adv image and original image and uses it to compute target dirichlet distributions.
        For those perturbations that are far away from the in-domain image, we output a uniform
        distribution similar to OOD samples.

        params
        ------
            id_advs: adv images corresponding to id_images, in order to calculate the perturbation (delta).
            ood_advs: adv images corresponding to ood_images, for caluclating the perturbation (delta).

        returns
        -------
            the parameters of the desired target dirichlet distribution
            for in domain and out domain dataset.
        
        References
        ----------
            [1] David Stutz, Matthias Hein, and Bernt Schiele. Confidence-Calibrated Adversarial Training:
            Generalizing to Unseen Attacks. 2019. arXiv: 1910.06259.
            
    """
    k = num_classes
    id_lambdas = []
    ood_lambdas = []
    norm = float('inf') if attack_norm == 'inf' else 2
    # compute lambdas for in samples
    for i in range(id_images.shape[0]):
        perturbation = (id_advs[i] - id_images[i])
        delta = perturbation.view(-1).norm(p=norm, dim=0) # norm of entire difference image
        eps = torch.zeros_like(delta) + attack_eps
        delta_by_eps = (delta/eps)
        lambda_param = 1 - torch.pow(torch.min(torch.ones_like(delta_by_eps), delta_by_eps), decay_param) # one item tensor
        id_lambdas.append(lambda_param)
    # compute lambdas for out samples
    for i in range(ood_images.shape[0]):
        perturbation = (ood_advs[i] - ood_images[i])
        delta = perturbation.view(-1).norm(p=norm, dim=0) # norm of entire difference image
        eps = torch.zeros_like(delta) + attack_eps
        delta_by_eps = (delta/eps)
        lambda_param = 1 - torch.pow(torch.min(torch.ones_like(delta_by_eps), delta_by_eps), decay_param) # one item tensor
        ood_lambdas.append(lambda_param)
        
    id_target_mean, id_target_precision = construct_target_dirichlet_in(id_advs,
                                                                        id_labels,
                                                                        target_precision,
                                                                        num_classes,
                                                                        smoothing_factor)
    id_target_mean_adjusted, id_target_precision_adjusted = construct_adv_target_dirichlet(id_lambdas,
                                                                                           num_classes,
                                                                                           id_target_mean,
                                                                                           id_target_precision)
    # print(f"Adjusted ID precision: {torch.min(id_target_precision_adjusted, dim=0)[0]}, {torch.max(id_target_precision_adjusted, dim=0)[0]}")
    # number of ID samples with less than 20 as precision
    id_precision_check = id_target_precision_adjusted.clone().detach()
    num_less_precision_samples = torch.sum(torch.tensor(id_precision_check < 20, dtype=int)).item()
    # print("Number of ID samples with <20 precision: ", num_less_precision_samples)
    ood_target_mean, ood_target_precision = construct_target_dirichlet_out(ood_advs, num_classes)
    ood_target_mean_adjusted, ood_target_precision_adjusted = construct_adv_target_dirichlet(ood_lambdas,
                                                                                             num_classes,
                                                                                             ood_target_mean,
                                                                                             ood_target_precision)
    # print(f"Adjusted OOD precision: {torch.min(ood_target_precision_adjusted, dim=0)[0]}, {torch.max(ood_target_precision_adjusted, dim=0)[0]}")
    return (id_target_mean_adjusted, id_target_precision_adjusted), (ood_target_mean_adjusted, ood_target_precision_adjusted)

def construct_adv_target_dirichlet(lambdas, num_classes, old_target_mean, old_target_precision):
    k = num_classes
    lambdas = torch.tensor(lambdas).view(-1, 1)
    lambdas_repeated = lambdas.repeat(1, k)
    one_minus_lambdas = 1 - lambdas_repeated
    target_mean_adjusted = torch.mul(lambdas_repeated, old_target_mean).add(torch.mul(one_minus_lambdas,
                                                                                  torch.ones_like(old_target_mean) * 1/k))
    target_precision_adjusted = lambdas * old_target_precision
    target_precision_adjusted = target_precision_adjusted.clamp(min=num_classes)
    return target_mean_adjusted, target_precision_adjusted

def construct_target_dirichlet_in(id_images, id_labels, target_precision, num_classes, smoothing_factor):
    """
        Calculates and returns the target peaky Dirichlet distributions for the given in-domain samples.
        
        params
        ------
        smoothing_factor - float
            Indicates the amount of density to be given to classes which are not the actual truth labels.
        target_precision - int
            Indicates the weight of the Dirichlet distribution, most of which will be concentrated at the
            target truth label.
    """
    # this is the epsilon smoothing param in paper
    k = num_classes
    id_target_mean = id_images.new_ones((id_images.shape[0], num_classes)) * smoothing_factor
    id_target_mean = torch.clone(id_target_mean).scatter_(1, id_labels[:, None],
                                                    1-(k-1) * smoothing_factor)
    id_target_precision = id_images.new_ones((id_images.shape[0], 1)) * target_precision
    return id_target_mean, id_target_precision

def construct_target_dirichlet_out(ood_images, num_classes):
    """
        Calculates and returns the target flat Dirichlet distributions for the given out-domain samples.
        Precision of this distribution is equal to the number of classes.
    """
    # ood sample, set all alphas to 1 to get a flat simplex
    # or precision = num_classes, mean = 1/precision
    ood_target_alphas = ood_images.new_ones((ood_images.shape[0], num_classes))
    ood_target_precision = torch.sum(ood_target_alphas, dim=1, keepdim=True)
    ood_target_mean = ood_target_alphas / ood_target_precision
    return ood_target_mean, ood_target_precision