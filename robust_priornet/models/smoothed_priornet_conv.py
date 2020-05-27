"""
This module contains the VGG based convolution model.
"""
import torch
import torch.nn as nn
import numpy as np

class SmoothedPriorNet(nn.Module):
    def __init__(self, base_classifier: nn.Module,
                 n_out: int,
                 image_normalization_params: dict,
                 noise_std_dev: float,
                 num_mc_samples: int, **kwargs):
        super(SmoothedPriorNet, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = n_out
        self.image_normalization_params = image_normalization_params
        self.noise_std_dev = noise_std_dev
        self.num_samples = num_mc_samples
        self.epsilon = 1e-8

    def _normalize_image(self, inputs: torch.tensor):
        mean = inputs.new_tensor(self.image_normalization_params['mean'])
        mean = mean.repeat(inputs.shape[0], inputs.shape[2],
                           inputs.shape[3], 1) # (batch_size, H, W, 3)
        mean = mean.permute(0, 3, 1, 2)
        std = inputs.new_tensor(self.image_normalization_params['std'])
        std = std.repeat(inputs.shape[0], inputs.shape[2],
                         inputs.shape[3], 1) # (batch_size, H, W, 3)
        std = std.permute(0, 3, 1, 2)
        normalized_inputs = (inputs - mean) / std
        return normalized_inputs

    def _update_count(self, counts: torch.tensor, preds: torch.tensor):
        for i in range(len(counts)):
            counts[i] += len((preds == i).nonzero())

    def _eval_on_base_classifier(self, inputs):
        logits = self.base_classifier(inputs)
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def _get_count_vector(self, image: torch.tensor, num_samples: int, batch_size: int):
        num_classes = self.num_classes
        counts = torch.zeros((num_classes), requires_grad=True) + self.epsilon
        for _ in range(int(np.ceil(num_samples / batch_size))):
            this_batch_size = min(batch_size, num_samples)
            num_samples -= this_batch_size

            batch = image.repeat((this_batch_size, 1, 1, 1))
            # sample noise from normal dist
            noise = torch.randn_like(batch) * self.noise_std_dev
            predictions = self._eval_on_base_classifier(batch + noise)
            self._update_count(counts, predictions)
        return counts

    def forward(self, x):
        """
        For each input, returns a log(count vector), where count vector
        is a vector of class label counts accumulated during MC Sampling.
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            image = x[i] # (C, H, W)
            # eval on gaussian perturbed inputs
            count_vector = self._get_count_vector(image, self.num_samples, batch_size)
            # eval on original input
            org_input_pred = self._eval_on_base_classifier(image.unsqueeze(0))
            self._update_count(count_vector, org_input_pred)

            count_vector = count_vector.unsqueeze(1)
            outputs.append(count_vector)
        return torch.log(torch.cat(outputs, dim=1).transpose(0,1))
