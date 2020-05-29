"""
This module contains the VGG based convolution model.
"""
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -1 * Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

class SmoothedPriorNet(nn.Module):
    def __init__(self, base_classifier: nn.Module,
                 n_in: int,
                 n_out: int,
                 num_channels: int,
                 image_normalization_params: dict,
                 drop_rate: float,
                 noise_std_dev: float,
                 num_mc_samples: int, **kwargs): # num_mc_samples not used now, remove later
        super(SmoothedPriorNet, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = n_out
        self.image_normalization_params = image_normalization_params
        self.noise_std_dev = noise_std_dev
        self.epsilon = 1e-8
        self.num_samples = num_mc_samples
        # initial number of samples generated after a few fc layers - NOT USED NOW
        fc_layers = [500, 250, 100, 50, 25]
        layers = []
        prev_layer_size = n_in * n_in * num_channels
        for layer in fc_layers:
            layers += [nn.Linear(prev_layer_size, layer),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=drop_rate)]
            prev_layer_size = layer

        # last layer for num_samples unit
        layers += [nn.Linear(fc_layers[-1], 1)]
        self.samples_layer = nn.Sequential(*layers)

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
        #for i in range(len(counts)):
        #    counts[i] += len((preds == i).nonzero())
        counts += torch.sum(preds, dim=0)

    def _eval_on_base_classifier(self, inputs):
        logits = self.base_classifier(inputs)
        # probs = torch.nn.functional.softmax(logits, dim=1)
        # preds = torch.argmax(probs, dim=1)
        preds = gumbel_softmax(logits, 0.01) # one hot vector
        return preds

    def _get_count_vector(self, image: torch.tensor, num_samples: int, batch_size: int):
        num_classes = self.num_classes
        # num_samples = int(torch.ceil(num_samples).item())
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

    def forward(self, x, in_domain=True):
        """
        For each input, returns a log(count vector), where count vector
        is a vector of class label counts accumulated during MC Sampling.
        """
        batch_size = x.shape[0]
        # flatten the image) - NOT USED NOW, num_samples is fixed as of now.
        # (Batch_size, 1) each with num_samples to be generated.
        # samples = self.samples_layer(x.mean(dim=1).view(x.shape[0], -1))

        # samples = self.num_samples if in_domain else self.num_classes

        samples = self.num_samples
        outputs = []
        for i in range(batch_size):
            image = x[i] # (C, H, W)
            # eval on gaussian perturbed inputs
            count_vector = self._get_count_vector(image, samples, batch_size)
            # eval on original input
            org_input_pred = self._eval_on_base_classifier(image.unsqueeze(0))
            self._update_count(count_vector, org_input_pred)

            count_vector = count_vector.unsqueeze(1)
            outputs.append(count_vector)
        return torch.log(torch.cat(outputs, dim=1).transpose(0,1))
