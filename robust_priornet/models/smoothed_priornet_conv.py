"""
This module contains the VGG based convolution model.
"""
import math

import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB
from torch.autograd import Variable
from ..eval.uncertainty import UncertaintyEvaluatorTorch, UncertaintyMeasuresEnum


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -1 * Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
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

class SmoothedPriorNetCount(nn.Module):
    def __init__(self, base_classifier: nn.Module,
                 n_in: int,
                 n_out: int,
                 num_channels: int,
                 image_normalization_params: dict,
                 drop_rate: float,
                 noise_std_dev: float,
                 num_mc_samples: int, **kwargs):
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
        if counts is not None:
            # accumulate one hot preds
            return torch.sum(torch.cat((counts, preds), dim=0), dim=0, keepdim=True)
        return torch.sum(preds, dim=0, keepdim=True)

    def _eval_on_base_classifier(self, inputs):
        logits = self.base_classifier(inputs)
        # probs = torch.nn.functional.softmax(logits, dim=1)
        # preds = torch.argmax(probs, dim=1)
        preds = gumbel_softmax(logits, 0.01) # one hot vector
        return preds

    def _get_count_vector(self, image: torch.tensor, num_samples: int, batch_size: int):
        num_classes = self.num_classes
        # num_samples = int(torch.ceil(num_samples).item())
        # counts = torch.zeros((num_classes), requires_grad=True) + self.epsilon
        counts = None
        for _ in range(int(np.ceil(num_samples / batch_size))):
            this_batch_size = min(batch_size, num_samples)
            num_samples -= this_batch_size

            batch = image.repeat((this_batch_size, 1, 1, 1))
            # sample noise from normal dist
            noise = torch.randn_like(batch) * self.noise_std_dev
            predictions = self._eval_on_base_classifier(batch + noise)
            counts = self._update_count(counts, predictions)
        return counts

    def forward(self, x):
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
            count_vector = self._update_count(count_vector, org_input_pred)

            count_vector += self.epsilon
            outputs.append(count_vector)
        return torch.log(torch.cat(outputs, dim=0))


class SmoothedPriorNetSimple(nn.Module):
    def __init__(self, base_classifier: nn.Module,
                 n_in: int,
                 n_out: int,
                 num_channels: int,
                 image_normalization_params: dict,
                 drop_rate: float,
                 noise_std_dev: float,
                 num_mc_samples: int, **kwargs):
        super(SmoothedPriorNetSimple, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = n_out
        self.image_normalization_params = image_normalization_params
        self.noise_std_dev = noise_std_dev
        self.epsilon = 1e-8
        self.num_samples = num_mc_samples

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

    def _accumulate_logits(self, curr_logits: torch.tensor, logits: torch.tensor):
        if curr_logits is not None:
            return torch.sum(torch.cat((curr_logits, logits), dim=0), dim=0, keepdim=True)
        return torch.sum(logits, dim=0, keepdim=True)

    def _eval_on_base_classifier(self, inputs):
        logits = self.base_classifier(inputs)
        return logits

    def _do_rand_smoothing(self, image: torch.tensor, num_samples: int, batch_size: int):
        overall_logits = None
        for _ in range(int(np.ceil(num_samples / batch_size))):
            this_batch_size = min(batch_size, num_samples)
            num_samples -= this_batch_size

            batch = image.repeat((this_batch_size, 1, 1, 1))
            # sample noise from normal dist
            noise = torch.randn_like(batch) * self.noise_std_dev
            logits = self._eval_on_base_classifier(batch + noise)
            overall_logits = self._accumulate_logits(overall_logits, logits)
        return overall_logits

    def forward(self, x):
        """
        For each input, returns a mean(logits), where logits are the logits
        outputted by the model for gaussian noise perturbed samples during MC Sampling.
        """
        batch_size = x.shape[0]

        samples = self.num_samples
        outputs = []
        for i in range(batch_size):
            image = x[i] # (C, H, W)
            # eval on gaussian perturbed inputs
            overall_logits = self._do_rand_smoothing(image, samples, batch_size)
            # eval on original input
            org_input_logits = self._eval_on_base_classifier(image.unsqueeze(0))
            overall_logits = self._accumulate_logits(overall_logits, org_input_logits)

            overall_logits = torch.div(overall_logits, samples+1)
            outputs.append(overall_logits)
        return torch.cat(outputs, dim=0)

class SmoothedPriorNet(nn.Module):
    """
    Simply has a noise layer during training which converts input images into
    a gaussian noise perturbed image (1 noisy image for every input image).
    During eval, several noise draws are used to make a final prediction.
    """
    def __init__(self, base_classifier: nn.Module,
                 n_in: int,
                 n_out: int,
                 num_channels: int,
                 image_normalization_params: dict,
                 drop_rate: float,
                 noise_std_dev: float,
                 num_mc_samples: int, 
                 reduction_method, **kwargs):
        super(SmoothedPriorNet, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = n_out
        self.image_normalization_params = image_normalization_params
        self.noise_std_dev = noise_std_dev
        self.epsilon = 1e-8
        self.num_samples = num_mc_samples
        self.reduction_method = reduction_method
        # needed only for count based logit reduction.
        self.uncertainty_measures_thresholds = kwargs.get('uncertainty_measures_thresholds')
        self.max_alpha_threshold = kwargs.get('max_alpha_threshold')
    
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

    def _noise_layer(self, inputs: torch.tensor):
        # add gaussian noise to entire batch
        noise = torch.randn_like(inputs) * self.noise_std_dev
        inputs = inputs + noise
        # normalize the images
        return self._normalize_image(inputs)

    def forward(self, x):
        """
        Used during training/validation phase
        """
        x = self._noise_layer(x)
        x = self.base_classifier(x) # train on noisy images
        return x

    def _do_rand_smoothing(self, image: torch.tensor, num_samples: int, batch_size: int):
        overall_logits = []
        for _ in range(int(np.ceil(num_samples / batch_size))):
            this_batch_size = min(batch_size, num_samples)
            num_samples -= this_batch_size

            batch = image.repeat((this_batch_size, 1, 1, 1))
            # sample noise from normal dist
            noise = torch.randn_like(batch) * self.noise_std_dev
            # normalize the input images
            normalized_inputs = self._normalize_image(batch + noise)
            logits = self.base_classifier(normalized_inputs)
            overall_logits.append(logits)
        return overall_logits

    def _log_cosh_reduction(self, logits):
        n = logits.shape[0] # number of samples
        u = logits.new_zeros((1, self.num_classes))
        for c in range(self.num_classes):
            z_c = logits[:,c]
            try:
                # Create a new model
                m = gp.Model("logcosh-opt")
                print("created new model")
                # to turn off unnecessary comments being logged during optimization
                m.setParam('OutputFlag', False)
                # Add variables
                u_c = m.addVar(name='uc', lb=-GRB.INFINITY, ub=GRB.INFINITY)
                y = m.addVars(n, name='y', lb=-GRB.INFINITY) # log cosh value across n samples
                x = m.addVars(n, name='x', lb=-GRB.INFINITY) # z_ic - u_c
                xneg = m.addVars(n, name='xneg', lb=-GRB.INFINITY) # -1 * (z_ic - u_c)
                exp1 = m.addVars(n, name='exp1', lb=-GRB.INFINITY)
                exp2 = m.addVars(n, name='exp2', lb=-GRB.INFINITY)
                cosh_result = m.addVars(n, name='cosh_result', lb=0)
                # set the non-linear constraints which make up the final log cosh loss fn
                for i in range(n):
                    m.addLConstr(z_c[i].item() - u_c, GRB.EQUAL, x[i])
                    m.addLConstr(-1 * (z_c[i].item() - u_c), GRB.EQUAL, xneg[i])
                    m.addGenConstrExp(x[i], exp1[i])
                    m.addGenConstrExp(xneg[i], exp2[i])
                    m.addLConstr((exp1[i] + exp2[i])/2, GRB.EQUAL, cosh_result[i])
                    m.addGenConstrLog(cosh_result[i], y[i])
                # set objective and optimize
                m.setObjective(y.sum(), GRB.MINIMIZE)
                m.optimize()
                
                # get the solution and print it
                obj = m.getObjective()
                print(f"The objective value is: {obj.getValue()}")
                v = m.getVars()
                print(f"Optimal u_c: {v[0].varName} {v[0].x}")
                u[:, c] = v[0].x # storing value of u_c variable which is the minimizer
                
                # Status checking
                status = m.Status
                if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
                    print("The model cannot be solved because it is infeasible or "
                        "unbounded")
                if status != GRB.OPTIMAL:
                    print('Optimization was stopped with status ' + str(status))
            except gp.GurobiError as e:
                print('Error code ' + str(e.errno) + ': ' + str(e))
            except AttributeError:
                print('Encountered an attribute error')
            except:
                print("some exception occured!!")
                print(e)
        return u

    def _count_reduction(self, logits):
        # make predictions on all logits
        alphas = torch.exp(logits)
        alpha0 = torch.sum(alphas, dim=1)
        probs = alphas/alpha0
        preds = torch.argmax(probs, dim=1)
        # consider only those predictions which are valid as per restrictions
        valid_indices = np.arange(0, preds.shape[0])
        if self.max_alpha_threshold > 0:
            good_max_alpha_indices = (torch.max(alphas, dim=1)[0] >= self.max_alpha_threshold).nonzero().squeeze(1)
            valid_indices = np.intersect1d(valid_indices, good_max_alpha_indices)
        
        for un_measure in self.uncertainty_measures_thresholds.keys():
            enum = UncertaintyMeasuresEnum.get_enum(un_measure)
            uncertainty_values = UncertaintyEvaluatorTorch(logits).get_uncertainty(enum, negate_confidence=True).squeeze(1)
            good_uncertainty_indices = (uncertainty_values < self.uncertainty_measures_thresholds[un_measure]).nonzero().squeeze(1)
            valid_indices = np.intersect1d(valid_indices, good_uncertainty_indices)
        invalid_indices = np.setdiff1d(np.arange(0, preds.shape[0]), valid_indices)
        preds[(invalid_indices)] = -1 # these predictions will be dropped
        # return the log(count vector), as when we take exp again to get alpha we get back count vector as the dir params ?
        counts = torch.zeros((1, self.num_classes), dtype=float)
        counts.add_(self.epsilon)
        for i in range(len(counts)):
            counts[0, i] += (preds == i).nonzero().shape[0]
        return torch.log(counts)

    def test(self, x):
        """
        Used during testing phase
        """
        assert self.reduction_method in ['mean', 'median', 'log_cosh', 'count']
        batch_size = x.shape[0]

        samples = self.num_samples
        outputs = []
        for i in range(batch_size):
            image = x[i] # (C, H, W)
            # eval on gaussian perturbed inputs
            overall_logits = self._do_rand_smoothing(image, samples, batch_size)
            # eval on original input
            # org_input_logits = self._eval_on_base_classifier(image.unsqueeze(0)) ?
            # overall_logits.append(org_input_logits)

            overall_logits = torch.cat(overall_logits, dim=0)
            if self.reduction_method == 'mean':
                overall_logits = torch.mean(overall_logits, dim=0, keepdim=True)
            elif self.reduction_method == 'median':
                overall_logits = torch.median(overall_logits, dim=0, keepdim=True)[0]
            elif self.reduction_method == 'log_cosh':
                overall_logits = self._log_cosh_reduction(overall_logits)
            elif self.reduction_method == 'count':
                overall_logits = self._count_reduction(overall_logits)
            outputs.append(overall_logits)
        return torch.cat(outputs, dim=0)
