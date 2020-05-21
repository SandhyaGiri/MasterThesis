import math
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..datasets.adversarial_dataset import AdversarialDataset
from ..eval.model_prediction_eval import ClassifierPredictionEvaluator
from ..eval.uncertainty import UncertaintyEvaluator, UncertaintyMeasuresEnum
from ..utils.common_data import ATTACK_CRITERIA_MAP, OOD_ATTACK_CRITERIA_MAP
from ..utils.persistence import persist_image_dataset
from ..utils.pytorch import save_model_with_params_from_ckpt
from .trainer import PriorNetTrainer


def get_optimal_threshold(src_dir, uncertainty_measure: UncertaintyMeasuresEnum):
    id_uncertainty = np.loadtxt(os.path.join(src_dir,
                                             'id_' + uncertainty_measure._value_ + '.txt'))
    ood_uncertainty = np.loadtxt(os.path.join(src_dir,
                                              'ood_' + uncertainty_measure._value_ + '.txt'))
    target_labels = np.concatenate((np.zeros_like(id_uncertainty),
                                    np.ones_like(ood_uncertainty)), axis=0)
    decision_fn_value = np.concatenate((id_uncertainty, ood_uncertainty), axis=0)
    if uncertainty_measure == UncertaintyMeasuresEnum.CONFIDENCE:
        decision_fn_value *= -1.0
    fpr, tpr, thresholds = roc_curve(target_labels, decision_fn_value)
    opt_fn_value = (tpr - fpr)
    indices = np.argwhere(opt_fn_value == np.amax(opt_fn_value)).flatten().tolist()
    median_indices = []
    num_optimal_thresholds = len(indices)
    if num_optimal_thresholds % 2 == 0:
        index1 = indices[math.floor(num_optimal_thresholds/2)]
        index2 = indices[math.floor(num_optimal_thresholds/2) -1]
        if fpr[index2] < fpr[index1]:
            median_indices.append(index2)
        else:
            median_indices.append(index1)
    else:
        median_indices.append(indices[math.floor(num_optimal_thresholds/2)])
    # log the TPR and FPR at these thresholds
    for ind in median_indices:
        print("Max fn value: ", opt_fn_value[ind])
        print(f"Threshold: {thresholds[ind]}, TPR: {tpr[ind]}, FPR: {fpr[ind]}")
    return np.mean(thresholds[median_indices])

class AdversarialPriorNetTrainer(PriorNetTrainer):
    """
    Priornet trainer which performs a training epoch with normal images
    and additional training epoch with adversarial images generated from
    the normal images.
    """
    def __init__(self, model, id_train_dataset, id_val_dataset,
                 ood_train_dataset, ood_val_dataset,
                 criterion, id_criterion, ood_criterion, optimizer_fn, 
                 adv_attack_type, adv_attack_criteria,
                 optimizer_params: Dict[str, Any] = {},
                 lr_scheduler=None,
                 lr_scheduler_params={},
                 batch_size=64, patience=20, device=None, clip_norm=10.0, num_workers=4,
                 pin_memory=False, log_dir='.',
                 attack_params: Dict[str, Any] = {},
                 dataset_persistence_params=[],
                 adv_training_type: str='normal',
                 uncertainty_measure: UncertaintyMeasuresEnum=UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY):
        """
        for "ood-detect adversarial training, we need to know the uncertainty_measure used to the binary
        classification between in-domain and out-domain samples."
        """
        assert adv_training_type in ['normal', 'ood-detect']
        
        super(AdversarialPriorNetTrainer, self).__init__(model, id_train_dataset,
                                                         id_val_dataset, ood_train_dataset,
                                                         ood_val_dataset, criterion,
                                                         id_criterion, ood_criterion,
                                                         optimizer_fn, optimizer_params,
                                                         lr_scheduler, lr_scheduler_params,
                                                         batch_size, patience, device,
                                                         clip_norm, num_workers,
                                                         pin_memory, log_dir,
                                                         adv_training_type == 'ood-detect')
        self.id_train_dataset = id_train_dataset
        self.ood_train_dataset = ood_train_dataset
        self.batch_size = batch_size
        self.adv_attack_type = adv_attack_type
        self.adv_id_attack_criteria = ATTACK_CRITERIA_MAP[adv_attack_criteria]
        self.adv_ood_attack_criteria = OOD_ATTACK_CRITERIA_MAP[adv_attack_criteria]
        self.attack_params = attack_params
        self.dataset_persistence_params = dataset_persistence_params
        self.adv_training_type = adv_training_type
        self.uncertainty_measure = uncertainty_measure

    def train(self, num_epochs=None, num_steps=None, resume=False):
        """
        Provide either num_epochs, or num_steps indicating total number of training
        steps to be performed.
        """
        if num_epochs is None:
            assert isinstance(num_steps, int)
            num_epochs = math.ceil(num_steps / len(self.id_train_loader))
        else:
            assert isinstance(num_epochs, int)

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            self.epochs = epoch
            ###################
            # train the model #
            ###################
            start = time.time()
            # train and update metrics list
            train_results = self._train_single_epoch(self.id_train_loader,
                                                     self.ood_train_loader)
            adv_train_results = self._adv_train_single_epoch() # train with adversarial images
            end = time.time()

            ######################
            # validate the model #
            ######################
            val_results = self._val_single_epoch() # validate and update metrics list

            # store and print epoch summary
            summary = {
                'train_results': train_results,
                'adv_train_results': adv_train_results,
                'val_results': val_results,
                'time_taken': np.round(((end-start) / 60.0), 2),
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f"Train loss: {train_results['loss']}, \
                  Train accuracy: {train_results['id_accuracy']}")
            print(f"Adv Train loss: {adv_train_results['loss']}, \
                  Adv Train accuracy: {adv_train_results['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']} mins")

            # step through lr scheduler
            self.lr_scheduler.step()

        save_model_with_params_from_ckpt(self.model, self.log_dir)

    def _adv_train_single_epoch(self):
        """
        Generates adversarial dataset and trains the model on them.
        """
        if self.adv_training_type == "normal":
            additional_args = {''} # TODO - complete it
        elif self.adv_training_type == "ood-detect":
            threshold = get_optimal_threshold(os.path.join(self.log_dir,
                                                           f'epoch-{self.epochs+1}-uncertainties'),
                                              self.uncertainty_measure)
            additional_args = {'only_true_adversaries': True,
                               'uncertainty_measure': self.uncertainty_measure,
                               'uncertainty_threshold': threshold
                               }
        id_train_adv_set = AdversarialDataset(self.id_train_dataset,
                                              self.adv_attack_type.lower(),
                                              self.model,
                                              self.attack_params['epsilon'],
                                              self.adv_id_attack_criteria,
                                              self.attack_params['norm'],
                                              self.attack_params['step_size'],
                                              self.attack_params['max_steps'],
                                              batch_size=self.batch_size,
                                              device=self.device,
                                              adv_success_detect_type=self.adv_training_type,
                                              **additional_args)
        ood_train_adv_set = AdversarialDataset(self.ood_train_dataset,
                                               self.adv_attack_type.lower(),
                                               self.model,
                                               self.attack_params['epsilon'],
                                               self.adv_ood_attack_criteria,
                                               self.attack_params['norm'],
                                               self.attack_params['step_size'],
                                               self.attack_params['max_steps'],
                                               batch_size=self.batch_size,
                                               device=self.device,
                                               adv_success_detect_type=self.adv_training_type,
                                               ood_dataset=True,
                                               **additional_args)
        print(f"Number of in->out adversarials generated: {len(id_train_adv_set)}")
        print(f"Number of out->in adversarials generated: {len(ood_train_adv_set)}")
        # persist images. as model progresses in training, attack gradients
        # will differ and we should get different adv images
        if self.attack_params['adv_persist_images']:
            # persist only random sampled 100 images
            indices = np.arange(len(id_train_adv_set))

            id_train_adv_subset = data.Subset(id_train_adv_set, random.sample(list(indices), min(100, len(indices))))
            adv_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-adv-images')
            os.makedirs(adv_dir)
            persist_image_dataset(id_train_adv_subset,
                                  *self.dataset_persistence_params,
                                  adv_dir)

            indices = np.arange(len(ood_train_adv_set))
            ood_train_adv_subset = data.Subset(ood_train_adv_set, random.sample(list(indices), min(100, len(indices))))
            adv_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-adv-images-ood')
            os.makedirs(adv_dir)
            persist_image_dataset(ood_train_adv_subset,
                                  *self.dataset_persistence_params,
                                  adv_dir)

        # Dataloaders for adv train dataset
        id_train_adv_loader = None
        if len(id_train_adv_set) > 0:
            id_train_adv_loader = DataLoader(id_train_adv_set,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=self.num_workers,
                                             pin_memory=self.pin_memory)

        ood_train_adv_loader = None
        if len(ood_train_adv_set) > 0:
            ood_train_adv_loader = DataLoader(ood_train_adv_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=self.num_workers,
                                              pin_memory=self.pin_memory)

        return self._train_single_epoch_adversarial(id_train_adv_loader,
                                        ood_train_adv_loader)
        
    def _train_single_epoch_adversarial(self, id_train_loader, ood_train_loader):
        """
        Overriding normal training epoch, as for adversarial datasets
        we could have imbalanced id and ood datasets or sometimes no
        adversarial for in->out or out->in.
        """
        # Set model in train mode
        self.model.train()

        # metrics to be collected
        accuracies = 0.0
        kl_loss = 0.0
        id_loss, ood_loss = 0.0, 0.0
        id_precision, ood_precision = 0.0, 0.0
        id_outputs_all = None
        ood_outputs_all = None
        
        if id_train_loader is not None:
            # train on id samples separately
            for i, (data) in enumerate(id_train_loader, 0):
                # Get inputs
                inputs, labels = data
                # print("In domain tensor shape: ", inputs.shape)
                # print("Out domain tensor shape: ", ood_inputs.shape)
                if self.device is not None:
                    inputs, labels = map(lambda x: x.to(self.device,
                                                        non_blocking=self.pin_memory),
                                        (inputs, labels))

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # eval id_samples
                id_outputs = self.model(inputs)
                # accumulate all outputs by the model
                if id_outputs_all is None:
                    id_outputs_all = id_outputs
                else:
                    id_outputs_all = torch.cat((id_outputs_all, id_outputs), dim=0)

                # Calculate train loss (only ID loss)
                loss = self.id_criterion(id_outputs, labels)
                assert torch.all(torch.isfinite(loss)).item()
                kl_loss += loss.item()

                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

                # precision of the dirichlet dist outputed by the model (id), 
                # averaged across all samples in batch
                id_alphas = torch.exp(id_outputs)
                id_precision += torch.mean(torch.sum(id_alphas, dim=1)).item()

                probs = F.softmax(id_outputs, dim=1)
                accuracy = ClassifierPredictionEvaluator.compute_accuracy(probs, labels,
                                                                        self.device).item()
                accuracies += accuracy

            # average the metrics over all steps (batches) in this epoch
            num_batches = len(self.id_train_loader)
            accuracies /= num_batches
            kl_loss /= num_batches
            id_precision /= num_batches

            if self.log_uncertainties:
                id_uncertainties = UncertaintyEvaluator(id_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
                uncertainty_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-uncertainties-adv')
                if not os.path.exists(uncertainty_dir):
                    os.makedirs(uncertainty_dir)
                for key in id_uncertainties.keys():
                    np.savetxt(os.path.join(uncertainty_dir, 'id_' + key._value_ + '.txt'),
                            id_uncertainties[key])
        id_loss = kl_loss
        kl_loss = 0.0

        if ood_train_loader is not None:
            # train on ood samples separately
            for i, (ood_data) in enumerate(ood_train_loader, 0):
                # Get inputs
                ood_inputs, _ = ood_data
                if self.device is not None:
                    ood_inputs = ood_inputs.to(self.device, non_blocking=self.pin_memory)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # eval ood_samples
                ood_outputs = self.model(ood_inputs)
                # accumulate all outputs by the model
                if ood_outputs_all is None:
                    ood_outputs_all = ood_outputs
                else:
                    ood_outputs_all = torch.cat((ood_outputs_all, ood_outputs), dim=0)

                # Calculate train loss (only OOD loss)
                loss = self.ood_criterion(ood_outputs, None)
                assert torch.all(torch.isfinite(loss)).item()
                kl_loss += loss.item()

                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

                # precision of the dirichlet dist outputed by the model (ood),
                # averaged across all samples in batch
                ood_alphas = torch.exp(ood_outputs)
                ood_precision += torch.mean(torch.sum(ood_alphas, dim=1)).item()

            # average the metrics over all steps (batches) in this epoch
            num_batches = len(self.ood_train_loader)
            kl_loss /= num_batches
            ood_precision /= num_batches

            if self.log_uncertainties:
                ood_uncertainties = UncertaintyEvaluator(ood_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
                uncertainty_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-uncertainties-adv')
                if not os.path.exists(uncertainty_dir):
                    os.makedirs(uncertainty_dir)
                for key in ood_uncertainties.keys():
                    np.savetxt(os.path.join(uncertainty_dir, 'id_' + key._value_ + '.txt'),
                            ood_uncertainties[key])
        ood_loss = kl_loss
        # for adv epoch, we cannot compute overall PriorNetWeightedLoss having both id and ood loss
        kl_loss = 0.0
        # returns average metrics (loss, accuracy, dirichlet_dist_precision)
        return {
            'loss': np.round(kl_loss, 4),
            'id_loss': np.round(id_loss, 4),
            'ood_loss': np.round(ood_loss, 4),
            'id_accuracy': np.round(100.0 * accuracies, 2),
            'id_precision': np.round(id_precision, 4),
            'ood_precision': np.round(ood_precision, 4)
        }
