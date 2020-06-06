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
from ..utils.pytorch import load_model, save_model_with_params_from_ckpt
from .early_stopping import EarlyStopper
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
                 batch_size=64,
                 min_epochs=25, patience=20,
                 device=None, clip_norm=10.0, num_workers=4,
                 pin_memory=False, log_dir='.',
                 attack_params: Dict[str, Any] = {},
                 dataset_persistence_params=[],
                 adv_training_type: str = 'normal',
                 uncertainty_measure: UncertaintyMeasuresEnum = UncertaintyMeasuresEnum.DIFFERENTIAL_ENTROPY,
                 use_fixed_threshold=False, known_threshold_value=0.0,
                 only_out_in_adversarials: bool = False):
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
                                                         batch_size, min_epochs, patience, device,
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
        self.only_out_in_adversarials = only_out_in_adversarials
        self.use_fixed_threshold = use_fixed_threshold
        self.known_threshold_value = known_threshold_value

    def train(self, num_epochs=None, num_steps=None, resume=False, ckpt=None):
        """
        Provide either num_epochs, or num_steps indicating total number of training
        steps to be performed.
        """
        if num_epochs is None:
            assert isinstance(num_steps, int)
            num_epochs = math.ceil(num_steps / len(self.id_train_loader))
        else:
            assert isinstance(num_epochs, int)

        assert resume is False or ckpt is not None
        init_epoch = 0
        if resume is True:
            init_epoch = ckpt['epochs'] + 1
            self.optimizer.load_state_dict(ckpt['opt_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            print(f"Model restored from checkpoint at epoch {init_epoch}")

        # initialize the early_stopping object
        early_stopping = EarlyStopper(self.min_epochs, self.patience, verbose=True)

        for epoch in range(init_epoch, num_epochs):
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

            # save the checkpoint every epoch
            save_model_with_params_from_ckpt(self.model, self.log_dir,
                                             name='checkpoint.tar',
                                             additional_params={
                                                 'epochs': self.epochs,
                                                 'opt_state_dict': self.optimizer.state_dict(),
                                                 'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
                                             })

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

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping.register_epoch(val_results['loss'], self.model, self.log_dir)

            if early_stopping.do_early_stop:
                print("Early stopping")
                self.training_early_stopped = True
                break

            # step through lr scheduler
            self.lr_scheduler.step()

        # load the last checkpoint with the best model
        if self.training_early_stopped:
            self.model, _ = load_model(self.log_dir,
                                    device=self.device,
                                    name=early_stopping.best_model_name)

        save_model_with_params_from_ckpt(self.model, self.log_dir)

    def _persist_adv_dataset(self, adv_dataset, dir_name):
        indices = np.arange(len(adv_dataset))
        adv_dataset = data.Subset(adv_dataset,
                                  random.sample(list(indices),
                                                min(100, len(indices))))
        adv_dir = os.path.join(self.log_dir, dir_name)
        os.makedirs(adv_dir)
        persist_image_dataset(adv_dataset,
                              *self.dataset_persistence_params,
                              adv_dir)

    def _generate_adversarial_dataset(self):
        """
        Creates an adversarial dataset based on the current trained model.
        For ood-detect adv training, generates both in->out adversarials and
        out->in adversarials. While for normal adv training, generates only
        misclassification adversarials on in domain dataset.

        Also persists the generated adv images, if needed.
        """
        id_train_adv_set = []
        ood_train_adv_set = []
        if self.adv_training_type == "normal":
                # only works on in domain samples which get misclassified under attack
            additional_args = {'only_true_adversaries': True}
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
            print(f"Number of misclassification adversarials generated: {len(id_train_adv_set)}")
            if self.attack_params['adv_persist_images']:
                # persist only random sampled 100 images
                self._persist_adv_dataset(id_train_adv_set, f'epoch-{self.epochs+1}-adv-images')
        elif self.adv_training_type == "ood-detect":
            if not self.use_fixed_threshold:
                threshold = get_optimal_threshold(os.path.join(self.log_dir,
                                                               f'epoch-{self.epochs+1}-uncertainties'),
                                                  self.uncertainty_measure)
            else:
                threshold = self.known_threshold_value
            additional_args = {'only_true_adversaries': True,
                               'uncertainty_measure': self.uncertainty_measure,
                               'uncertainty_threshold': threshold
                               }
            if not self.only_out_in_adversarials:
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
                                                      adv_success_detect_type=
                                                      self.adv_training_type,
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
                self._persist_adv_dataset(id_train_adv_set, f'epoch-{self.epochs+1}-adv-images')
                self._persist_adv_dataset(ood_train_adv_set, f'epoch-{self.epochs+1}-adv-images-ood')
            
        return id_train_adv_set, ood_train_adv_set

    def _get_adv_data_loaders(self, id_train_adv_set, ood_train_adv_set):
        id_train_adv_loader = None
        ood_train_adv_loader = None
        if len(id_train_adv_set) > 0:
            id_train_adv_loader = DataLoader(id_train_adv_set,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=self.num_workers,
                                             pin_memory=self.pin_memory)

        if len(ood_train_adv_set) > 0:
            ood_train_adv_loader = DataLoader(ood_train_adv_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=self.num_workers,
                                              pin_memory=self.pin_memory)
        return id_train_adv_loader, ood_train_adv_loader

    def _adv_train_single_epoch(self):
        """
        Generates adversarial dataset and trains the model on them.
        """
        id_train_adv_set, ood_train_adv_set = self._generate_adversarial_dataset()
        # Dataloaders for adv train dataset
        id_train_adv_loader, ood_train_adv_loader = self._get_adv_data_loaders(id_train_adv_set,
                                                                               ood_train_adv_set)

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
            print("Training on in->out adversarial images")
            for i, (data) in enumerate(id_train_loader, 0):
                adv_batch_result = self._train_single_batch_criteria(data,
                                                                     self.id_criterion,
                                                                     is_ood=False)
                loss, precision, accuracy, id_outputs = adv_batch_result

                # accumulate all outputs by the model
                if id_outputs_all is None:
                    id_outputs_all = id_outputs
                else:
                    id_outputs_all = torch.cat((id_outputs_all, id_outputs), dim=0)

                kl_loss += loss
                id_precision += precision
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
            print("Training on out->in adversarial images")
            for i, (ood_data) in enumerate(ood_train_loader, 0):
                adv_batch_result = self._train_single_batch_criteria(ood_data,
                                                                     self.ood_criterion,
                                                                     is_ood=True)
                loss, precision, _ , ood_outputs = adv_batch_result

                # accumulate all outputs by the model
                if ood_outputs_all is None:
                    ood_outputs_all = ood_outputs
                else:
                    ood_outputs_all = torch.cat((ood_outputs_all, ood_outputs), dim=0)

                kl_loss += loss
                ood_precision += precision

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
                    np.savetxt(os.path.join(uncertainty_dir, 'ood_' + key._value_ + '.txt'),
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
