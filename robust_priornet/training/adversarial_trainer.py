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
from .early_stopping import EarlyStopper, EarlyStopperSteps
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
    # filter only tpr>=0.5 and fpr<=0.5 (if it goes beyond then we might have biased model which trains on false adversarials)
    #good_tpr_indices = np.argwhere(tpr >= 0.5)
    #good_fpr_indices = np.argwhere(fpr <= 0.5)
    #considerable_indices = np.intersect1d(good_tpr_indices, good_fpr_indices)
    #opt_fn_value = opt_fn_value[considerable_indices]
    indices = []
    sorted_opt_fn_values = np.argsort(opt_fn_value)
    print(sorted_opt_fn_values)
    argmax = sorted_opt_fn_values[-1]
    print(f"argmax: {argmax}, threshold: {thresholds[argmax]}, tpr: {tpr[argmax]}, fpr: {fpr[argmax]}")
    indices.append(argmax)
    # find next index larger than argmax
    secondmax = -1
    for i in range(2, len(opt_fn_value)):
        secondmax = sorted_opt_fn_values[-i]
        # print(f"threshold: {thresholds[secondmax]}, tpr: {tpr[secondmax]}, fpr: {fpr[secondmax]}")
        # use this second argmax only when TPR and FPR is in a valid range
        if secondmax > argmax and abs(tpr[secondmax] - tpr[argmax]) < 0.4 and abs(fpr[secondmax] - fpr[argmax]) < 0.4:
            indices.append(secondmax)
            print(f"second max: {secondmax}, threshold: {thresholds[secondmax]}, tpr: {tpr[secondmax]}, fpr: {fpr[secondmax]}")
            break
    #return np.mean((thresholds[secondmax], np.mean(thresholds[indices]))) # picking a threshold closer to in domain data
    return np.mean(thresholds[indices])

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
                 add_ce_loss=False,
                 ce_weight=0.5,
                 batch_size=64,
                 min_epochs=25, patience=20,
                 device=None, clip_norm=10.0, num_workers=0,
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
                                                         add_ce_loss, ce_weight,
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

    def _get_inputs_from_dataset(self, dataset):
        length = len(dataset)
        inputs = []
        for i in range(length):
            inputs.append(dataset[i][0].unsqueeze(0))
        return torch.cat(inputs, dim=0)

    def train_stepwise_mixed_batch(self, num_epochs, val_after_steps, resume=False, ckpt=None):
        """
        Does batch mixing: Generates adv images 10% of samples in each batch and replaces the
        corresponding normal images with adv images. Then the model is trained on this new batch
        containing 90% original images and 10% of perturbed/adversarial samples.
        """
        assert resume is False or ckpt is not None
        init_epoch = 0
        if resume is True:
            init_epoch = ckpt['epochs'] + 1
            self.optimizer.load_state_dict(ckpt['opt_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            print(f"Model restored from checkpoint at epoch {init_epoch}")
        
        # initialize the early_stopping object - consider min_epochs as steps here
        early_stopping = EarlyStopperSteps(self.min_epochs, self.patience, val_after_steps, verbose=True)
        # metrics to be collected (for every val steps)
        step_accuracies = 0.0
        step_kl_loss = 0.0
        step_id_loss, step_ood_loss = 0.0, 0.0
        step_id_precision, step_ood_precision = 0.0, 0.0
        last_validated_step = 0
        # overflow from previous epoch
        overflow = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        id_adv_dataset = None
        ood_data_dataset = None
        for epoch in range(init_epoch, num_epochs):
            self.epochs = epoch
            ###################
            # train the model #
            ###################
            epoch_start = time.time()
            # metrics to be collected (epoch_level)
            accuracies = 0.0
            total_kl_loss = 0.0
            total_id_loss, total_ood_loss = 0.0, 0.0
            total_id_precision, total_ood_precision = 0.0, 0.0
            num_steps_per_epoch = len(self.id_train_loader)
            id_outputs_all = None
            ood_outputs_all = None
            # iterators for the data loaders, as we don't have common num_batches across them
            id_train_iterator = iter(self.id_train_loader)
            ood_train_iterator = iter(self.ood_train_loader)
            
            # beginning of training epoch
            start = time.time()
            for i in range(num_steps_per_epoch): # each batch
                # choose 10% samples in each id, ood batches and generate adversarials
                id_data, id_labels = next(id_train_iterator)
                ood_data, ood_labels = next(ood_train_iterator)
                
                if epoch > 0: # only on starting second epoch
                    id_adv_end_index = int(0.1 * id_data.shape[0]) # 10% batch size
                    ood_adv_end_index = int(0.1 * ood_data.shape[0])
                    
                    id_first, id_second = id_data.split((id_adv_end_index, id_data.shape[0]-id_adv_end_index))
                    id_labels_first, id_labels_second = id_labels.split(((id_adv_end_index, id_data.shape[0]-id_adv_end_index)))
                    ood_first, ood_second = ood_data.split((ood_adv_end_index, ood_data.shape[0]-ood_adv_end_index))
                    ood_labels_first, ood_labels_second = ood_labels.split((ood_adv_end_index, ood_data.shape[0]-ood_adv_end_index))
                    
                    id_adv_dataset, ood_adv_dataset = self._generate_adversarial_dataset(id_dataset=
                                                                                data.TensorDataset(id_first, id_labels_first),
                                                                                ood_dataset=
                                                                                data.TensorDataset(ood_first, ood_labels_first),
                                                                                only_true_adversaries=
                                                                                False,
                                                                                previous_epoch=
                                                                                True)
                    id_data = torch.cat((self._get_inputs_from_dataset(id_adv_dataset), id_second), dim=0)
                    ood_data = torch.cat((self._get_inputs_from_dataset(ood_adv_dataset) if ood_adv_dataset != [] else ood_first,
                                          ood_second), dim=0)
                
                # train on these images
                batch_train_results = self._train_single_batch((id_data, id_labels),
                                                               (ood_data, ood_labels))
                kl_loss, id_loss, ood_loss, id_precision, ood_precision, accuracy, id_outputs, ood_outputs = batch_train_results
                
                # increment 1 step
                self.steps += 1

                # accumulate all outputs by the model
                if id_outputs_all is None:
                    id_outputs_all = id_outputs
                    ood_outputs_all = ood_outputs
                else:
                    id_outputs_all = torch.cat((id_outputs_all, id_outputs), dim=0)
                    ood_outputs_all = torch.cat((ood_outputs_all, ood_outputs), dim=0)

                # accumulate the metrics
                step_kl_loss += kl_loss
                step_id_loss += id_loss
                step_ood_loss += ood_loss
                step_id_precision += id_precision
                step_ood_precision += ood_precision
                step_accuracies += accuracy
                
                # validate on valset
                if self.steps % val_after_steps == 0:
                    # marks end of training
                    end = time.time()
                    last_validated_step = self.steps
                    step_val_results = self._val_single_epoch()
                    # accumulate epoch level metrics
                    total_kl_loss += step_kl_loss
                    total_id_loss += step_id_loss
                    total_ood_loss += step_ood_loss
                    total_id_precision += step_id_precision
                    total_ood_precision += step_ood_precision
                    accuracies += step_accuracies
                    # log step results so far
                    step_summary = {
                        'train_results': {
                            'loss': np.round(step_kl_loss/val_after_steps, 4),
                            'id_loss': np.round(step_id_loss/val_after_steps, 4),
                            'ood_loss': np.round(step_ood_loss/val_after_steps, 4),
                            'id_accuracy': np.round(100.0 * step_accuracies/val_after_steps, 2),
                            'id_precision': np.round(step_id_precision/val_after_steps, 4),
                            'ood_precision': np.round(step_ood_precision/val_after_steps, 4)
                        },
                        'val_results': step_val_results,
                        'time_taken': np.round(((end-start) / 60.0), 2),
                    }
                    torch.save(step_summary, os.path.join(self.log_dir, f'step_summary_{self.steps}.pt'))
                    print(f"Step {self.steps}:")
                    print(f"Train loss: {step_summary['train_results']['loss']}, \
                        Train accuracy: {step_summary['train_results']['id_accuracy']}")
                    print(f"Val loss: {step_val_results['loss']}, \
                        Val accuracy: {step_val_results['id_accuracy']}")
                    print(f"Time taken for train epoch: {step_summary['time_taken']} mins")
                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping.register_step(self.steps, step_val_results['loss'], self.model, self.log_dir)
                    
                    if early_stopping.do_early_stop:
                        print(f"Early stopping. Restoring model to step {early_stopping.best_step}")
                        self.training_early_stopped = True
                        break
                    # reset timer for next _ steps
                    start = time.time()
                    # reset step metric variables
                    step_accuracies = 0.0
                    step_kl_loss = 0.0
                    step_id_loss, step_ood_loss = 0.0, 0.0
                    step_id_precision, step_ood_precision = 0.0, 0.0

                # step through lr scheduler for batch level ones
                if self.lr_step_after_batch:
                    self.lr_scheduler.step()
            if self.training_early_stopped:
                break
            # accumulate epoch level metrics (subtract previous overflow)
            total_kl_loss -= overflow[0]
            total_id_loss -= overflow[1]
            total_ood_loss -= overflow[2]
            total_id_precision -= overflow[3]
            total_ood_precision -= overflow[4]
            accuracies -= overflow[5]
            #  (for last unaccounted val step)
            if self.steps > last_validated_step:
                total_kl_loss += step_kl_loss
                total_id_loss += step_id_loss
                total_ood_loss += step_ood_loss
                total_id_precision += step_id_precision
                total_ood_precision += step_ood_precision
                accuracies += step_accuracies
                # these values will be extra in the next epoch, so should be subtracted later
                overflow = (step_kl_loss, step_id_loss, step_ood_loss, step_id_precision, step_ood_precision, step_accuracies)
            # log uncertainties needed for ood-detect adv generation
            if self.log_uncertainties:
                id_uncertainties = UncertaintyEvaluator(id_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
                ood_uncertainties = UncertaintyEvaluator(ood_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
                uncertainty_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-uncertainties')
                os.makedirs(uncertainty_dir)
                for key in ood_uncertainties.keys():
                    np.savetxt(os.path.join(uncertainty_dir, 'ood_' + key._value_ + '.txt'),
                            ood_uncertainties[key])
                    np.savetxt(os.path.join(uncertainty_dir, 'id_' + key._value_ + '.txt'),
                            id_uncertainties[key])
            epoch_end = time.time()
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
                'train_results': {
                    'loss': np.round(total_kl_loss/num_steps_per_epoch, 4),
                    'id_loss': np.round(total_id_loss/num_steps_per_epoch, 4),
                    'ood_loss': np.round(total_ood_loss/num_steps_per_epoch, 4),
                    'id_accuracy': np.round(100.0 * accuracies/num_steps_per_epoch, 2),
                    'id_precision': np.round(total_id_precision/num_steps_per_epoch, 4),
                    'ood_precision': np.round(total_ood_precision/num_steps_per_epoch, 4)
                    },
                'val_results': val_results,
                'time_taken': np.round(((epoch_end-epoch_start) / 60.0), 2),
                'num_steps': self.steps, # number of steps completed when epoch finished
                #'num_in-out_adv': len(id_adv_dataset) if id_adv_dataset is not None else 0,
                #'num_out-in_adv': len(ood_adv_dataset) if ood_adv_dataset is not None else 0
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            print(f"Train loss: {summary['train_results']['loss']}, \
                  Train accuracy: {summary['train_results']['id_accuracy']}")
            # print(f"Adv Train loss: {adv_train_results['loss']}, \
            #      Adv Train accuracy: {adv_train_results['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']} mins")

            # step through lr scheduler
            if not self.lr_step_after_batch:
                self.lr_scheduler.step()

        # load the last checkpoint with the best model
        if self.training_early_stopped:
            self.model, _ = load_model(self.log_dir,
                                    device=self.device,
                                    name=early_stopping.best_model_name)

        save_model_with_params_from_ckpt(self.model, self.log_dir)

    def train_stepwise(self, num_epochs, val_after_steps, resume=False, ckpt=None):
        """
        Does batch interleaving: Generates adv images after first epoch on entire dataset,
        starting second epoch, trains the model on one batch of normal images (shuffled) and
        another batch of adv images generated previously (again shuffled) if present. 
        """
        assert resume is False or ckpt is not None
        init_epoch = 0
        if resume is True:
            init_epoch = ckpt['epochs'] + 1
            self.optimizer.load_state_dict(ckpt['opt_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            print(f"Model restored from checkpoint at epoch {init_epoch}")

        # initialize the early_stopping object - consider min_epochs as steps here
        early_stopping = EarlyStopperSteps(self.min_epochs, self.patience, val_after_steps, verbose=True)

        # initialize adv datasets and dataloaders
        id_adv_dataset = None
        ood_adv_dataset = None
        id_adv_loader, ood_adv_loader = None, None
        # metrics to be collected (for every val steps)
        step_accuracies = 0.0
        step_kl_loss = 0.0
        step_id_loss, step_ood_loss = 0.0, 0.0
        step_id_precision, step_ood_precision = 0.0, 0.0
        last_validated_step = 0
        # overflow from previous epoch
        overflow = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for epoch in range(init_epoch, num_epochs):
            self.epochs = epoch
            ###################
            # train the model #
            ###################
            epoch_start = time.time()
            # metrics to be collected (epoch_level)
            accuracies = 0.0
            total_kl_loss = 0.0
            total_id_loss, total_ood_loss = 0.0, 0.0
            total_id_precision, total_ood_precision = 0.0, 0.0
            id_outputs_all = None
            ood_outputs_all = None
            
            num_steps_per_epoch = len(self.id_train_loader)
            # iterators for the data loaders, as we don't have common num_batches across them
            id_train_iterator = iter(self.id_train_loader)
            ood_train_iterator = iter(self.ood_train_loader)
            if id_adv_loader is not None:
                id_adv_train_iterator, id_adv_train_batches = iter(id_adv_loader), len(id_adv_loader)
            if ood_adv_loader is not None:
                ood_adv_train_iterator, ood_adv_train_batches = iter(ood_adv_loader), len(ood_adv_loader)
            # beginning of training epoch
            start = time.time()
            for i in range(num_steps_per_epoch): # each batch
                # train on normal images
                batch_train_results = self._train_single_batch(next(id_train_iterator),
                                                               next(ood_train_iterator))
                kl_loss, id_loss, ood_loss, id_precision, ood_precision, accuracy, id_outputs, ood_outputs = batch_train_results
                # train on adv images - id
                if id_adv_loader is not None and i < id_adv_train_batches:
                    id_adv_batch_results = self._train_single_batch_criteria(next(id_adv_train_iterator),
                                                                             self.id_criterion,
                                                                             is_ood=False)
                # train on adv images - ood
                if ood_adv_loader is not None and i < ood_adv_train_batches:
                    ood_adv_batch_results = self._train_single_batch_criteria(next(ood_adv_train_iterator),
                                                                              self.ood_criterion,
                                                                              is_ood=True)
                # increment 1 step
                self.steps += 1
                
                # accumulate all outputs by the model
                if id_outputs_all is None:
                    id_outputs_all = id_outputs
                    ood_outputs_all = ood_outputs
                else:
                    id_outputs_all = torch.cat((id_outputs_all, id_outputs), dim=0)
                    ood_outputs_all = torch.cat((ood_outputs_all, ood_outputs), dim=0)
                # accumulate the metrics
                step_kl_loss += kl_loss
                step_id_loss += id_loss
                step_ood_loss += ood_loss
                step_id_precision += id_precision
                step_ood_precision += ood_precision
                step_accuracies += accuracy
                
                # validate on valset
                if self.steps % val_after_steps == 0:
                    # marks end of training
                    end = time.time()
                    last_validated_step = self.steps
                    step_val_results = self._val_single_epoch()
                    # accumulate epoch level metrics
                    total_kl_loss += step_kl_loss
                    total_id_loss += step_id_loss
                    total_ood_loss += step_ood_loss
                    total_id_precision += step_id_precision
                    total_ood_precision += step_ood_precision
                    accuracies += step_accuracies
                    # log step results so far
                    step_summary = {
                        'train_results': {
                            'loss': np.round(step_kl_loss/val_after_steps, 4),
                            'id_loss': np.round(step_id_loss/val_after_steps, 4),
                            'ood_loss': np.round(step_ood_loss/val_after_steps, 4),
                            'id_accuracy': np.round(100.0 * step_accuracies/val_after_steps, 2),
                            'id_precision': np.round(step_id_precision/val_after_steps, 4),
                            'ood_precision': np.round(step_ood_precision/val_after_steps, 4)
                        },
                        'val_results': step_val_results,
                        'time_taken': np.round(((end-start) / 60.0), 2),
                    }
                    torch.save(step_summary, os.path.join(self.log_dir, f'step_summary_{self.steps}.pt'))
                    print(f"Step {self.steps}:")
                    print(f"Train loss: {step_summary['train_results']['loss']}, \
                        Train accuracy: {step_summary['train_results']['id_accuracy']}")
                    print(f"Val loss: {step_val_results['loss']}, \
                        Val accuracy: {step_val_results['id_accuracy']}")
                    print(f"Time taken for train epoch: {step_summary['time_taken']} mins")
                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping.register_step(self.steps, step_val_results['loss'], self.model, self.log_dir)
                    
                    if early_stopping.do_early_stop:
                        print(f"Early stopping. Restoring model to step {early_stopping.best_step}")
                        self.training_early_stopped = True
                        break
                    # reset timer for next _ steps
                    start = time.time()
                    # reset step metric variables
                    step_accuracies = 0.0
                    step_kl_loss = 0.0
                    step_id_loss, step_ood_loss = 0.0, 0.0
                    step_id_precision, step_ood_precision = 0.0, 0.0

                # step through lr scheduler for batch level ones
                if self.lr_step_after_batch:
                    self.lr_scheduler.step()
            if self.training_early_stopped:
                break
            # accumulate epoch level metrics (subtract previous overflow)
            total_kl_loss -= overflow[0]
            total_id_loss -= overflow[1]
            total_ood_loss -= overflow[2]
            total_id_precision -= overflow[3]
            total_ood_precision -= overflow[4]
            accuracies -= overflow[5]
            #  (for last unaccounted val step)
            if self.steps > last_validated_step:
                total_kl_loss += step_kl_loss
                total_id_loss += step_id_loss
                total_ood_loss += step_ood_loss
                total_id_precision += step_id_precision
                total_ood_precision += step_ood_precision
                accuracies += step_accuracies
                # these values will be extra in the next epoch, so should be subtracted later
                overflow = (step_kl_loss, step_id_loss, step_ood_loss, step_id_precision, step_ood_precision, step_accuracies)
            # log uncertainties needed for ood-detect adv generation
            if self.log_uncertainties:
                id_uncertainties = UncertaintyEvaluator(id_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
                ood_uncertainties = UncertaintyEvaluator(ood_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
                uncertainty_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-uncertainties')
                os.makedirs(uncertainty_dir)
                for key in ood_uncertainties.keys():
                    np.savetxt(os.path.join(uncertainty_dir, 'ood_' + key._value_ + '.txt'),
                            ood_uncertainties[key])
                    np.savetxt(os.path.join(uncertainty_dir, 'id_' + key._value_ + '.txt'),
                            id_uncertainties[key])
            # generate adv images (at the end of each epoch except last epoch)
            if self.epochs < num_epochs:
                print("Generating adversarial images")
                id_adv_dataset, ood_adv_dataset = self._generate_adversarial_dataset()
                id_adv_loader, ood_adv_loader = self._get_adv_data_loaders(id_adv_dataset,
                                                                            ood_adv_dataset)
            epoch_end = time.time()
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
                'train_results': {
                    'loss': np.round(total_kl_loss/num_steps_per_epoch, 4),
                    'id_loss': np.round(total_id_loss/num_steps_per_epoch, 4),
                    'ood_loss': np.round(total_ood_loss/num_steps_per_epoch, 4),
                    'id_accuracy': np.round(100.0 * accuracies/num_steps_per_epoch, 2),
                    'id_precision': np.round(total_id_precision/num_steps_per_epoch, 4),
                    'ood_precision': np.round(total_ood_precision/num_steps_per_epoch, 4)
                    },
                'val_results': val_results,
                'time_taken': np.round(((epoch_end-epoch_start) / 60.0), 2),
                'num_steps': self.steps, # number of steps completed when epoch finished
                'num_in-out_adv': len(id_adv_dataset) if id_adv_dataset is not None else 0,
                'num_out-in_adv': len(ood_adv_dataset) if ood_adv_dataset is not None else 0
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            print(f"Train loss: {summary['train_results']['loss']}, \
                  Train accuracy: {summary['train_results']['id_accuracy']}")
            # print(f"Adv Train loss: {adv_train_results['loss']}, \
            #      Adv Train accuracy: {adv_train_results['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']} mins")

            # step through lr scheduler
            if not self.lr_step_after_batch:
                self.lr_scheduler.step()

        # load the last checkpoint with the best model
        if self.training_early_stopped:
            self.model, _ = load_model(self.log_dir,
                                    device=self.device,
                                    name=early_stopping.best_model_name)

        save_model_with_params_from_ckpt(self.model, self.log_dir)

    def train(self, num_epochs=None, resume=False, ckpt=None, stepwise_train=False, val_after_steps=100):
        """
        Provide either num_epochs for epoch level training or along with stepwise_train indicating
        that validation needs to happen every 'val_after_steps' steps.
        """
        assert isinstance(num_epochs, int)

        if stepwise_train:
            #self.train_stepwise(num_epochs, val_after_steps, resume, ckpt)
            self.train_stepwise_mixed_batch(num_epochs, val_after_steps, resume, ckpt)
            return

        # initialize the early_stopping object
        early_stopping = EarlyStopper(self.min_epochs, self.patience, verbose=True)

        # resume from previous ckpt
        assert resume is False or ckpt is not None
        init_epoch = 0
        if resume is True:
            init_epoch = ckpt['epochs'] + 1
            self.optimizer.load_state_dict(ckpt['opt_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            early_stopping.resume_from_ckpt(ckpt['early_stopping_best_epoch'],
                                            ckpt['early_stopping_best_val_loss'],
                                            ckpt['early_stopping_patience_counter'])
            print(f"Model restored from checkpoint at epoch {init_epoch}")

        for epoch in range(init_epoch, num_epochs):
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
                                                 'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                                                 'early_stopping_best_epoch': early_stopping.best_epoch,
                                                 'early_stopping_best_val_loss': early_stopping.best_val_loss,
                                                 'early_stopping_patience_counter': early_stopping.counter
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
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            print(f"Train loss: {train_results['loss']}, \
                  Train accuracy: {train_results['id_accuracy']}")
            print(f"Adv Train loss: {adv_train_results['loss']}, \
                  Adv Train accuracy: {adv_train_results['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']} mins")

            # early_stopping needs the validation loss to check if it has decreased, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping.register_epoch(self.epochs+1, val_results['loss'], self.model, self.log_dir)

            if early_stopping.do_early_stop:
                print(f"Early stopping. Restoring model to epoch {early_stopping.best_epoch+1}")
                self.training_early_stopped = True
                break

            # step through lr scheduler
            if not self.lr_step_after_batch:
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

    def _generate_adversarial_dataset(self,
                                      id_dataset=None,
                                      ood_dataset=None,
                                      only_true_adversaries=True,
                                      step_level=False,
                                      previous_epoch=False):
        """
        Creates an adversarial dataset based on the current trained model.
        For ood-detect adv training, generates both in->out adversarials and
        out->in adversarials. While for normal adv training, generates only
        misclassification adversarials on in domain dataset.

        Also persists the generated adv images, if needed.
        """
        id_train_adv_set = []
        ood_train_adv_set = []
        epochs = self.epochs if previous_epoch else self.epochs+1
        dir_prefix = f'step-{self.steps}' if step_level else f'epoch-{epochs}'
        if self.adv_training_type == "normal":
                # only works on in domain samples which get misclassified under attack
            additional_args = {'only_true_adversaries': only_true_adversaries}
            id_train_adv_set = AdversarialDataset(self.id_train_dataset if id_dataset is None else id_dataset,
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
                self._persist_adv_dataset(id_train_adv_set, f'{dir_prefix}-adv-images')
        elif self.adv_training_type == "ood-detect":
            if not self.use_fixed_threshold:
                threshold = get_optimal_threshold(os.path.join(self.log_dir,
                                                               f'{dir_prefix}-uncertainties'),
                                                  self.uncertainty_measure)
            else:
                threshold = self.known_threshold_value
            additional_args = {'only_true_adversaries': only_true_adversaries,
                               'uncertainty_measure': self.uncertainty_measure,
                               'uncertainty_threshold': threshold
                               }
            if not self.only_out_in_adversarials:
                id_train_adv_set = AdversarialDataset(self.id_train_dataset if id_dataset is None else id_dataset,
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
            ood_train_adv_set = AdversarialDataset(self.ood_train_dataset if ood_dataset is None else ood_dataset,
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
                self._persist_adv_dataset(id_train_adv_set, f'{dir_prefix}-adv-images')
                self._persist_adv_dataset(ood_train_adv_set, f'{dir_prefix}-adv-images-ood')
            
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
