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
from .adversarial_trainer import AdversarialPriorNetTrainer
from .early_stopping import EarlyStopper


class AdversarialPriorNetBatchTrainer(AdversarialPriorNetTrainer):
    """
    Priornet trainer which performs a training epoch with normal images
    and generates adversarial images based on the trained model. Starting from
    second epoch, batches of normal and adv images are interleaved.
    Also monitors val performance every {val_measurement_steps} steps/batches instead of epochs.
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
                 only_out_in_adversarials: bool = False,
                 validate_after_steps: int = 100):
        assert adv_training_type in ['normal', 'ood-detect']

        super(AdversarialPriorNetBatchTrainer, self).__init__(model, id_train_dataset,
                                                              id_val_dataset, ood_train_dataset,
                                                              ood_val_dataset, criterion,
                                                              id_criterion, ood_criterion,
                                                              optimizer_fn,
                                                              adv_attack_type, adv_attack_criteria,
                                                              optimizer_params,
                                                              lr_scheduler, lr_scheduler_params,
                                                              batch_size, min_epochs, patience, device,
                                                              clip_norm, num_workers,
                                                              pin_memory, log_dir,
                                                              attack_params=
                                                              attack_params,
                                                              dataset_persistence_params=
                                                              dataset_persistence_params,
                                                              adv_training_type=
                                                              adv_training_type,
                                                              uncertainty_measure=
                                                              uncertainty_measure,
                                                              use_fixed_threshold=
                                                              use_fixed_threshold,
                                                              known_threshold_value=
                                                              known_threshold_value,
                                                              only_out_in_adversarials=
                                                              only_out_in_adversarials)
        self.validate_after_steps = validate_after_steps

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
        for epoch in range(init_epoch, num_epochs):
            print(f'Epoch: {epoch + 1} / {num_epochs}')
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
                if self.steps % self.validate_after_steps == 0:
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
                            'loss': np.round(step_kl_loss/self.validate_after_steps, 4),
                            'id_loss': np.round(step_id_loss/self.validate_after_steps, 4),
                            'ood_loss': np.round(step_ood_loss/self.validate_after_steps, 4),
                            'id_accuracy': np.round(100.0 * step_accuracies/self.validate_after_steps, 2),
                            'id_precision': np.round(step_id_precision/self.validate_after_steps, 4),
                            'ood_precision': np.round(step_ood_precision/self.validate_after_steps, 4)
                        },
                        'val_results': step_val_results,
                        'time_taken': np.round(((end-start) / 60.0), 2),
                    }
                    torch.save(step_summary, os.path.join(self.log_dir, f'step_summary_{self.steps}.pt'))
                    # reset timer for next _ steps
                    start = time.time()
                    # reset step metric variables
                    step_accuracies = 0.0
                    step_kl_loss = 0.0
                    step_id_loss, step_ood_loss = 0.0, 0.0
                    step_id_precision, step_ood_precision = 0.0, 0.0
            # accumulate epoch level metrics (for last unaccounted val step)
            if self.steps > last_validated_step:
                total_kl_loss += step_kl_loss
                total_id_loss += step_id_loss
                total_ood_loss += step_ood_loss
                total_id_precision += step_id_precision
                total_ood_precision += step_ood_precision
                accuracies += step_accuracies
            epoch_end = time.time()
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
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f"Train loss: {summary['train_results']['loss']}, \
                  Train accuracy: {summary['train_results']['id_accuracy']}")
            # print(f"Adv Train loss: {adv_train_results['loss']}, \
            #      Adv Train accuracy: {adv_train_results['id_accuracy']}")
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
