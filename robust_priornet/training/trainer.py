import math
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..eval.model_prediction_eval import ClassifierPredictionEvaluator
from ..eval.uncertainty import UncertaintyEvaluator
from ..utils.pytorch import load_model, save_model_with_params_from_ckpt
from .early_stopping import EarlyStopper, EarlyStopperSteps


class PriorNetTrainer:

    def __init__(self, model, id_train_dataset, id_val_dataset,
                 ood_train_dataset, ood_val_dataset,
                 criterion, id_criterion, ood_criterion, optimizer_fn, 
                 optimizer_params: Dict[str, Any] = {},
                 lr_scheduler=None,
                 lr_scheduler_params={},
                 add_ce_loss=False,
                 ce_weight=0.5,
                 batch_size=64,
                 min_epochs=25, patience=20,
                 device=None, clip_norm=10.0, num_workers=0,
                 pin_memory=False, log_dir='.',
                 log_uncertainties=False):
                # may be make num_workers 0 if you have problems with concurrency (in RPN)
        # validate if both datasets are of same size (to avoid inductive bias in model)
        assert len(id_train_dataset) == len(ood_train_dataset)
        assert len(id_val_dataset) == len(ood_val_dataset)

        self.model = model
        self.criterion = criterion
        self.id_criterion = id_criterion
        self.ood_criterion = ood_criterion
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log_dir = log_dir
        self.min_epochs = min_epochs
        self.patience = patience
        self.device = device
        self.clip_norm = clip_norm
        self.log_uncertainties = log_uncertainties
        self.add_ce_loss = add_ce_loss
        self.ce_weight = ce_weight

        if lr_scheduler is not None:
            self.lr_step_after_batch = False
            self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_params)
            # if lr_scheduler == torch.optim.lr_scheduler.OneCycleLR:
            #     self.lr_step_after_batch = True

        # Dataloaders for train dataset
        self.id_train_loader = DataLoader(id_train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers,
                                          pin_memory=self.pin_memory)

        self.ood_train_loader = DataLoader(ood_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)

        # Dataloaders for val dataset
        self.id_val_loader = DataLoader(id_val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory)
        self.ood_val_loader = DataLoader(ood_val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=self.num_workers,
                                         pin_memory=self.pin_memory)

        # Track training metrics
        self.train_loss, self.train_accuracy, self.train_eval_steps = [], [], []
        # Track validation metrics
        self.val_loss, self.val_accuracy, self.val_eval_steps = [], [], []

        # Training step counter
        self.steps: int = 0
        self.epochs: int = 0
        
        # Early stopping
        self.training_early_stopped = False


    def train_stepwise(self, num_epochs, val_after_steps, resume=False, ckpt=None):
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
            # beginning of training epoch
            start = time.time()
            for i in range(num_steps_per_epoch): # each batch
                # train on normal images
                batch_train_results = self._train_single_batch(next(id_train_iterator),
                                                               next(ood_train_iterator))
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
                    # marks end of training now we need to validate
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
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            print(f"Train loss: {summary['train_results']['loss']}, \
                  Train accuracy: {summary['train_results']['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']} mins")

            # step through lr scheduler (epoch level schedulers)
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
            self.train_stepwise(num_epochs, val_after_steps, resume, ckpt)
            return

        # initialize the early_stopping object
        early_stopping = EarlyStopper(self.min_epochs, self.patience, verbose=True)

        # resume model from previous checkpoint
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
                'val_results': val_results,
                'time_taken': np.round(((end-start) / 60.0), 2),
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            print(f"Train loss: {train_results['loss']}, \
                  Train accuracy: {train_results['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']} mins")

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping.register_epoch(self.epochs+1, val_results['loss'], self.model, self.log_dir)

            if early_stopping.do_early_stop:
                print(f"Early stopping. Restoring model to epoch {early_stopping.best_epoch+1}")
                self.training_early_stopped = True
                break

            # step through lr scheduler (only for epoch level steps)
            if not self.lr_step_after_batch:
                self.lr_scheduler.step()

        # load the last checkpoint with the best model
        if self.training_early_stopped:
            self.model, _ = load_model(self.log_dir,
                                    device=self.device,
                                    name=early_stopping.best_model_name)

        save_model_with_params_from_ckpt(self.model, self.log_dir)

    def _eval_logits_id_ood_samples(self, id_inputs, ood_inputs):
        id_outputs = self.model(id_inputs)
        ood_outputs = self.model(ood_inputs)
        return id_outputs, ood_outputs

    def _train_single_batch_criteria(self, data, criterion, is_ood=False):
        """
        Trains on the given samples (may be id or ood batch) on
        the specified criertion - KL div loss for peaky target dirichlet
        on id samples, KL div loss for flat target dirichlet on ood
        samples.
        """
        self.model.train()
        
        inputs, labels = data
        if self.device is not None:
            inputs, labels = map(lambda x: x.to(self.device,
                                                non_blocking=self.pin_memory),
                                 (inputs, labels))
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # eval samples
        outputs = self.model(inputs)
        
        # Calculate train loss
        loss = criterion(outputs, None if is_ood else labels)
        assert torch.all(torch.isfinite(loss)).item()

        # divide the loss by target dirichlet precision (if any), so that loss is inline with lr
        if criterion.target_precision > 0:
            loss = loss / criterion.target_precision

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()

        alphas = torch.exp(outputs)
        precision = torch.mean(torch.sum(alphas, dim=1)).item()

        accuracy = None
        if not is_ood:
            # measure classification accuracy
            probs = F.softmax(outputs, dim=1)
            accuracy = ClassifierPredictionEvaluator.compute_accuracy(probs, labels,
                                                                      self.device).item()

        return (loss.item(), precision, accuracy, outputs)

    def _train_single_batch(self, id_data, ood_data):
        """
        Trains on both id and ood samples on the combined kl div loss/criterion.
        """
        # Set model in train mode
        self.model.train()
        # Get inputs
        inputs, labels = id_data
        ood_inputs, _ = ood_data
        if self.device is not None:
            inputs, labels, ood_inputs = map(lambda x: x.to(self.device,
                                                            non_blocking=self.pin_memory),
                                             (inputs, labels, ood_inputs))

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # eval the id and ood inputs
        id_outputs, ood_outputs = self._eval_logits_id_ood_samples(inputs, ood_inputs)

        # Calculate train loss (overall loss including both id, ood samples)
        loss = self.criterion((id_outputs, ood_outputs), (labels, None))
        assert torch.all(torch.isfinite(loss)).item()

        # include CE loss if needed
        if self.add_ce_loss:
            ce_loss = torch.nn.CrossEntropyLoss()
            ce_loss_val = ce_loss(id_outputs, labels)
            loss.add_(self.ce_weight * ce_loss_val)

        # Measures ID and OOD losses
        id_loss = self.id_criterion(id_outputs, labels).item()
        ood_loss = self.ood_criterion(ood_outputs, None).item()

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()

        # precision of the dirichlet dist output by the model (id, ood seprately)
        id_alphas = torch.exp(id_outputs)
        id_precision = torch.mean(torch.sum(id_alphas, dim=1)).item()
        ood_alphas = torch.exp(ood_outputs)
        ood_precision = torch.mean(torch.sum(ood_alphas, dim=1)).item()

        probs = F.softmax(id_outputs, dim=1)
        accuracy = ClassifierPredictionEvaluator.compute_accuracy(probs, labels,
                                                                  self.device).item()
        return (loss.item(), id_loss, ood_loss, id_precision, ood_precision, accuracy, id_outputs, ood_outputs)

    def _train_single_epoch(self, id_train_loader, ood_train_loader, is_adversarial_epoch=False):
        # metrics to be collected
        accuracies = 0.0
        total_kl_loss = 0.0
        total_id_loss, total_ood_loss = 0.0, 0.0
        total_id_precision, total_ood_precision = 0.0, 0.0
        id_outputs_all = None
        ood_outputs_all = None
        for i, (data, ood_data) in enumerate(
                zip(id_train_loader, ood_train_loader), 0):
            
            batch_result = self._train_single_batch(data, ood_data)
            kl_loss, id_loss, ood_loss, id_precision, ood_precision, accuracy, id_outputs, ood_outputs = batch_result
            # Update the number of steps
            self.steps += 1
            # accumulate all outputs by the model
            if id_outputs_all is None:
                id_outputs_all = id_outputs
                ood_outputs_all = ood_outputs
            else:
                id_outputs_all = torch.cat((id_outputs_all, id_outputs), dim=0)
                ood_outputs_all = torch.cat((ood_outputs_all, ood_outputs), dim=0)

            total_kl_loss += kl_loss
            total_id_loss += id_loss
            total_ood_loss += ood_loss
            total_id_precision += id_precision
            total_ood_precision += ood_precision
            accuracies += accuracy

            if self.lr_step_after_batch:
                self.lr_scheduler.step()

        # average the metrics over all steps (batches) in this epoch
        num_batches = len(self.id_train_loader)
        accuracies /= num_batches
        total_kl_loss /= num_batches
        total_id_loss /= num_batches
        total_ood_loss /= num_batches
        total_id_precision /= num_batches
        total_ood_precision /= num_batches

        if self.log_uncertainties:
            id_uncertainties = UncertaintyEvaluator(id_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
            ood_uncertainties = UncertaintyEvaluator(ood_outputs_all.detach().cpu().numpy()).get_all_uncertainties()
            if is_adversarial_epoch:
                uncertainty_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-uncertainties-adv')
            else:
                uncertainty_dir = os.path.join(self.log_dir, f'epoch-{self.epochs+1}-uncertainties')
            os.makedirs(uncertainty_dir)
            for key in ood_uncertainties.keys():
                np.savetxt(os.path.join(uncertainty_dir, 'ood_' + key._value_ + '.txt'),
                           ood_uncertainties[key])
                np.savetxt(os.path.join(uncertainty_dir, 'id_' + key._value_ + '.txt'),
                           id_uncertainties[key])
        # returns average metrics (loss, accuracy, dirichlet_dist_precision)
        return {
            'loss': np.round(total_kl_loss, 4),
            'id_loss': np.round(total_id_loss, 4),
            'ood_loss': np.round(total_ood_loss, 4),
            'id_accuracy': np.round(100.0 * accuracies, 2),
            'id_precision': np.round(total_id_precision, 4),
            'ood_precision': np.round(total_ood_precision, 4)
        }


    def _val_single_epoch(self):
        # Set model in eval mode
        self.model.eval()

        accuracies = 0.0
        kl_loss = 0.0
        id_loss, ood_loss = 0.0, 0.0
        id_precision, ood_precision = 0.0, 0.0

        with torch.no_grad():
            for i, (data, ood_data) in enumerate(
                    zip(self.id_val_loader, self.ood_val_loader), 0):
                # Get inputs
                inputs, labels = data
                ood_inputs, _ = ood_data
                if self.device is not None:
                    inputs, labels, ood_inputs = map(lambda x: x.to(self.device,
                                                                    non_blocking=self.pin_memory),
                                                     (inputs, labels, ood_inputs))

                # append id samples with ood samples
                id_outputs, ood_outputs = self._eval_logits_id_ood_samples(inputs, ood_inputs)

                # Calculate train loss
                loss = self.criterion((id_outputs, ood_outputs), (labels, None))
                assert torch.all(torch.isfinite(loss)).item()
                
                # include CE loss if needed
                if self.add_ce_loss:
                    ce_loss = torch.nn.CrossEntropyLoss()
                    ce_loss_val = ce_loss(id_outputs, labels)
                    loss.add_(self.ce_weight * ce_loss_val)
                kl_loss += loss.item()

                # Measures ID and OOD losses
                id_loss += self.id_criterion(id_outputs, labels).item()
                ood_loss += self.ood_criterion(ood_outputs, None).item()

                # precision of the dirichlet dist output by the model (id, ood seprately),
                # averaged across all samples in batch
                id_alphas = torch.exp(id_outputs)
                id_precision += torch.mean(torch.sum(id_alphas, dim=1)).item()
                ood_alphas = torch.exp(ood_outputs)
                ood_precision += torch.mean(torch.sum(ood_alphas, dim=1)).item()

                probs = F.softmax(id_outputs, dim=1)
                accuracy = ClassifierPredictionEvaluator.compute_accuracy(probs, labels,
                                                                          self.device).item()
                accuracies += accuracy

        # average the metrics over all steps (batches) in this epoch
        num_batches = len(self.id_val_loader)
        accuracies /= num_batches
        kl_loss /= num_batches
        id_loss /= num_batches
        ood_loss /= num_batches
        id_precision /= num_batches
        ood_precision /= num_batches

        return {
            'loss': np.round(kl_loss, 4),
            'id_loss': np.round(id_loss, 4),
            'ood_loss': np.round(ood_loss, 4),
            'id_accuracy': np.round(100.0 * accuracies, 2),
            'id_precision': np.round(id_precision, 4),
            'ood_precision': np.round(ood_precision, 4)
        }
