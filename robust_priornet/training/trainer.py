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
from .early_stopping import EarlyStopper


class PriorNetTrainer:

    def __init__(self, model, id_train_dataset, id_val_dataset,
                 ood_train_dataset, ood_val_dataset,
                 criterion, id_criterion, ood_criterion, optimizer_fn, 
                 optimizer_params: Dict[str, Any] = {},
                 lr_scheduler=None,
                 lr_scheduler_params={},
                 batch_size=64,
                 min_epochs=25, patience=20,
                 device=None, clip_norm=10.0, num_workers=4,
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

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_params)

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
                'val_results': val_results,
                'time_taken': np.round(((end-start) / 60.0), 2),
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f"Train loss: {train_results['loss']}, \
                  Train accuracy: {train_results['id_accuracy']}")
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

    def _eval_logits_id_ood_samples(self, id_inputs, ood_inputs):
        id_outputs = self.model(id_inputs)
        ood_outputs = self.model(ood_inputs)
        return id_outputs, ood_outputs

    def _train_single_epoch(self, id_train_loader, ood_train_loader, is_adversarial_epoch=False):
        # Set model in train mode
        self.model.train()

        # metrics to be collected
        accuracies = 0.0
        kl_loss = 0.0
        id_loss, ood_loss = 0.0, 0.0
        id_precision, ood_precision = 0.0, 0.0
        id_outputs_all = None
        ood_outputs_all = None
        for i, (data, ood_data) in enumerate(
                zip(id_train_loader, ood_train_loader), 0):
            # Get inputs
            inputs, labels = data
            ood_inputs, _ = ood_data
            if self.device is not None:
                inputs, labels, ood_inputs = map(lambda x: x.to(self.device,
                                                                non_blocking=self.pin_memory),
                                                 (inputs, labels, ood_inputs))

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # append id samples with ood samples
            id_outputs, ood_outputs = self._eval_logits_id_ood_samples(inputs, ood_inputs)
            # accumulate all outputs by the model
            if id_outputs_all is None:
                id_outputs_all = id_outputs
                ood_outputs_all = ood_outputs
            else:
                id_outputs_all = torch.cat((id_outputs_all, id_outputs), dim=0)
                ood_outputs_all = torch.cat((ood_outputs_all, ood_outputs), dim=0)
            # Calculate train loss
            loss = self.criterion((id_outputs, ood_outputs), (labels, None))
            assert torch.all(torch.isfinite(loss)).item()
            kl_loss += loss.item()

            # Measures ID and OOD losses
            prev_id_loss = id_loss
            id_loss += self.id_criterion(id_outputs, labels).item()
            prev_ood_loss = ood_loss
            ood_loss += self.ood_criterion(ood_outputs, None).item()
            if self.steps % 100 == 0:
                print(f"Step {self.steps}: ID loss {id_loss - prev_id_loss}, OOD loss: {ood_loss - prev_ood_loss}")

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

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
        num_batches = len(self.id_train_loader)
        accuracies /= num_batches
        kl_loss /= num_batches
        id_loss /= num_batches
        ood_loss /= num_batches
        id_precision /= num_batches
        ood_precision /= num_batches

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
            'loss': np.round(kl_loss, 4),
            'id_loss': np.round(id_loss, 4),
            'ood_loss': np.round(ood_loss, 4),
            'id_accuracy': np.round(100.0 * accuracies, 2),
            'id_precision': np.round(id_precision, 4),
            'ood_precision': np.round(ood_precision, 4)
        }


    def _val_single_epoch(self):
        # Set model in eval mode
        self.model.eval()

        accuracies = 0.0
        kl_loss = 0.0
        id_loss, ood_loss = 0.0, 0.0
        id_precision, ood_precision = 0.0, 0.0

        kl_loss_all_steps = []
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
                kl_loss += loss.item()
                kl_loss_all_steps.append(loss.item())

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
            'ood_precision': np.round(ood_precision, 4),
            'step_wise_kl_loss': kl_loss_all_steps
        }
