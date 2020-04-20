import math
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..utils.pytorch import get_accuracy, save_model


class PriorNetTrainer:

    def __init__(self, model, id_train_dataset, id_val_dataset,
                    ood_train_dataset, ood_val_dataset,
                    criterion, id_criterion, ood_criterion, optimizer_fn, 
                    optimizer_params: Dict[str, Any] = {},
                    lr_scheduler=None,
                    lr_scheduler_params={},
                    batch_size=64, patience=20, device=None, clip_norm=10.0, num_workers=4,
                    pin_memory=False, log_dir='.'):

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
        self.patience = patience
        self.device = device
        self.clip_norm = clip_norm

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

        # initialize the early_stopping object
        # early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1} / {num_epochs}')
            ###################
            # train the model #
            ###################
            start = time.time()
            train_results = self._train_single_epoch() # train and update metrics list
            end = time.time()

            ######################
            # validate the model #
            ######################
            val_results = self._val_single_epoch() # validate and update metrics list

            # store and print epoch summary
            summary = {
                'train_results': train_results,
                'val_results': val_results,
                'time_taken': (end-start),
            }
            torch.save(summary, os.path.join(self.log_dir, f'epoch_summary_{epoch+1}.pt'))
            print(f"Train loss: {train_results['loss']}, \
                  Train accuracy: {train_results['id_accuracy']}")
            print(f"Val loss: {val_results['loss']}, \
                  Val accuracy: {val_results['id_accuracy']}")
            print(f"Time taken for train epoch: {summary['time_taken']}")

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            #early_stopping(val_results['loss'], self.model)

            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

            # step through lr scheduler
            self.lr_scheduler.step()

        # load the last checkpoint with the best model
        # self.model.load_state_dict(torch.load('checkpoint.pt'))
        save_model(self.model, {}, self.log_dir)

    def _eval_logits_id_ood_samples(self, id_inputs, ood_inputs):
        # append id samples with ood samples
        cat_inputs = torch.cat([id_inputs, ood_inputs], dim=1).view(
            torch.Size([2 * id_inputs.size()[0]]) + id_inputs.size()[1:])
        logits = self.model(cat_inputs).view([id_inputs.size()[0], -1])
        id_outputs, ood_outputs = torch.chunk(logits, 2, dim=1)
        return id_outputs, ood_outputs

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        # metrics to be collected
        accuracies = 0.0
        kl_loss = 0.0
        id_loss, ood_loss = 0.0, 0.0
        id_precision, ood_precision = 0.0, 0.0
        for i, (data, ood_data) in enumerate(
                zip(self.id_train_loader, self.ood_train_loader), 0):
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

            # Calculate train loss
            loss = self.criterion((id_outputs, ood_outputs), (labels, None))
            assert torch.all(torch.isfinite(loss)).item()
            kl_loss += loss.item()

            # Measures ID and OOD losses
            id_loss += self.id_criterion(id_outputs, labels).item()
            ood_loss += self.ood_criterion(ood_outputs, None).item()

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # precision of the dirichlet dist output by the model (id, ood seprately), 
            # averaged across all samples in batch
            id_alphas = torch.exp(id_outputs - torch.max(id_outputs, dim=0)[0]) 
            id_precision += torch.mean(torch.sum(id_alphas, dim=1)).item()
            ood_alphas = torch.exp(ood_outputs - torch.max(ood_outputs, dim=0)[0])
            ood_precision += torch.mean(torch.sum(ood_alphas, dim=1)).item()

            probs = F.softmax(id_outputs, dim=1)
            accuracy = get_accuracy(probs, labels, self.device).item()
            accuracies += accuracy

        # average the metrics over all steps (batches) in this epoch
        num_batches = len(self.id_train_loader)
        accuracies /= num_batches
        kl_loss /= num_batches
        id_loss /= num_batches
        ood_loss /= num_batches
        id_precision /= num_batches
        ood_precision /= num_batches

        # returns average metrics (loss, accuracy, dirichlet_dist_precision)
        return {
            'loss': np.round(kl_loss, 4),
            'id_loss': np.round(id_loss, 2),
            'ood_loss': np.round(ood_loss, 2),
            'id_accuracy': np.round(100.0 * accuracies, 2),
            'id_precision': np.round(id_precision, 2),
            'ood_precision': np.round(ood_precision, 2)
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
                kl_loss += loss.item()

                # Measures ID and OOD losses
                id_loss += self.id_criterion(id_outputs, labels).item()
                ood_loss += self.ood_criterion(ood_outputs, None).item()

                # precision of the dirichlet dist output by the model (id, ood seprately), 
                # averaged across all samples in batch
                id_alphas = torch.exp(id_outputs - torch.max(id_outputs, dim=0)[0]) 
                id_precision += torch.mean(torch.sum(id_alphas, dim=1)).item()
                ood_alphas = torch.exp(ood_outputs - torch.max(ood_outputs, dim=0)[0])
                ood_precision += torch.mean(torch.sum(ood_alphas, dim=1)).item()

                probs = F.softmax(id_outputs, dim=1)
                accuracy = get_accuracy(probs, labels, self.device).item()
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
            'id_loss': np.round(id_loss, 2),
            'ood_loss': np.round(ood_loss, 2),
            'id_accuracy': np.round(100.0 * accuracies, 2),
            'id_precision': np.round(id_precision, 2),
            'ood_precision': np.round(ood_precision, 2)
        }
