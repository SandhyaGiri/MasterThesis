import math
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from ..datasets.adversarial_dataset import AdversarialDataset
from ..utils.common_data import ATTACK_CRITERIA_MAP, OOD_ATTACK_CRITERIA_MAP
from ..utils.persistence import persist_image_dataset
from ..utils.pytorch import save_model_with_params_from_ckpt
from .trainer import PriorNetTrainer


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
                 dataset_persistence_params=[]):
        super(AdversarialPriorNetTrainer, self).__init__(model, id_train_dataset,
                                                         id_val_dataset, ood_train_dataset,
                                                         ood_val_dataset, criterion,
                                                         id_criterion, ood_criterion,
                                                         optimizer_fn, optimizer_params,
                                                         lr_scheduler, lr_scheduler_params,
                                                         batch_size, patience, device,
                                                         clip_norm, num_workers,
                                                         pin_memory, log_dir)
        self.id_train_dataset = id_train_dataset
        self.ood_train_dataset = ood_train_dataset
        self.batch_size = batch_size
        self.adv_attack_type = adv_attack_type
        self.adv_id_attack_criteria = ATTACK_CRITERIA_MAP[adv_attack_criteria]
        self.adv_ood_attack_criteria = OOD_ATTACK_CRITERIA_MAP[adv_attack_criteria]
        self.attack_params = attack_params
        self.dataset_persistence_params = dataset_persistence_params

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
            ###################
            # train the model #
            ###################
            start = time.time()
            # train and update metrics list
            train_results = self._train_single_epoch(self.id_train_loader,
                                                     self.ood_train_loader)
            adv_train_results = self._adv_train_single_epoch(epoch) # train with adversarial images
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

    def _adv_train_single_epoch(self, epoch):
        """
        Generates adversarial dataset and trains the model on them.
        """
        id_train_adv_set = AdversarialDataset(self.id_train_dataset,
                                              self.adv_attack_type.lower(),
                                              self.model,
                                              self.attack_params['epsilon'],
                                              self.adv_id_attack_criteria,
                                              self.attack_params['norm'],
                                              self.attack_params['step_size'],
                                              self.attack_params['max_steps'],
                                              batch_size=self.batch_size,
                                              device=self.device)
        ood_train_adv_set = AdversarialDataset(self.ood_train_dataset,
                                               self.adv_attack_type.lower(),
                                               self.model,
                                               self.attack_params['epsilon'],
                                               self.adv_ood_attack_criteria,
                                               self.attack_params['norm'],
                                               self.attack_params['step_size'],
                                               self.attack_params['max_steps'],
                                               batch_size=self.batch_size,
                                               device=self.device)
        # persist images? as model progresses in training, attack gradients
        # will differ and we should get different adv images
        if self.attack_params['adv_persist_images']:
            # persist only random sampled 100 images
            indices = np.arange(len(id_train_adv_set))

            id_train_adv_subset = data.Subset(id_train_adv_set, random.sample(list(indices), 100))
            adv_dir = os.path.join(self.log_dir, f'epoch-{epoch}-adv-images')
            os.makedirs(adv_dir)
            persist_image_dataset(id_train_adv_subset,
                                  *self.dataset_persistence_params,
                                  adv_dir)

            indices = np.arange(len(ood_train_adv_set))
            ood_train_adv_subset = data.Subset(ood_train_adv_set, random.sample(list(indices), 100))
            adv_dir = os.path.join(self.log_dir, f'epoch-{epoch}-adv-images-ood')
            os.makedirs(adv_dir)
            persist_image_dataset(ood_train_adv_subset,
                                  *self.dataset_persistence_params,
                                  adv_dir)
        
        # Dataloaders for adv train dataset
        id_train_adv_loader = DataLoader(id_train_adv_set,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers,
                                          pin_memory=self.pin_memory)

        ood_train_adv_loader = DataLoader(ood_train_adv_set,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)
        return self._train_single_epoch(id_train_adv_loader, ood_train_adv_loader)
