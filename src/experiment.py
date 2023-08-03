import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

    def forward(self, action: Tensor, context: Tensor, **kwargs) -> Tensor:
        return self.model(action, context, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        X, y = batch
        X, y = X.float(), y.float()
        self.curr_device = X.device

        results = self.forward(y, X[:, self.model.categorical_dim:])
        train_loss = self.model.loss_function(*results,
                                              kld_weight=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item()
                      for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        X, y = batch
        X, y = X.float(), y.float()
        self.curr_device = X.device

        results = self.forward(y, X[:, self.model.categorical_dim:])
        val_loss = self.model.loss_function(*results,
                                            kld_weight=0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item()
                      for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)
                return optims, scheds
        except:
            return optims
