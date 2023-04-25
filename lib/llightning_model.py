import lightning as pl
import torch
import torchmetrics

import wandb
from thop import profile
from torch import nn

from lib import gnn


class Criterion(nn.Module):
    def __init__(self, criterion_dict):
        super().__init__()
        criteria = []
        weights = []

        for criterion in criterion_dict:
            loss = eval(criterion_dict[criterion]["module_name"])()
            criteria.append(loss)
            weights.append(criterion_dict[criterion]["weight"])
        self.criteria = criteria
        self.weights = weights

    def forward(self, predicted, target):
        loss = 0
        for criterion, weight in zip(self.criteria, self.weights):
            loss += criterion(predicted, target) * weight
        return loss


class LitNodePredictor(pl.LightningModule):
    def __init__(self, network_module, network_params, criterion, optimizer_module,
                 optimizer_params, scheduler_module, scheduler_params, metrics):
        super().__init__()
        self.save_hyperparameters()
        network_module = getattr(gnn, network_module)
        self.network = network_module(**network_params)
        self.criterion = Criterion(criterion_dict=criterion)
        optimizer_module = eval(optimizer_module)
        self.optimizer = optimizer_module(self.network.parameters(), **optimizer_params)
        scheduler_module = eval(scheduler_module)
        self.scheduler = scheduler_module(self.optimizer, **scheduler_params)
        for metric in metrics:
            metrics[metric] = eval(metrics[metric]["module"])(**metrics[metric]["params"])
        metrics = nn.ModuleDict(metrics)
        self.metrics = metrics
        self.best_metrics = {key: 0 for key in metrics}
        self.logged_complexity = False
        self.epoch_log = {}
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        if not self.logged_complexity:
            macs, params = profile(self.network, inputs=(batch,))
            self.epoch_log['Computational complexity'] = macs
            self.epoch_log['Number of parameters'] = params
            self.logged_complexity = True
        output = self.network(batch)
        loss = self.criterion(output, batch.y)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.mean(torch.stack(self.training_step_outputs))
        self.epoch_log["train_loss"] = loss
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result = {}
        output = self.network(batch)
        for metric_name in self.metrics:
            result[metric_name] = self.metrics[metric_name](output, batch.y)
        self.validation_step_outputs.append(result)
        return result

    def on_validation_end(self):
        result = {}
        for val_out in self.validation_step_outputs:
            for key in val_out:
                if key in result:
                    result[key].append(val_out[key].view(-1))
                else:
                    result[key] = [val_out[key].view(-1), ]
        for key in result:
            result[key] = torch.mean(torch.cat(result[key]))

        for key in self.best_metrics:
            if result[key] > self.best_metrics[key]:
                self.best_metrics[key] = result[key]
            metric_name = key.split("_")[-1]
            result[f"best_{metric_name}"] = self.best_metrics[key]

        log = {**self.epoch_log, **result}
        wandb.log(log)
        self.epoch_log = {}
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def forward(self, batch):
        return self.network(batch)

    def configure_optimizers(self):
        return [self.optimizer, ], [self.scheduler, ]
