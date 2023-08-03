import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data.new_dataset import RoboticArmDataset
from pytorch_lightning.strategies import DDPStrategy
from models import vae_models
import copy
import torch 
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


parser = argparse.ArgumentParser(
    description='Generic runner for VAE models for neural-robotic-arm')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)


data = RoboticArmDataset(
    **config["data_params"], pin_memory=len(config['trainer_params']['devices']) != 0)

data.setup()

lr_prints = ['1e-2','1e-3','5e-3','1e-4']

all_metrics = {}

for hd in [[20,20],[50,50],[100,100]]:
    for i, lr in enumerate([0.01,0.001,0.005,0.0001]):
        config['logging_params']['name'] = f"cat-vae-grasp-hd-{hd[0]}-{hd[1]}-lr-{lr_prints[i]}-klw-1e-2-ne-100"
        config['exp_params']['LR'] = lr
        config['model_params']['hidden_dims']  = hd


        wdb_logger = WandbLogger(save_dir=config['logging_params']['save_dir'],
                                name=config['logging_params']['name'],
                                project="neural-robotic-arm",
                                entity="ucla-ncel-robotics")

        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(model,
                                config['exp_params'])

        runner = Trainer(logger=wdb_logger,
                        callbacks=[
                            LearningRateMonitor(),
                            ModelCheckpoint(save_top_k=2,
                                            dirpath=os.path.join(
                                                wdb_logger.save_dir, "checkpoints", config['logging_params']['name']),
                                            monitor="val_loss",
                                            save_last=True),
                        ],
                        strategy=DDPStrategy(find_unused_parameters=False),
                        **config['trainer_params'])


        print(f"======= Training {config['logging_params']['name']} =======")
        runner.fit(experiment, datamodule=data)

        wdb_logger.experiment.finish()

        test_dl = data.test_dataloader()

        preds = []
        actuals = []

        la_predicted = []

        for (i, (X, y)) in enumerate(test_dl):
            X, y = X.float(), y.float()
            _, _, q = experiment.model(action=y, context=X[:, model.categorical_dim:])
            q = torch.nn.functional.softmax(q,  dim=-1)
            q = q.detach().numpy()
            yhat = experiment.model.decode(z=X[:, :model.categorical_dim],context=X[:, model.categorical_dim:])
            yhat = yhat.detach().numpy()
            y = y.numpy()
            preds.append(yhat)
            actuals.append(y)
            # if i == 0:
            #     print(q)
            la_predicted.append(np.argmax(q, axis=1))

        preds, actuals, la_predicted = np.vstack(preds), np.vstack(actuals), np.hstack(la_predicted)

        cosine_errors = 1-(preds*actuals).sum(axis=1)/(np.linalg.norm(preds, axis=1)*np.linalg.norm(actuals, axis=1))

        metrics = {'mae': mean_absolute_error(actuals, preds),
                    'mse': mean_squared_error(actuals, preds),
                    'mce': cosine_errors.mean(),
                    'max_ce': cosine_errors.max(),
                    'r2': r2_score(actuals, preds)}

        all_metrics[config['logging_params']['name']] = copy.deepcopy(metrics)


print(all_metrics)
