import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data.new_dataset import RoboticArmDataset
from pytorch_lightning.strategies import DDPStrategy
from models import AlignmentModule, vae_models
from experiment import VAEXperiment


parser = argparse.ArgumentParser(
    description='Generic runner for VAE models for neural-robotic-arm')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--ckpt',
                    dest="decoder",
                    help='checkpoint for decoder')


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])
ckpt = torch.load(args.ckpt)
experiment = VAEXperiment(model, config['exp_params'])
experiment.load_state_dict(ckpt['state_dict'])


wdb_logger = WandbLogger(save_dir=config['logging_params']['save_dir'],
                         name=config['logging_params']['name'],
                         project="neural-robotic-arm",
                         entity="ucla-ncel-robotics")

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = RoboticArmDataset(
    **config["data_params"], pin_memory=len(config['trainer_params']['devices']) != 0)

data.setup()
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
