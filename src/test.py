import sys
sys.path.append('/home3/shivam/neural-robotic-arm/src/')

from experiment import VAEXperiment
import numpy as np
import torch
import argparse
import yaml
import torch 
from models import vae_models
import matplotlib.pyplot as plt
from data.new_dataset import RoboticArmDataset
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

parser = argparse.ArgumentParser(
    description='')

parser.add_argument('--checkpoint',
                    dest="checkpoint",
                    help='checkpoint path of the VAE model')

args = parser.parse_args()

config = yaml.safe_load(open('configs/cat_vae.yml'))
model = vae_models[config['model_params']['name']](**config['model_params'])
ckpt = torch.load(args.checkpoint)
experiment = VAEXperiment(model, config['exp_params'])
experiment.load_state_dict(ckpt['state_dict'])

experiment.model.eval()

data = RoboticArmDataset(
    **config["data_params"], pin_memory=len(config['trainer_params']['devices']) != 0)
    
data.setup()
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

metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['metric'])
metrics_df.index.name = 'metric_type'
metrics_df.reset_index(inplace=True)

print(metrics_df)

unique, counts = np.unique(la_predicted, return_counts=True)

print(dict(zip(unique, counts)))