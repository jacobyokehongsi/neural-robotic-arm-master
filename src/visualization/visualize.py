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

parser = argparse.ArgumentParser(
    description='')

parser.add_argument('--checkpoint',
                    dest="checkpoint",
                    help='checkpoint path of the VAE model')

parser.add_argument('--img_path',
                    dest="img_path",
                    help='path to save visualizations')

args = parser.parse_args()

init = np.array([0.335, -0.02, -0.07])
goals = np.array([[0.537, -0.02, -0.07], [0.335, -0.19, -0.07], ])

dirs = torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
dir_names = ['Up', 'Right', 'Down', 'Left']

config = yaml.safe_load(open('configs/cat_vae.yml'))
model = vae_models[config['model_params']['name']](**config['model_params'])
ckpt = torch.load(args.checkpoint)
experiment = VAEXperiment(model, config['exp_params'])
experiment.load_state_dict(ckpt['state_dict'])

experiment.model.eval()

step = 1e-2

points = {0:[],1:[],2:[],3:[]}

for i in range(4):
    curr = init.copy()
    d_ab = np.linalg.norm(curr-goals, axis=1).min()
    limit = int(d_ab/(2*step))
    it = 0
    while(True):
        direction = experiment.model.decode(z=dirs[i,:].view(1,-1).float(),context=torch.from_numpy(np.hstack((curr, np.array([0.08])))).view(1,-1).float())
        direction = direction.detach().numpy()
        curr += step*(direction[0][:-1]/np.linalg.norm(direction[0][:-1]))
        points[i].append(curr.copy())
        new_d_ab = np.linalg.norm(curr-goals, axis=1).min()
        it += 1
        if(it > limit and new_d_ab > d_ab):
            break
        else:
            d_ab = new_d_ab

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = {0:[],1:[],2:[],3:[]}
ys = {0:[],1:[],2:[],3:[]}
zs = {0:[],1:[],2:[],3:[]}
c = {0:[],1:[],2:[],3:[]}
clrs = ['b','r','g','y']

for i in range(4):
    xs[i] = [x[0] for i, x in enumerate(points[i])]
    ys[i] = [x[1] for i, x in enumerate(points[i])]
    zs[i] = [x[2] for i, x in enumerate(points[i])]
    c[i] = [clrs[i]]*len(xs[i])
    ax.scatter(xs[i], ys[i], zs[i], c=c[i], label=dir_names[i])

ax.scatter(*list(init),marker='x',s=81,label='Start')
ax.scatter(*list(goals[0,:]),marker='x',s=81,label='Up Goal')
ax.scatter(*list(goals[1,:]),marker='x',s=81,label='Right Goal')

plt.legend()
plt.title(config['logging_params']['name'])
plt.savefig(args.img_path+config['logging_params']['name']+'.png')

for i in range(4):
    for j in range(2):
        for k in range(3):
            print(f'The output for {dir_names[i]} direction at Goal {j} for gripper width {0.08-k*0.02} is {experiment.model.decode(z=dirs[i,:].view(1,-1).float(),context=torch.from_numpy(np.hstack((goals[j,:], np.array([0.08-k*0.02])))).view(1,-1).float()).detach().numpy()[0]}.')