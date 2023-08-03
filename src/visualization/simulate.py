import panda_gym
import gym
import torch
import numpy as np
from argparse import ArgumentParser
import multiprocessing as mp
import platform
from time import sleep
from typing import Iterable
import sys
sys.path.append('./')
from visualization.joystick import JoystickController
from envs.cyclic_traj_env import CyclicTrajEnv
from experiment import VAEXperiment
import yaml
from models import vae_models

def simulate(
        decoder,
        start: np.array,
        step_rate: float):

    env = CyclicTrajEnv(render=True, control_type="ee")
    _ = env.reset()

    controller = JoystickController()

    start_ja = env.robot.inverse_kinematics(
            link=11, position=start, orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    env.robot.set_joint_angles(start_ja)

    done = False
    while not done:
        try:
            latent_action = controller.get_action()
        except Exception as e:
            print(e)
            print('Simulation exiting...')
            return

        c1 = np.hstack(
                [env.robot.get_ee_position(), np.array([env.robot.get_fingers_width()])])
        if latent_action.sum() > 0:
            context = torch.Tensor(c1)

            action = decoder.decode(z=latent_action, context=context.view(1,-1))
            action = action.detach().numpy()
            action = np.squeeze(action)
            action[:3] = 2e-3*action[:3]/np.linalg.norm(action[:3])
        else:
            action = np.zeros(4)

        c1 += action
        target_ja = env.robot.inverse_kinematics(
        link=11, position=c1[:3], orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        env.robot.control_joints(target_angles = target_ja)
        env.sim.step()
        sleep(step_rate)

    env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--checkpoint', default=None, type=str, required=True)
    parser.add_argument('--step_rate', default=0.1, type=float)
    args = parser.parse_args()

    config = yaml.safe_load(open('configs/cat_vae.yml'))
    model = vae_models[config['model_params']['name']](**config['model_params'])
    ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    experiment = VAEXperiment(model, config['exp_params'])
    experiment.load_state_dict(ckpt['state_dict'])

    experiment.model.eval()

    start = np.array([0.335, -0.02, -0.07])

    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')

    p_sim = mp.Process(
        target=simulate,
        args=(
            experiment.model,
            start,
            args.step_rate))

    p_sim.start()
    p_sim.join()
