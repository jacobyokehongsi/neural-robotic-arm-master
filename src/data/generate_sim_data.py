import numpy as np
import matplotlib.pyplot as plt
from .old_dataset import EpisodicDataset, Episode, Step
import pickle

init = np.array([0.335, -0.02, -0.07])
obs = np.array([[0.537, -0.02, -0.07], [0.335, -0.19, -0.07]])
goals = np.array([[0.537, -0.02, -0.07], [0.335, -0.19, -0.07]])
has = np.array([[1,0,0,0], [0,1,0,0]])

def generate_trajectory(a, b, shift, step, eps):
    t = a.copy()
    c = (a+b)/2
    c[2] -= shift
    points = []
    d_ab = np.linalg.norm(a-b)
    while(np.linalg.norm(t-b) > eps and t[2] > c[2]):
        points.append(t.copy())
        z_d = np.dot(a[:2]-b[:2], t[:2]-c[:2])/(t[2]-c[2])
        d = np.array([b[0]-a[0],b[1]-a[1],z_d])
        unit_d = d/np.linalg.norm(d)
        t += step*unit_d
        new_d_ab = np.linalg.norm(t-b)
        if(new_d_ab > d_ab):
            break
        else:
            d_ab = new_d_ab

    return np.array(points)

epi_ds = EpisodicDataset()
for i in range(2):
    goal = goals[i,:]
    ha = has[i,:]
    dist = np.linalg.norm(init-goal)
    for shift in np.arange(0.25,0.87,0.01):
        curr_epi = Episode()
        points = generate_trajectory(init, goal, shift, dist*1e-3, dist*1e-3).copy()
        for pt_i in range(len(points)-1):
            curr_step = Step(joint_velocity=np.array([]), 
                             ee_velocity=np.array(points[pt_i+1]-points[pt_i]), 
                             context = {'ee_pos':points[pt_i].copy(), 'gripper_width': np.array([0.08])})
            curr_step.ee_velocity /= np.linalg.norm(curr_step.ee_velocity)
            curr_step.gripper_velocity = np.array([0.])
            curr_step.human_action = ha.copy()
            curr_epi.steps.append(curr_step)
        epi_ds.episodes.append(curr_epi)

epi_ds.dump('data/raw/cyclic_traj_simple.pkl')

test_idxs = np.random.randint(0, len(epi_ds.episodes), int(0.1*len(epi_ds.episodes)))
epi_ds_test = EpisodicDataset()
epi_ds_test.episodes = [x for i,x in enumerate(epi_ds.episodes) if i in test_idxs]
epi_ds_test.dump('data/interim/cyclic_traj_simple_test.pkl')
epi_ds.episodes = [x for i,x in enumerate(epi_ds.episodes) if i not in test_idxs]
epi_ds.dump('data/interim/cyclic_traj_simple_train.pkl')

train_data, train_label = [], []
for epi in epi_ds.episodes:
    for st in epi.steps:
        train_data.append(np.hstack((st.human_action, st.context['ee_pos'], st.context['gripper_width'])))
        train_label.append(np.hstack((st.ee_velocity, st.gripper_velocity)))
        
train_data = np.array(train_data)
train_label = np.array(train_label)

test_data, test_label = [], []
for epi in epi_ds_test.episodes:
    for st in epi.steps:
        test_data.append(np.hstack((st.human_action, st.context['ee_pos'], st.context['gripper_width']))) # human action index = 0-3, pos + gripper width = 4-7
        test_label.append(np.hstack((st.ee_velocity, st.gripper_velocity)))
        
test_data = np.array(test_data)
test_label = np.array(test_label)

np.save('data/processed/cyclic_traj_simple_train_data.npy', train_data)
np.save('data/processed/cyclic_traj_simple_train_label.npy', train_label)
np.save('data/processed/cyclic_traj_simple_test_data.npy', test_data)
np.save('data/processed/cyclic_traj_simple_test_label.npy', test_label)