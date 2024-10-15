import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=False, default="saved_obs.pkl")
args = parser.parse_args()

obses = pickle.load(open(args.data, "rb"))
num_dofs = 15
dof_poses = []  # (dof, num_obs)
actions = []  # (dof, num_obs)

for i in range(num_dofs):
    dof_poses.append([])
    actions.append([])
    for obs in obses:
        dof_poses[i].append(obs[4 : 4 + 15][i])
        actions[i].append(obs[-18:-3][i])


isaac_joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "left_antenna",
    "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

# plot action vs dof pos

nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(actions[i * nb_cols + j], label="action")
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="dof_pos")
        axs[i, j].legend()
        axs[i, j].set_title(f"{isaac_joints_order[i * nb_cols + j]}")

plt.show()
