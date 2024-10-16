import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=False, default="saved_obs.pkl")
args = parser.parse_args()

obses = pickle.load(open(args.data, "rb"))
num_obs = len(obses[0])

channels = [
    "projected gravity 0",
    "projected gravity 1",
    "projected gravity 2",
    "pos left hip yaw",
    "pos left hip roll",
    "pos left hip pitch",
    "pos left knee",
    "pos left ankle",
    "pos neck pitch",
    "pos head pitch",
    "pos head yaw",
    "pos left antenna",
    "pos right antenna",
    "pos right hip yaw",
    "pos right hip roll",
    "pos right hip pitch",
    "pos right knee",
    "pos right ankle",
    "vel left hip yaw",
    "vel left hip roll",
    "vel left hip pitch",
    "vel left knee",
    "vel left ankle",
    "vel neck pitch",
    "vel head pitch",
    "vel head yaw",
    "vel left antenna",
    "vel right antenna",
    "vel right hip yaw",
    "vel right hip roll",
    "vel right hip pitch",
    "vel right knee",
    "vel right ankle",
    "action left hip yaw",
    "action left hip roll",
    "action left hip pitch",
    "action left knee",
    "action left ankle",
    "action neck pitch",
    "action head pitch",
    "action head yaw",
    "action left antenna",
    "action right antenna",
    "action right hip yaw",
    "action right hip roll",
    "action right hip pitch",
    "action right knee",
    "action right ankle",
    "commands 0",
    "commands 1",
    "commands 2",
]
# obses contain a list of observation lists over time. each observation list contains num_obs elements
import os

pkl_name = os.path.basename(args.data)

fig, axs = plt.subplots(8, 7, figsize=(15, 17), sharex=True)
fig.suptitle(f"Channels over Time - {pkl_name}")

time = np.arange(len(obses))

for idx, channel in enumerate(channels):
    ax = axs[idx // 7, idx % 7]
    data = [obs[idx] for obs in obses]
    ax.plot(time, data)
    ax.set_title(channel)
    ax.grid(True)
    ax.set_ylim(-3, 3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
