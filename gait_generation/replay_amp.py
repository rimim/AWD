import argparse
import json
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument(
    "--hardware",
    action="store_true",
    help="use AMP_for_hardware format. If false, use IsaacGymEnvs format",
)
args = parser.parse_args()

fv = Viewer()
fv.start()

episode = json.load(open(args.file))

frame_duration = episode["FrameDuration"]

frames = episode["Frames"]
if "Debug_info" in episode:
    debug = episode["Debug_info"]
else:
    debug = None


# y_speeds = []
# avg_y_speeds = []
# gait_phase_size = 0.432 / frame_duration
# prev_y_pos = 0
# y_pos = 0
pose = np.eye(4)
if args.hardware:
    vels = {}
    vels["linear_vel"] = []
    vels["angular_vel"] = []
    vels["joint_vels"] = []
    root_z = []
for i, frame in enumerate(frames):
    root_position = frame[:3]
    root_orientation_quat = frame[3:7]
    root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()

    pose[:3, 3] = root_position
    pose[:3, :3] = root_orientation_mat

    fv.pushFrame(pose, "aze")

    # prev_y_pos = y_pos
    # y_pos = pose[:3, 3][1]
    # y_speed = (y_pos - prev_y_pos) / frame_duration
    # y_speeds.append(y_speed)
    # y_speeds = y_speeds[-int(gait_phase_size) :]
    # avg_y_speeds.append(np.mean(y_speeds))

    if debug is not None:
        left_foot_pose = np.array(debug[i]["left_foot_pose"]).reshape(4, 4)
        right_foot_pose = np.array(debug[i]["right_foot_pose"]).reshape(4, 4)
        fv.pushFrame(left_foot_pose, "left")
        fv.pushFrame(right_foot_pose, "right")

    if args.hardware:
        vels["linear_vel"].append(frame[28:31])
        vels["angular_vel"].append(frame[31:34])
        vels["joint_vels"].append(frame[34:49])
        root_z.append(R.from_matrix(pose[:3, :3]).as_euler("xyz")[2])

        left_toe_pos = frame[22:25]
        right_toe_pos = frame[25:28]
        fv.pushFrame(fv_utils.make_pose(left_toe_pos, [0, 0, 0]), "left_toe")
        fv.pushFrame(fv_utils.make_pose(right_toe_pos, [0, 0, 0]), "right_toe")

    time.sleep(frame_duration)


if args.hardware:
    # plot vels
    from matplotlib import pyplot as plt

    # TODO
    x_lin_vel = [vels["linear_vel"][i][0] for i in range(len(frames))]
    y_lin_vel = [vels["linear_vel"][i][1] for i in range(len(frames))]
    z_lin_vel = [vels["linear_vel"][i][2] for i in range(len(frames))]

    joints_vel = [vels["joint_vels"][i] for i in range(len(frames))]
    angular_vel_x = [vels["angular_vel"][i][0] for i in range(len(frames))]
    angular_vel_y = [vels["angular_vel"][i][1] for i in range(len(frames))]
    angular_vel_z = [vels["angular_vel"][i][2] for i in range(len(frames))]

    plt.plot(angular_vel_x, label="angular_vel_x")
    plt.plot(angular_vel_y, label="angular_vel_y")
    plt.plot(angular_vel_z, label="angular_vel_z")

    # plt.plot(avg_y_speeds, label="avg_y_speeds")

    plt.plot(root_z, label="root_z")

    plt.legend()
    plt.show()
