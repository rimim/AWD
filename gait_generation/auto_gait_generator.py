"""
Runs gait generator for ranges of dx, dy and dtheta
"""

import argparse
import json
import os
import time
import webbrowser
import threading
import numpy as np
import placo
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from gait.placo_walk_engine import PlacoWalkEngine

FPS = 60


class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, ".5f"))


def record(pwe, args_dict):
    episode = {
        "LoopMode": "Wrap",
        "FPS": FPS,
        "FrameDuration": np.around(1 / FPS, 4),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": False,
        "Joints": [],
        "Vel_x": [],
        "Vel_y": [],
        "Yaw": [],
        "Placo": [],
        "Frame_offset": [],
        "Frame_size": [],
        "Frames": [],
        "MotionWeight": 1,
    }
    if args_dict["debug"]:
        episode["Debug_info"] = []

    pwe.reset()
    first_joints_positions = list(pwe.get_angles().values())
    first_T_world_fbase = pwe.robot.get_T_world_fbase()
    first_T_world_leftFoot = pwe.robot.get_T_world_left()
    first_T_world_rightFoot = pwe.robot.get_T_world_right()

    pwe.set_traj(args_dict["dx"], args_dict["dy"], args_dict["dtheta"])  # + 0.00955)
    DT = 0.001
    start = time.time()

    last_record = 0
    prev_root_position = [0, 0, 0]
    prev_root_orientation_euler = [0, 0, 0]
    prev_left_toe_pos = [0, 0, 0]
    prev_right_toe_pos = [0, 0, 0]
    prev_joints_positions = None
    i = 0
    prev_initialized = False
    avg_x_lin_vel = []
    avg_y_lin_vel = []
    avg_yaw_vel = []
    added_frame_info = False
    # center_y_pos = None
    center_y_pos = -(pwe.parameters.feet_spacing / 2)
    print(f"center_y_pos: {center_y_pos}")
    while True:
        pwe.tick(DT)
        if pwe.t <= 0 + args_dict["skip_warmup"] * 1:
            start = pwe.t
            last_record = pwe.t + 1 / FPS
            continue

        # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

        if pwe.t - last_record >= 1 / FPS:
            if args_dict["stand"]:
                T_world_fbase = first_T_world_fbase
            else:
                T_world_fbase = pwe.robot.get_T_world_fbase()
            root_position = list(T_world_fbase[:3, 3])
            if not args_dict["mini"]:
                root_position[2] = round(root_position[2], 1)

            # if center_y_pos is None:
            #    center_y_pos = root_position[1]

            # Why ?
            # Commented this for mini bdx as it shifted the trunk frame
            if not args_dict["mini"]:
                root_position[1] = root_position[1] - center_y_pos

            if round(root_position[2], 5) < 0:
                print(f"BAD root_position: {root_position[2]:.5f}")
            root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
            joints_positions = list(pwe.get_angles().values())

            if args_dict["stand"]:
                joints_positions = first_joints_positions
                T_world_leftFoot = first_T_world_leftFoot
                T_world_rightFoot = first_T_world_rightFoot
            else:
                joints_positions = list(pwe.get_angles().values())
                T_world_leftFoot = pwe.robot.get_T_world_left()
                T_world_rightFoot = pwe.robot.get_T_world_right()

            T_body_leftFoot = (
                T_world_leftFoot  # np.linalg.inv(T_world_fbase) @ T_world_leftFoot
            )
            T_body_rightFoot = (
                T_world_rightFoot  # np.linalg.inv(T_world_fbase) @ T_world_rightFoot
            )

            left_toe_pos = list(T_body_leftFoot[:3, 3])
            right_toe_pos = list(T_body_rightFoot[:3, 3])

            if not prev_initialized:
                prev_root_position = root_position.copy()
                prev_root_orientation_euler = (
                    R.from_quat(root_orientation_quat).as_euler("xyz").copy()
                )
                prev_left_toe_pos = left_toe_pos.copy()
                prev_right_toe_pos = right_toe_pos.copy()
                prev_joints_positions = joints_positions.copy()
                prev_initialized = True

            world_linear_vel = list(
                (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
            )
            avg_x_lin_vel.append(world_linear_vel[0])
            avg_y_lin_vel.append(world_linear_vel[1])
            body_rot_mat = T_world_fbase[:3, :3]
            body_linear_vel = list(body_rot_mat.T @ world_linear_vel)
            # print("world linear vel", world_linear_vel)
            # print("body linear vel", body_linear_vel)

            world_angular_vel = list(
                (
                    R.from_quat(root_orientation_quat).as_euler("xyz")
                    - prev_root_orientation_euler
                )
                / (1 / FPS)
            )
            avg_yaw_vel.append(world_angular_vel[2])
            body_angular_vel = list(body_rot_mat.T @ world_angular_vel)
            # print("world angular vel", world_angular_vel)
            # print("body angular vel", body_angular_vel)

            if prev_joints_positions == None:
                prev_joints_positions = [0] * len(joints_positions)
            joints_vel = list(
                (np.array(joints_positions) - np.array(prev_joints_positions))
                / (1 / FPS)
            )
            left_toe_vel = list(
                (np.array(left_toe_pos) - np.array(prev_left_toe_pos)) / (1 / FPS)
            )
            right_toe_vel = list(
                (np.array(right_toe_pos) - np.array(prev_right_toe_pos)) / (1 / FPS)
            )

            if prev_initialized:
                if args_dict["hardware"]:
                    episode["Frames"].append(
                        root_position
                        + root_orientation_quat
                        + joints_positions
                        + left_toe_pos
                        + right_toe_pos
                        + world_linear_vel
                        + world_angular_vel
                        + joints_vel
                        + left_toe_vel
                        + right_toe_vel
                    )
                else:
                    episode["Frames"].append(
                        root_position + root_orientation_quat + joints_positions
                    )
                if args_dict["debug"]:
                    episode["Debug_info"].append(
                        {
                            "left_foot_pose": list(T_world_leftFoot.flatten()),
                            "right_foot_pose": list(T_world_rightFoot.flatten()),
                        }
                    )
                if not added_frame_info:
                    added_frame_info = True
                    offset = 0
                    offset_root_pos = offset
                    offset = offset + len(root_position)
                    offset_root_quat = offset
                    offset = offset + len(root_orientation_quat)
                    offset_joints_pos = offset
                    offset = offset + len(joints_positions)
                    offset_left_toe_pos = offset
                    offset = offset + len(left_toe_pos)
                    offset_right_toe_pos = offset
                    offset = offset + len(right_toe_pos)
                    offset_world_linear_vel = offset
                    offset = offset + len(world_linear_vel)
                    offset_world_angular_vel = offset
                    offset = offset + len(world_angular_vel)
                    offset_joints_vel = offset
                    offset = offset + len(joints_vel)
                    offset_left_toe_vel = offset
                    offset = offset + len(left_toe_vel)
                    offset_right_toe_vel = offset
                    offset = offset + len(right_toe_vel)

                    episode["Joints"] = list(pwe.get_angles().keys())
                    episode["Frame_offset"].append(
                        {
                            "root_pos": offset_root_pos,
                            "root_quat": offset_root_quat,
                            "joints_pos": offset_joints_pos,
                            "left_toe_pos": offset_left_toe_pos,
                            "right_toe_pos": offset_right_toe_pos,
                            "world_linear_vel": offset_world_linear_vel,
                            "world_angular_vel": offset_world_angular_vel,
                            "joints_vel": offset_joints_vel,
                            "left_toe_vel": offset_left_toe_vel,
                            "right_toe_vel": offset_right_toe_vel,
                        }
                    )
                    episode["Frame_size"].append(
                        {
                            "root_pos": len(root_position),
                            "root_quat": len(root_orientation_quat),
                            "joints_pos": len(joints_positions),
                            "left_toe_pos": len(left_toe_pos),
                            "right_toe_pos": len(right_toe_pos),
                            "world_linear_vel": len(world_linear_vel),
                            "world_angular_vel": len(world_angular_vel),
                            "joints_vel": len(joints_vel),
                            "left_toe_vel": len(left_toe_vel),
                            "right_toe_vel": len(right_toe_vel),
                        }
                    )

            prev_root_position = root_position.copy()
            prev_root_orientation_euler = (
                R.from_quat(root_orientation_quat).as_euler("xyz").copy()
            )
            prev_left_toe_pos = left_toe_pos.copy()
            prev_right_toe_pos = right_toe_pos.copy()
            prev_joints_positions = joints_positions.copy()
            prev_initialized = True

            last_record = pwe.t

        # if pwe.t - start > args.length:
        #    break
        if len(episode["Frames"]) == args_dict["length"] * FPS:
            break

        i += 1

    mean_avg_x_lin_vel = np.around(np.mean(avg_x_lin_vel), 4)
    mean_avg_y_lin_vel = np.around(np.mean(avg_y_lin_vel), 4)
    mean_yaw_vel = np.around(np.mean(avg_yaw_vel), 4)
    print("recorded", len(episode["Frames"]), "frames")
    print(f"avg lin_vel_x {mean_avg_x_lin_vel}")
    print(f"avg lin_vel_y {mean_avg_y_lin_vel}")
    print(f"avg yaw {mean_yaw_vel}")
    episode["Vel_x"] = mean_avg_x_lin_vel
    episode["Vel_y"] = mean_avg_y_lin_vel
    episode["Yaw"] = mean_yaw_vel
    episode["Placo"] = {
        "dx": args_dict["dx"],
        "dy": args_dict["dy"],
        "dtheta": args_dict["dtheta"],
        "duration": args_dict["length"],
        "hardware": args_dict["hardware"],
        "double_support_ratio": pwe.parameters.double_support_ratio,
        "startend_double_support_ratio": pwe.parameters.startend_double_support_ratio,
        "planned_timesteps": pwe.parameters.planned_timesteps,
        "replan_timesteps": pwe.parameters.replan_timesteps,
        "walk_com_height": pwe.parameters.walk_com_height,
        "walk_foot_height": pwe.parameters.walk_foot_height,
        "walk_trunk_pitch": np.rad2deg(pwe.parameters.walk_trunk_pitch),
        "walk_foot_rise_ratio": pwe.parameters.walk_foot_rise_ratio,
        "single_support_duration": pwe.parameters.single_support_duration,
        "single_support_timesteps": pwe.parameters.single_support_timesteps,
        "foot_length": pwe.parameters.foot_length,
        "feet_spacing": pwe.parameters.feet_spacing,
        "zmp_margin": pwe.parameters.zmp_margin,
        "foot_zmp_target_x": pwe.parameters.foot_zmp_target_x,
        "foot_zmp_target_y": pwe.parameters.foot_zmp_target_y,
        "walk_max_dtheta": pwe.parameters.walk_max_dtheta,
        "walk_max_dy": pwe.parameters.walk_max_dy,
        "walk_max_dx_forward": pwe.parameters.walk_max_dx_forward,
        "walk_max_dx_backward": pwe.parameters.walk_max_dx_backward,
    }

    file_name = args_dict["name"] + str(".json")
    file_path = os.path.join(args.output_dir, file_name)
    os.makedirs(args_dict["output_dir"], exist_ok=True)
    print("DONE, saving", file_name)
    with open(file_path, "w") as f:
        json.encoder.c_make_encoder = None
        json.encoder.float = RoundingFloat
        json.dump(episode, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="recordings")
    parser.add_argument("-l", "--length", type=int, default=10)
    parser.add_argument(
        "-n", "--num_recordings", type=int, default=10, help="Number recordings"
    )
    parser.add_argument("--mini", action="store_true", default=False)
    parser.add_argument("--preset", type=str, default="")
    parser.add_argument("--min_dx", type=float, default=-0.04)
    parser.add_argument("--max_dx", type=float, default=0.04)
    parser.add_argument("--min_dy", type=float, default=-0.05)
    parser.add_argument("--max_dy", type=float, default=0.05)
    parser.add_argument("--min_dtheta", type=float, default=-0.15)
    parser.add_argument("--max_dtheta", type=float, default=0.15)
    args = parser.parse_args()

    # for mini_bdx
    dx_range = [args.min_dx, args.max_dx]
    dy_range = [args.min_dy, args.max_dy]
    dtheta_range = [args.min_dtheta, args.max_dtheta]

    if args.mini:
        robot = "mini_bdx"
        robot_urdf = "urdf/bdx.urdf"
        asset_path = "../awd/data/assets/mini_bdx"
    else:
        robot = "go_bdx"
        robot_urdf = "go_bdx.urdf"
        asset_path = "../awd/data/assets/go_bdx"

    preset_filename = args.preset
    filename = os.path.join(asset_path, "placo_defaults.json")
    if preset_filename:
        if os.path.exists(preset_filename):
            filename = preset_filename
        else:
            print(f"No such file: {preset_filename}")
    with open(filename, "r") as f:
        gait_parameters = json.load(f)
        print(f"gait_parameters {gait_parameters}")

    pwe = PlacoWalkEngine(asset_path, robot_urdf, gait_parameters)

    args_dict = {}
    args_dict["name"] = ""
    args_dict["dx"] = 0
    args_dict["dy"] = 0
    args_dict["dtheta"] = 0
    args_dict["length"] = args.length
    args_dict["skip_warmup"] = False
    args_dict["stand"] = False
    args_dict["hardware"] = True
    args_dict["output_dir"] = args.output_dir
    args_dict["debug"] = True
    args_dict["mini"] = args.mini

    for i in range(args.num_recordings):
        args_dict["dx"] = round(np.random.uniform(dx_range[0], dx_range[1]), 2)
        args_dict["dy"] = round(np.random.uniform(dy_range[0], dy_range[1]), 2)
        args_dict["dtheta"] = round(
            np.random.uniform(dtheta_range[0], dtheta_range[1]), 2
        )
        args_dict["name"] = str(i)
        print("RECORDING ", args_dict["name"])
        print(
            "dx", args_dict["dx"], "dy", args_dict["dy"], "dtheta", args_dict["dtheta"]
        )
        print("==")
        pwe.reset()
        record(pwe, args_dict)
