import glob
import json
import logging
import os

import numpy as np
import torch
from pybullet_utils import transformations

from . import pose3d
from . import motion_util
from . import utils


class AMPLoader:
    def __init__(
        self,
        device,
        motion_files,
        time_between_frames,
        key_body_ids,
        preload_transitions=False,
        num_preload_transitions=1000000,
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self._key_body_ids = torch.tensor(key_body_ids, device=device)

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self._motion_lengths = []
        self._motion_num_frames = []
        self._motion_dt = []
        self.POS_SIZE = None

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])[: -120 * 1]
                # motion_data = self.reorder_from_pybullet_to_isaac(motion_data)

                self.init_field_offsets(
                    motion_json["Frame_offset"], motion_json["Frame_size"]
                )

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = self.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        self.POS_SIZE : (self.POS_SIZE + self.ROT_SIZE),
                    ] = root_rot

                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, self.ROOT_ROT_END_IDX : self.JOINT_VEL_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : self.JOINT_VEL_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = round((motion_data.shape[0] - 1) * frame_duration)
                print(f"traj_len: {traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self._motion_num_frames = torch.tensor(
            self.trajectory_num_frames, device=self.device
        )
        self._motion_lengths = torch.tensor(
            self.trajectory_lens, device=self.device, dtype=torch.float32
        )
        self._motion_dt = torch.tensor(
            self.trajectory_frame_durations, device=self.device, dtype=torch.float32
        )
        self._motion_weights = torch.tensor(
            self.trajectory_weights, dtype=torch.float32, device=self.device
        )
        self._motion_weights /= self._motion_weights.sum()

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )

            print(self.get_joint_pose_batch(self.preloaded_s).mean(dim=0))
            print(f"Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def init_field_offsets(self, frame_offset, frame_size):
        if self.POS_SIZE != None:
            return
        if frame_offset is None:
            raise ValueError("Frame_offset field is missing")
        if isinstance(frame_offset, list):
            if len(frame_offset) == 1:
                frame_offset = frame_offset[0]
            else:
                raise ValueError("Frame_offset array contains more than one item")
        if not isinstance(frame_offset, dict):
            raise TypeError(
                "Frame_offset must be either an object or a single-item array of objects"
            )
        if frame_size is None:
            raise ValueError("Frame_size field is missing in motion_json")
        if isinstance(frame_size, list):
            if len(frame_size) == 1:
                frame_size = frame_size[0]
            else:
                raise ValueError("frame_offset array contains more than one item")
        if not isinstance(frame_size, dict):
            raise TypeError(
                "frame_offset must be either an object or a single-item array of objects"
            )

        print(f"frame_offset: {frame_offset}")
        print(f"frame_size:   {frame_size}")
        print(frame_size["root_pos"])
        self.POS_SIZE = int(frame_size["root_pos"])
        self.ROT_SIZE = int(frame_size["root_quat"])
        self.JOINT_POS_SIZE = int(frame_size["joints_pos"])
        self.LEFT_TOE_POS_LOCAL_SIZE = int(frame_size["left_toe_pos"])
        self.RIGHT_TOE_POS_LOCAL_SIZE = int(frame_size["right_toe_pos"])
        self.LINEAR_VEL_SIZE = int(frame_size["world_linear_vel"])
        self.ANGULAR_VEL_SIZE = int(frame_size["world_angular_vel"])
        self.JOINT_VEL_SIZE = int(frame_size["joints_vel"])
        self.LEFT_TOE_VEL_LOCAL_SIZE = int(frame_size["left_toe_vel"])
        self.RIGHT_TOE_VEL_LOCAL_SIZE = int(frame_size["right_toe_vel"])

        self.ROOT_POS_START_IDX = int(frame_offset["root_pos"])
        self.ROOT_POS_END_IDX = self.ROOT_POS_START_IDX + self.POS_SIZE

        self.ROOT_ROT_START_IDX = int(frame_offset["root_quat"])
        self.ROOT_ROT_END_IDX = self.ROOT_ROT_START_IDX + self.ROT_SIZE

        self.JOINT_POSE_START_IDX = int(frame_offset["joints_pos"])
        self.JOINT_POSE_END_IDX = self.JOINT_POSE_START_IDX + self.JOINT_POS_SIZE

        self.LEFT_TOE_POS_LOCAL_START_IDX = int(frame_offset["left_toe_pos"])
        self.LEFT_TOE_POS_LOCAL_END_IDX = (
            self.LEFT_TOE_POS_LOCAL_START_IDX + self.LEFT_TOE_POS_LOCAL_SIZE
        )

        self.RIGHT_TOE_POS_LOCAL_START_IDX = int(frame_offset["right_toe_pos"])
        self.RIGHT_TOE_POS_LOCAL_END_IDX = (
            self.RIGHT_TOE_POS_LOCAL_START_IDX + self.RIGHT_TOE_POS_LOCAL_SIZE
        )

        self.LINEAR_VEL_START_IDX = int(frame_offset["world_linear_vel"])
        self.LINEAR_VEL_END_IDX = self.LINEAR_VEL_START_IDX + self.LINEAR_VEL_SIZE

        self.ANGULAR_VEL_START_IDX = int(frame_offset["world_angular_vel"])
        self.ANGULAR_VEL_END_IDX = self.ANGULAR_VEL_START_IDX + self.ANGULAR_VEL_SIZE

        self.JOINT_VEL_START_IDX = int(frame_offset["joints_vel"])
        self.JOINT_VEL_END_IDX = self.JOINT_VEL_START_IDX + self.JOINT_VEL_SIZE

        self.LEFT_TOE_VEL_LOCAL_START_IDX = int(frame_offset["left_toe_vel"])
        self.LEFT_TOE_VEL_LOCAL_END_IDX = (
            self.LEFT_TOE_VEL_LOCAL_START_IDX + self.LEFT_TOE_VEL_LOCAL_SIZE
        )

        self.RIGHT_TOE_VEL_LOCAL_START_IDX = int(frame_offset["right_toe_vel"])
        self.RIGHT_TOE_VEL_LOCAL_END_IDX = (
            self.RIGHT_TOE_VEL_LOCAL_START_IDX + self.RIGHT_TOE_VEL_LOCAL_SIZE
        )

    def reorder_from_pybullet_to_isaac(self, motion_data):
        """Convert from PyBullet ordering to Isaac ordering.

        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].
        """
        root_pos = self.get_root_pos_batch(motion_data)
        root_rot = self.get_root_rot_batch(motion_data)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(
            self.get_joint_pose_batch(motion_data), 4, axis=1
        )
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])

        fp_fr, fp_fl, fp_rr, fp_rl = np.split(
            self.get_tar_toe_pos_local_batch(motion_data), 4, axis=1
        )
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])

        lin_vel = self.get_linear_vel_batch(motion_data)
        ang_vel = self.get_angular_vel_batch(motion_data)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(
            self.get_joint_vel_batch(motion_data), 4, axis=1
        )
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(
            self.get_tar_toe_vel_local_batch(motion_data), 4, axis=1
        )
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        return np.hstack(
            [
                root_pos,
                root_rot,
                joint_pos,
                foot_pos,
                lin_vel,
                ang_vel,
                joint_vel,
                foot_vel,
            ]
        )

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
        )

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = (
            self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs))
            - subst
        )
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_starts = torch.zeros(
            len(traj_idxs), self.observation_dim, device=self.device
        )
        all_frame_ends = torch.zeros(
            len(traj_idxs), self.observation_dim, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(
            p * n - idx_low, device=self.device, dtype=torch.float32
        ).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        idx_low = np.clip(idx_low, 0, n - 1)
        idx_high = np.clip(idx_high, 0, n - 1)
        all_frame_pos_starts = torch.zeros(
            len(traj_idxs), self.POS_SIZE, device=self.device
        )
        all_frame_pos_ends = torch.zeros(
            len(traj_idxs), self.POS_SIZE, device=self.device
        )
        all_frame_rot_starts = torch.zeros(
            len(traj_idxs), self.ROT_SIZE, device=self.device
        )
        all_frame_rot_ends = torch.zeros(
            len(traj_idxs), self.ROT_SIZE, device=self.device
        )
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs),
            self.JOINT_VEL_END_IDX - self.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs),
            self.JOINT_VEL_END_IDX - self.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(
                trajectory[idx_low[traj_mask]]
            )
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(
                trajectory[idx_high[traj_mask]]
            )
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(
                trajectory[idx_low[traj_mask]]
            )
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(
                trajectory[idx_high[traj_mask]]
            )
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, self.JOINT_POSE_START_IDX : self.JOINT_VEL_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, self.JOINT_POSE_START_IDX : self.JOINT_VEL_END_IDX
            ]

        blend = torch.tensor(
            p * n - idx_low, device=self.device, dtype=torch.float32
        ).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)

        rot_blend = utils.quaternion_slerp(
            all_frame_rot_starts, all_frame_rot_ends, blend
        )

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = self.get_tar_toe_pos_local(
            frame0
        ), self.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = self.get_linear_vel(frame0), self.get_linear_vel(
            frame1
        )
        angular_vel_0, angular_vel_1 = self.get_angular_vel(
            frame0
        ), self.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(
            frame1
        )

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend
        )
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32,
            device=self.device,
        )
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat(
            [
                blend_root_pos,
                blend_root_rot,
                blend_joints,
                blend_tar_toe_pos,
                blend_linear_vel,
                blend_angular_vel,
                blend_joints_vel,
            ]
        )

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[
                    idxs, self.JOINT_POSE_START_IDX : self.JOINT_VEL_END_IDX
                ]
                s = torch.cat(
                    [
                        s,
                        self.preloaded_s[
                            idxs,
                            self.ROOT_POS_START_IDX + 2 : self.ROOT_POS_START_IDX + 3,
                        ],
                    ],
                    dim=-1,
                )
                s_next = self.preloaded_s_next[
                    idxs, self.JOINT_POSE_START_IDX : self.JOINT_VEL_END_IDX
                ]
                s_next = torch.cat(
                    [
                        s_next,
                        self.preloaded_s_next[
                            idxs,
                            self.ROOT_POS_START_IDX + 2 : self.ROOT_POS_START_IDX + 3,
                        ],
                    ],
                    dim=-1,
                )
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames
                        )
                    )

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    def num_motions(self):
        return len(self.trajectory_num_frames)

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._motion_weights, num_samples=n, replacement=True
        )
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self.device)

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_num_frames[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        phase = motion_times % motion_len
        frames = self.get_full_frame_at_time_batch(
            motion_ids.cpu().numpy(), phase.cpu().numpy()
        )
        root_pos = self.get_root_pos_batch(frames)
        root_rot = self.get_root_rot_batch(frames)
        dof_pos = self.get_joint_pose_batch(frames)
        root_vel = self.get_linear_vel_batch(frames)
        root_ang_vel = self.get_angular_vel_batch(frames)
        dof_vel = self.get_joint_vel_batch(frames)
        key_pos0 = self.get_left_toe_pos_local_batch(frames)
        key_pos1 = self.get_right_toe_pos_local_batch(frames)

        # key_pos0[:, 2] += 0.01
        # key_pos1[:, 2] += 0.01

        return (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            torch.cat((key_pos0.unsqueeze(1), key_pos1.unsqueeze(1)), dim=1),
        )

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] + 1

    def get_root_pos(self, pose):
        return pose[self.ROOT_POS_START_IDX : self.ROOT_POS_END_IDX]

    def get_root_pos_batch(self, poses):
        return poses[:, self.ROOT_POS_START_IDX : self.ROOT_POS_END_IDX]

    def get_root_rot(self, pose):
        return pose[self.ROOT_ROT_START_IDX : self.ROOT_ROT_END_IDX]

    def get_root_rot_batch(self, poses):
        return poses[:, self.ROOT_ROT_START_IDX : self.ROOT_ROT_END_IDX]

    def get_joint_pose(self, pose):
        return pose[self.JOINT_POSE_START_IDX : self.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(self, poses):
        return poses[:, self.JOINT_POSE_START_IDX : self.JOINT_POSE_END_IDX]

    def get_left_toe_pos_local(self, pose):
        return pose[self.LEFT_TOE_POS_LOCAL_START_IDX : self.LEFT_TOE_POS_LOCAL_END_IDX]

    def get_left_toe_pos_local_batch(self, poses):
        return poses[
            :,
            self.LEFT_TOE_POS_LOCAL_START_IDX : self.LEFT_TOE_POS_LOCAL_END_IDX,
        ]

    def get_right_toe_pos_local(self, pose):
        return pose[
            self.RIGHT_TOE_POS_LOCAL_START_IDX : self.RIGHT_TOE_POS_LOCAL_END_IDX
        ]

    def get_right_toe_pos_local_batch(self, poses):
        return poses[
            :,
            self.RIGHT_TOE_POS_LOCAL_START_IDX : self.RIGHT_TOE_POS_LOCAL_END_IDX,
        ]

    def get_linear_vel(self, pose):
        return pose[self.LINEAR_VEL_START_IDX : self.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(self, poses):
        return poses[:, self.LINEAR_VEL_START_IDX : self.LINEAR_VEL_END_IDX]

    def get_angular_vel(self, pose):
        return pose[self.ANGULAR_VEL_START_IDX : self.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(self, poses):
        return poses[:, self.ANGULAR_VEL_START_IDX : self.ANGULAR_VEL_END_IDX]

    def get_joint_vel(self, pose):
        return pose[self.JOINT_VEL_START_IDX : self.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.JOINT_VEL_START_IDX : self.JOINT_VEL_END_IDX]

    def get_left_toe_vel_local(self, pose):
        return pose[self.LEFT_TOE_VEL_LOCAL_START_IDX : self.LEFT_TOE_VEL_LOCAL_END_IDX]

    def get_left_toe_vel_local_batch(self, poses):
        return poses[
            :,
            self.LEFT_TOE_VEL_LOCAL_START_IDX : self.LEFT_TOE_VEL_LOCAL_END_IDX,
        ]

    def get_right_toe_vel_local(self, pose):
        return pose[
            self.RIGHT_TOE_VEL_LOCAL_START_IDX : self.RIGHT_TOE_VEL_LOCAL_END_IDX
        ]

    def get_right_toe_vel_local_batch(self, poses):
        return poses[
            :,
            self.RIGHT_TOE_VEL_LOCAL_START_IDX : self.RIGHT_TOE_VEL_LOCAL_END_IDX,
        ]
