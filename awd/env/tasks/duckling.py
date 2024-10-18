# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import yaml
import xml.etree.ElementTree as ET
import pickle

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask


class Duckling(BaseTask):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.asset_properties = None
        self._dof_axis = None
        self._dof_axis_array = None

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["terrain"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.randomize_com = self.cfg["env"]["randomizeCom"]
        self.com_range = self.cfg["env"]["comRange"]
        self.randomize_torques = self.cfg["env"]["randomizeTorques"]
        self.torque_multiplier_range = self.cfg["env"]["torqueMultiplierRange"]

        super().__init__(cfg=self.cfg)

        self.dt = self.control_freq_inv * sim_params.dt

        self.common_step_counter = 0
        self.push_interval = np.ceil(self.cfg["env"]["pushIntervalS"] / self.dt)

        # get gym GPU state tensors
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, sensors_per_env, 6
        )[..., :3]

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dof
        )

        # self.gym.set_light_parameters(
        #     self.sim,
        #     0,
        #     gymapi.Vec3(0.8, 0.8, 0.8),
        #     gymapi.Vec3(0.8, 0.8, 0.8),
        #     gymapi.Vec3(1, 2, 3),
        # )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._duckling_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )[..., 0, :]
        self._initial_duckling_root_states = self._duckling_root_states.clone()
        self._initial_duckling_root_states[:, 7:13] = 0.0
        self._initial_duckling_root_states[:, 2] = self.cfg["env"]["initHeight"]
        self._initial_duckling_root_states[:, 3:7] = torch.Tensor(
            self.cfg["env"]["initQuat"]
        ).to(self.device)

        self._duckling_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 0
        ]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 1
        ]

        self._default_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )

        props = self._get_asset_properties()
        for i, name in enumerate(props["init_pos"]):
            angle = props["init_pos"][name]
            self._default_dof_pos[:, i] = angle

        self.p_gains = props["p_gains"]
        self.d_gains = props["d_gains"]
        self.max_effort = props["max_effort"]

        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        self._initial_dof_pos[:, :] = self._default_dof_pos

        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float
        )

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )

        self._rigid_body_pos = rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[
            ..., : self.num_bodies, 10:13
        ]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self.num_bodies, :]

        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self._build_termination_heights()

        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self._key_body_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self._key_body_ids.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))

        self.projected_gravity = quat_rotate_inverse(
            self._root_states[: self.num_envs, 3:7], self.gravity_vec
        )

        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        if self.viewer != None:
            self._init_camera()

        if self.cfg["env"]["debugSaveObs"]:
            self.saved_actions = []

        self.period = self.cfg["env"]["period"]
        self.num_steps_per_period = int(self.period / self.dt)

        self.velocities_history = torch.zeros(
            self.num_envs,
            6 * self.num_steps_per_period,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        if self.cfg["env"]["terrain"]["terrainType"] == "plane":
            self._create_ground_plane()
        else:
            self._create_trimesh()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )
        return

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )
        self._reset_envs(env_ids)

        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.duckling_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )

        return

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._reset_custom_randomization(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._duckling_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0
        return

    def _reset_custom_randomization(self, env_ids):
        if self.randomize_torques:
            self.randomize_torques_factors[env_ids, :] = (
                (self.torque_multiplier_range[0] - self.torque_multiplier_range[1])
                * torch.rand(len(env_ids), self.num_actions, device=self.device)
                + self.torque_multiplier_range[1]
            ).float()

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _get_asset_root(self):
        return self.cfg["env"]["asset"]["assetRoot"]

    def _get_asset_file_name(self):
        return self.cfg["env"]["asset"]["assetFileName"]

    def _setup_character_props(self, key_bodies):
        asset_file = self._get_asset_file_name()
        num_key_bodies = len(key_bodies)

        props = self._get_asset_properties()
        print(f"props: {props}")
        self._joints = props["joints"]
        self._dof_body_ids = props["dof_body_ids"]
        self._dof_offsets = props["dof_offsets"]
        self._dof_obs_size = props["dof_obs_size"]
        self._num_actions = props["num_actions"]
        self._num_obs = props["num_obs"]
        self._stiffness = props["stiffness"]
        self._damping = props["damping"]
        self._friction = props["friction"]
        self._armature = props["armature"]
        self._velocity = props["max_velocity"]
        return

    def _build_termination_heights(self):
        head_term_height = self.cfg["env"]["headTerminationHeight"]
        # shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.duckling_handles[0], "head"
        )
        self._termination_heights[head_id] = max(
            head_term_height, self._termination_heights[head_id]
        )

        asset_file = self._get_asset_file_name()
        # if (asset_file == "mjcf/amp_duckling_sword_shield.xml"):
        #     left_arm_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.duckling_handles[0], "left_lower_arm")
        #     self._termination_heights[left_arm_id] = max(shield_term_height, self._termination_heights[left_arm_id])

        self._termination_heights = to_torch(
            self._termination_heights, device=self.device
        )
        return

    def _get_asset_properties(self):
        if self.asset_properties == None:
            asset_root = self._get_asset_root()
            asset_file = self._get_asset_file_name()

            asset_prop_name = os.path.splitext(asset_file)[0]
            asset_prop_name = f"{asset_prop_name}_props.yaml"
            asset_prop_name = os.path.join(asset_root, asset_prop_name)
            try:
                with open(asset_prop_name, "r") as file:
                    self.asset_properties = yaml.safe_load(file)
            except FileNotFoundError:
                print(f"Error: The file {asset_prop_name} was not found.")
                assert False
            except yaml.YAMLError:
                print(f"Error: Failed to parse {asset_prop_name}.")
                assert False
        return self.asset_properties

    def get_dof_axis(self):
        if self._dof_axis == None:
            asset_root = self._get_asset_root()
            asset_file = self._get_asset_file_name()

            self._dof_axis = {}
            self._dof_axis_array = []
            if asset_file.endswith(".urdf"):
                tree = ET.parse(os.path.join(asset_root, asset_file))
                root = tree.getroot()
                for joint in root.findall("joint"):
                    joint_name = joint.get("name")

                    # Find the axis element (some joints may not have an explicit axis, default is often [1, 0, 0])
                    axis_element = joint.find("axis")
                    if axis_element is not None:
                        axis_xyz = axis_element.get("xyz")
                    else:
                        axis_xyz = "1 0 0"
                    self._dof_axis[joint_name] = [float(x) for x in axis_xyz.split()]
            else:
                print(
                    f"WARNING [get_dof_axis] file extension for {asset_file} not supported"
                )
            self._dof_axis_array = [
                int(value) for axis in self._dof_axis.values() for value in axis
            ]
        return self._dof_axis

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self._get_asset_root()
        asset_file = self._get_asset_file_name()

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        props = self._get_asset_properties()
        asset_options = gymapi.AssetOptions()
        # asset_options.density = 0.001
        # asset_options.armature = props["armature"]
        # asset_options.thickness = props["thickness"]
        # asset_options.angular_damping = props["angular_damping"]
        asset_options.max_angular_velocity = 100.0
        asset_options.max_linear_velocity = 100.0
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        asset_options.default_dof_drive_mode = 0
        # asset_options.fix_base_link = True
        motor_efforts = None
        duckling_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        dof_axis = self.get_dof_axis()
        for key, value in dof_axis.items():
            print(f"DOF {key}: {value}")

        actuator_props = self.gym.get_asset_actuator_properties(duckling_asset)
        if len(actuator_props) != 0:
            motor_efforts = [prop.motor_effort for prop in actuator_props]
        else:
            motor_efforts = props["motor_efforts"]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(
            duckling_asset, "right_foot"
        )
        left_foot_idx = self.gym.find_asset_rigid_body_index(
            duckling_asset, "left_foot"
        )
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(duckling_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(duckling_asset, left_foot_idx, sensor_pose)

        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(duckling_asset)
        self.num_dof = self.gym.get_asset_dof_count(duckling_asset)
        print(f"self.num_dof: {self.num_dof}")
        self.num_joints = self.gym.get_asset_joint_count(duckling_asset)
        print(f"num_joints: {self.num_joints}")

        print(f"_joint count: {len(self._joints)}")
        self.duckling_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, duckling_asset)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(
            self.envs[0], self.duckling_handles[0]
        )
        for j in range(self.num_dof):
            dof_prop_upper = dof_prop["upper"][j]
            dof_prop_lower = dof_prop["lower"][j]
            if dof_prop_lower > dof_prop_upper:
                self.dof_limits_lower.append(dof_prop_upper)
                self.dof_limits_upper.append(dof_prop_lower)
            else:
                self.dof_limits_lower.append(dof_prop_lower)
                self.dof_limits_upper.append(dof_prop_upper)

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if self._pd_control == "isaac":
            self._build_pd_action_offset_scale()

        self.randomize_torques_factors = torch.ones(
            self.num_envs, self.num_actions, device=self.device
        )

        self.randomize_com_values = torch.zeros(self.num_envs, 3, device=self.device)

        if self.randomize_com:
            self.randomize_com_values[:, :] = (
                self.com_range[0] - self.com_range[1]
            ) * torch.rand(self.num_envs, 3, device=self.device) + self.com_range[1]

            for i in range(self.num_envs):
                body_props = self.gym.get_actor_rigid_body_properties(
                    self.envs[i], self.duckling_handles[i]
                )
                body_props[0].com += gymapi.Vec3(
                    self.randomize_com_values[i, 0],
                    self.randomize_com_values[i, 1],
                    self.randomize_com_values[i, 2],
                )
                self.gym.set_actor_rigid_body_properties(
                    self.envs[i],
                    self.duckling_handles[i],
                    body_props,
                    recomputeInertia=True,
                )
        return

    def _build_env(self, env_id, env_ptr, duckling_asset):
        col_group = env_id
        col_filter = self._get_duckling_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self._get_asset_file_name()
        char_h = 0.0

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        duckling_handle = self.gym.create_actor(
            env_ptr,
            duckling_asset,
            start_pose,
            "duckling",
            col_group,
            col_filter,
            segmentation_id,
        )

        self.gym.enable_actor_dof_force_sensors(env_ptr, duckling_handle)

        # for j in range(self.num_bodies):
        #     self.gym.set_rigid_body_color(env_ptr, duckling_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        dof_names = self.gym.get_asset_dof_names(duckling_asset)
        dof_prop = self.gym.get_asset_dof_properties(duckling_asset)
        if self._pd_control == "isaac":
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            for i, dof_name in enumerate(dof_names):
                if dof_name in self._stiffness:
                    dof_prop["stiffness"][i] = self._stiffness[dof_name]
                else:
                    print(f"WARNING: No stiffness values for {dof_name}")
                if dof_name in self._damping:
                    dof_prop["damping"][i] = self._damping[dof_name]
                else:
                    print(f"WARNING: No damping values for {dof_name}")
                # if dof_name in self._friction:
                dof_prop["friction"][i] = self._friction
                dof_prop["armature"][i] = self._armature
            # for key in dof_prop.dtype.names:
            #     print(f"{key}: {dof_prop[key]}")
        elif self._pd_control == "custom":
            dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
            for i, dof_name in enumerate(dof_names):
                dof_prop["friction"][i] = self._friction
                dof_prop["armature"][i] = self._armature
                dof_prop["velocity"][i] = self._velocity
            pass
        else:
            print(f"Unknown PD control type {self._pd_control}. Exiting.")
            exit()
        self.gym.set_actor_dof_properties(env_ptr, duckling_handle, dof_prop)
        self.duckling_handles.append(duckling_handle)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if dof_size == 3:
                curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

                # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_duckling_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_duckling_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_duckling_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            self._rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_heights,
        )
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # print(torch.isnan(self._dof_state).sum(), "### _dof_state")
        # print(torch.isnan(self._rigid_body_pos).sum(), "### _rigid_body_pos")
        # print(torch.isnan(self._rigid_body_rot).sum(), "### _rigid_body_rot")
        # print(torch.isnan(self._rigid_body_vel).sum(), "### _rigid_body_vel")
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_duckling_obs(env_ids)

        # TODO
        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_duckling_obs(self, env_ids=None):
        if env_ids is None:
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            obs = compute_duckling_observations(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
                self._dof_axis_array,
                self.projected_gravity,
                self.actions,
            )
        else:
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            obs = compute_duckling_observations(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
                self._dof_axis_array,
                self.projected_gravity[env_ids],
                self.actions[env_ids],
            )
        # obs = compute_duckling_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
        #                                         self._root_height_obs)

        # key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        # obs = compute_duckling_observations(self._rigid_body_pos[env_ids][:, 0, :],
        #                                     self._rigid_body_rot[env_ids][:, 0, :],
        #                                     self._rigid_body_vel[env_ids][:, 0, :],
        #                                     self._rigid_body_ang_vel[env_ids][:, 0, :],
        #                                     self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
        #                                     self._local_root_obs, self._root_height_obs,
        #                                     self._dof_obs_size, self._dof_offsets, self._dof_axis_array)
        return obs

    def _reset_actors(self, env_ids):
        self._duckling_root_states[env_ids] = self._initial_duckling_root_states[
            env_ids
        ]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        # actions = torch.zeros(
        #     self.num_envs,
        #     self.num_actions,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=False,
        # )
        # if self.cfg["env"]["debugSaveObs"]:
        #     self.saved_actions = []

        if self.cfg["env"]["debugSaveObs"]:
            self.saved_actions.append(actions[0].cpu().numpy())
            pickle.dump(self.saved_actions, open("saved_actions.pkl", "wb"))

        self.prev_actions = self.actions.clone()
        self.actions = actions.to(self.device).clone()
        if self._pd_control == "isaac":
            # TODO WARNING broke (?) this for go_bdx
            # pd_tar = self._action_to_pd_targets(self.actions)
            # pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            # pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            # self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

            tar = self.actions * self.power_scale + self._default_dof_pos
            tar_tensor = gymtorch.unwrap_tensor(tar)
            self.gym.set_dof_position_target_tensor(self.sim, tar_tensor)
        elif self._pd_control == "custom":
            self.render()
            for i in range(self.control_freq_inv):
                self.torques = self.p_gains * (
                    self.actions * self.power_scale
                    + self._default_dof_pos
                    - self._dof_pos
                ) - (self.d_gains * self._dof_vel)

                if self.randomize_torques:
                    self.torques *= self.randomize_torques_factors

                self.torques = torch.clip(
                    self.torques, -self.max_effort, self.max_effort
                )
                self.gym.set_dof_actuation_force_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.torques)
                )
                self.gym.simulate(self.sim)
                if self.device == "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
        else:
            print(f"Unknown PD control type {self._pd_control}. Exiting.")
            exit()
            # forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            # force_tensor = gymtorch.unwrap_tensor(forces)
            # self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def _physics_step(self):
        if self._pd_control == "isaac":
            super()._physics_step()

    def post_physics_step(self):
        self.progress_buf += 1
        self.common_step_counter += 1

        self.projected_gravity[:] = quat_rotate_inverse(
            self._root_states[: self.num_envs, 3:7], self.gravity_vec
        )

        # Computing average velocities over the last gait

        # Shift back.
        self.velocities_history[
            :, : 6 * (self.num_steps_per_period - 1)
        ] = self.velocities_history[:, 6 : 6 * self.num_steps_per_period]

        # add
        self.velocities_history[:, -6:] = self._root_states[:, 7:13]

        # reshape velocities_history so that its (num_envs, num_steps_per_period, 6)
        self.velocities_history_reshaped = self.velocities_history.view(
            self.num_envs, self.num_steps_per_period, 6
        )

        # Compute average velocities for each environment
        self.avg_velocities = torch.mean(self.velocities_history_reshaped, dim=1)

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        # push robots
        if self.cfg["env"]["pushRobots"] and (
            self.common_step_counter % self.push_interval == 0
        ):
            self._push_robots()

        return

    def render(self, sync_frame_time=False):
        # if self.viewer:
        #     self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.duckling_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.duckling_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg["env"]["maxPushVelXy"]
        self._root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self._root_states)
        )

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._duckling_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._duckling_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2]
        )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


# terrain generator
from isaacgym.terrain_utils import *


class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.type = cfg["terrainType"]
        if self.type in ["none", "plane"]:
            return
        self.horizontal_scale = cfg["horizontalScale"]
        self.vertical_scale = cfg["verticalScale"]
        self.border_size = cfg["borderSize"]
        self.num_per_env = cfg["numPerEnv"]
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.min_height = cfg["minHeight"]
        self.max_height = cfg["maxHeight"]
        self.step = cfg["step"]
        self.platform_size = cfg["platformSize"]
        self.step_height_range = cfg["stepHeightRange"]
        self.step_width = cfg["stepWidth"]
        self.proportions = [
            np.sum(cfg["terrainProportions"][: i + 1])
            for i in range(len(cfg["terrainProportions"]))
        ]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = (
            int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
        )

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(
                num_robots, num_terrains=self.env_cols, num_levels=self.env_rows
            )
        else:
            self.randomized_terrain()
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw,
            self.horizontal_scale,
            self.vertical_scale,
            cfg["slopeTreshold"],
        )

    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )
            random_uniform_terrain(
                terrain,
                min_height=self.min_height,
                max_height=self.max_height,
                step=self.step,
                downsampled_scale=0.2,
            )
            # choice = np.random.uniform(0, 1)
            # if choice < 0.1:
            #     if np.random.choice([0, 1]):
            #         pyramid_sloped_terrain(
            #             terrain,
            #             np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]),
            #             self.platform_size,
            #         )
            #         random_uniform_terrain(
            #             terrain,
            #             min_height=self.min_height,
            #             max_height=self.max_height,
            #             step=self.step,
            #             downsampled_scale=0.2,
            #         )
            #     else:
            #         pyramid_sloped_terrain(
            #             terrain,
            #             np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]),
            #             self.platform_size,
            #         )
            # elif choice < 0.6:
            #     # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
            #     step_height = np.random.choice(self.step_height_range)
            #     pyramid_stairs_terrain(
            #         terrain,
            #         step_width=self.step_width,
            #         step_height=step_height,
            #         platform_size=self.platform_size,
            #     )
            # elif choice < 1.0:
            #     discrete_obstacles_terrain(
            #         terrain, 0.15, 1.0, 2.0, 40, platform_size=self.platform_size
            #     )

            self.height_field_raw[
                start_x:end_x, start_y:end_y
            ] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
            env_origin_z = (
                np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            )
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    # TODO
    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=self.vertical_scale,
                    horizontal_scale=self.horizontal_scale,
                )
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
                    random_uniform_terrain(
                        terrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.025,
                        downsampled_scale=0.2,
                    )
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(
                        terrain,
                        step_width=0.31,
                        step_height=step_height,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(
                        terrain,
                        discrete_obstacles_height,
                        1.0,
                        2.0,
                        40,
                        platform_size=3.0,
                    )
                else:
                    stepping_stones_terrain(
                        terrain,
                        stone_size=stepping_stones_size,
                        stone_distance=0.1,
                        max_height=0.0,
                        platform_size=3.0,
                    )

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[
                    start_x:end_x, start_y:end_y
                ] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
                env_origin_z = (
                    np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
                )
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets, dof_axis):
    # type: (Tensor, int, List[int], List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_size *= joint_obs_size  # Added
    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif dof_size == 1:
            axis = torch.tensor(
                dof_axis[j * 3 : (j * 3) + 3],
                dtype=joint_pose.dtype,
                device=pose.device,
            )
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size) : ((j + 1) * joint_obs_size)] = joint_dof_obs

    assert (num_joints * joint_obs_size) == dof_obs_size

    return dof_obs


@torch.jit.script
def compute_duckling_observations(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    local_root_obs,
    root_height_obs,
    dof_obs_size,
    dof_offsets,
    dof_axis,
    projected_gravity,
    actions,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int], List[int], Tensor, Tensor) -> Tensor

    # realistic observations
    obs = torch.cat(
        (
            projected_gravity,
            dof_pos,
            dof_vel,
            actions,
        ),
        dim=-1,
    )
    return obs

    # Below is full obs (original implementation)
    # If using full obs, need to edit dof_obs_size and num_obs in props.yaml

    # root_h = root_pos[:, 2:3]
    # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    # if local_root_obs:
    #     root_rot_obs = quat_mul(heading_rot, root_rot)
    # else:
    #     root_rot_obs = root_rot
    # root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    # if not root_height_obs:
    #     root_h_obs = torch.zeros_like(root_h)
    # else:
    #     root_h_obs = root_h

    # local_root_vel = quat_rotate(heading_rot, root_vel)
    # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    # root_pos_expand = root_pos.unsqueeze(-2)
    # local_key_body_pos = key_body_pos - root_pos_expand

    # heading_rot_expand = heading_rot.unsqueeze(-2)
    # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    # flat_end_pos = local_key_body_pos.view(
    #     local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
    #     local_key_body_pos.shape[2],
    # )
    # flat_heading_rot = heading_rot_expand.view(
    #     heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
    #     heading_rot_expand.shape[2],
    # )
    # local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    # flat_local_key_pos = local_end_pos.view(
    #     local_key_body_pos.shape[0],
    #     local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    # )

    # dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, dof_axis)

    # obs = torch.cat(
    #     (
    #         root_h_obs,
    #         root_rot_obs,
    #         local_root_vel,
    #         local_root_ang_vel,
    #         dof_obs,
    #         dof_vel,
    #         flat_local_key_pos,
    #     ),
    #     dim=-1,
    # )

    # return obs


@torch.jit.script
def compute_duckling_observations_max(
    body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_duckling_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_duckling_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = fall_height  # torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated
