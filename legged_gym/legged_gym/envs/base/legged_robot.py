# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import random
import torch
from torch import Tensor
import torchvision
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


from legged_gym.envs.g1.g1_utils import (
    MotionLib, 
    load_imitation_dataset
)



def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.amp_obs_type = self.cfg.amp.obs_type
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history

        self.actor_obs_length = self.cfg.env.num_observations

        self._init_buffers()
        self._prepare_reward_function()
        self.num_amp_obs = cfg.amp.num_obs
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Tier A DR: action low-pass filter
        if getattr(self.cfg.domain_rand, 'randomize_action_filter', False):
            alpha = torch_rand_float(self.cfg.domain_rand.action_filter_alpha_min, 1.0,
                                     (self.num_envs, 1), device=self.device)
            self.actions = alpha * self.actions + (1.0 - alpha) * self.last_actions

        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)

        # Tier A DR: random action delay per env
        if getattr(self.cfg.domain_rand, 'randomize_action_delay', False):
            max_delay = self.cfg.domain_rand.action_delay_steps
            env_delay = torch.randint(0, max_delay + 1, (self.num_envs, 1), device=self.device)
            for i in range(self.cfg.control.decimation):
                mask = (i < env_delay).float()
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * mask

        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[_]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def get_amp_observations(self):
        """ with keys
        """

        return self.dof_pos.clone()



    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.catchstep -= 1
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])

        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]

        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity[:] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        # Stage 2: ball-visible pitch disturbance
        if getattr(self.cfg.domain_rand, 'randomize_ball_visible_pitch_disturb', False):
            # Detect ball becoming visible (flying transition: 0→1)
            end_target_local_check = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos)
            ball_flying = ((end_target_local_check[:,0] > 0.05) & (end_target_local_check[:,0] < 3.4) &
                           (end_target_local_check[:,1] > -2.0) & (end_target_local_check[:,1] < 2.0) &
                           (end_target_local_check[:,2] < 1.8))
            just_became_visible = ball_flying & ~self._ball_was_visible
            self._ball_was_visible = ball_flying
            # Reset step counter for newly visible envs
            self._ball_visible_step[just_became_visible] = 0
            self._ball_visible_push_active[just_became_visible] = (
                torch.rand(just_became_visible.sum().item(), device=self.device)
                < self.cfg.domain_rand.ball_visible_push_prob)
            # Increment step counter for active envs
            ball_window = self._ball_visible_step < int(self.cfg.domain_rand.ball_visible_push_duration / self.dt)
            self._ball_visible_step[ball_flying] += 1
            # Inject pitch angular velocity disturbance
            active = ball_flying & ball_window & self._ball_visible_push_active
            if active.any():
                p_min = self.cfg.domain_rand.ball_visible_pitch_impulse_min
                p_max = self.cfg.domain_rand.ball_visible_pitch_impulse_max
                impulse = torch_rand_float(p_min, p_max, (self.num_envs,), device=self.device)
                # Apply to root pitch angular velocity (index 10 in root_states)
                self.root_states[active, 10] += impulse[active]
                # Optional lateral push
                push = self.cfg.domain_rand.ball_visible_push_vel
                self.root_states[active, 7:9] += torch_rand_float(-push, push, (active.sum(), 2), device=self.device)
        
        
        # compute contact related quantities

        # compute joint powers
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)
        
        self._post_physics_step_callback()


        balllocal =  self.ball_states[:, 0] - self.env_origins[:, 0]
        approachidx = ((balllocal < 0.5) & (balllocal > 0.1) & (self.ball_states[:,7]  - self.ball_vel < 2.0)).nonzero(as_tuple=False).flatten()
        self.end_target[approachidx, :] = self.ball_states[approachidx, :3].clone()
        self.end_target[:, 0] = torch.clip(self.end_target[:, 0], min = self.env_origins[:, 0] + 0.1, max = self.env_origins[:, 0] + 1.0)

        hand_pos = self._get_hand_pos()
        hand_pos_l, hand_pos_r =  hand_pos[:,0,:], hand_pos[:,1,:]

        region0_dis = torch.norm(self.end_target[self.end_regions == 0] - hand_pos_l[self.end_regions == 0], dim = 1)
        region1_dis = torch.norm(self.end_target[self.end_regions == 1] - hand_pos_r[self.end_regions == 1], dim = 1)
        region2_dis = torch.norm(self.end_target[self.end_regions == 2] - hand_pos_l[self.end_regions == 2], dim = 1)
        region3_dis = torch.norm(self.end_target[self.end_regions == 3] - hand_pos_r[self.end_regions == 3], dim = 1)
        region4_dis = torch.norm(self.end_target[self.end_regions == 4] - hand_pos_l[self.end_regions == 4], dim = 1)
        region5_dis = torch.norm(self.end_target[self.end_regions == 5] - hand_pos_r[self.end_regions == 5], dim = 1)

        self.dist[self.end_regions == 0] = region0_dis
        self.dist[self.end_regions == 1] = region1_dis
        self.dist[self.end_regions == 2] = region2_dis
        self.dist[self.end_regions == 3] = region3_dis
        self.dist[self.end_regions == 4] = region4_dis
        self.dist[self.end_regions == 5] = region5_dis        


        # compute observations, rewards, resets, ...
        self.compute_reward()

        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        termination_privileged_obs = self.compute_termination_observations(env_ids)

        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """


        knee_threshold = float(getattr(self.cfg, 'termination', type('T',(),{'knee_height_threshold':0.10})).knee_height_threshold)
        self.reset_buf = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim=-1).values < knee_threshold
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        gravity_term_cfg = getattr(self.cfg, 'termination', None)
        grav_threshold = float(getattr(gravity_term_cfg, 'gravity_threshold', 0.8)) if gravity_term_cfg is not None else 0.8
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > grav_threshold, dim=1)
        sharpforce_buf = torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) > 1.5 * self.cfg.rewards.max_contact_force

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf
        self.reset_buf |= sharpforce_buf



    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        self.success_rate[env_ids, 0] += self.success_flag[env_ids]
        self.success_rate[env_ids, 1] += 1

        reachcount = self.success_rate[:,1] > 9    
        self.success_rate[reachcount,2] = self.success_rate[reachcount, 0] / self.success_rate[reachcount, 1]
        self.success_rate[reachcount, 0:2] *= 0.        


        self.refresh_actor_rigid_shape_props(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dof), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_motor_strength', False):
            low, high = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[env_ids] = torch_rand_float(low, high, (len(env_ids), self.num_actions), device=self.device)
        # Stage 1: hip-pitch-specific actuator DR (overrides global kp/kd/motor for hp joints)
        if getattr(self.cfg.domain_rand, 'randomize_hip_pitch_actuator', False):
            for hp_idx in self._hp_indices:
                # Kp override
                lo, hi = self.cfg.domain_rand.hip_pitch_kp_scale
                self.Kp_factors[env_ids, hp_idx] = torch_rand_float(lo, hi, (len(env_ids),), device=self.device)
                # Kd override
                lo, hi = self.cfg.domain_rand.hip_pitch_kd_scale
                self.Kd_factors[env_ids, hp_idx] = torch_rand_float(lo, hi, (len(env_ids),), device=self.device)
                # Motor strength override
                lo, hi = self.cfg.domain_rand.hip_pitch_motor_strength_scale
                self.motor_strength[env_ids, hp_idx] = torch_rand_float(lo, hi, (len(env_ids),), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.

        # Q1 feet separation episode metrics
        if self.feet_sep_enabled and len(env_ids) > 0:
            n = max(1, len(env_ids))
            valid_steps = self.ep_steps_foot[env_ids].clamp(min=1)
            self.extras["episode"]["feet_cross_episode_rate"] = self.ep_feet_cross[env_ids].mean().item()
            self.extras["episode"]["feet_too_close_step_rate"] = (self.ep_feet_too_close[env_ids] / valid_steps).mean().item()
            self.extras["episode"]["min_foot_separation"] = self.ep_min_foot_sep[env_ids].mean().item()
            self.extras["episode"]["mean_foot_separation"] = (self.ep_foot_sep_sum[env_ids] / valid_steps).mean().item()
            # Reset episode buffers
            self.ep_feet_too_close[env_ids] = 0.
            self.ep_feet_cross[env_ids] = 0.
            self.ep_min_foot_sep[env_ids] = 999.0
            self.ep_foot_sep_sum[env_ids] = 0.
            self.ep_steps_foot[env_ids] = 0.

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

            
        if (self.common_step_counter - self.last_step_counter) >500:
            
            self.startstep = 50 - random.randint(3, 10)

            self.curriculumupdate = int(torch.mean(self.episode_length_buf[env_ids].float()) / 50.)

            # self.curriculumsigma = torch.mean(self.success_rate[:,2]) * 5.0 + self.cfg.rewards.catch_sigma

            self.command_ranges[:, 0] =  torch.clip(self.command_ranges[:, 0] - 0.3 * self.curriculumupdate, self.command_bound[:,0], self.command_bound[:,1])
            self.command_ranges[:, 1] =  torch.clip(self.command_ranges[:, 1] + 0.3 * self.curriculumupdate,  self.command_bound[:,0], self.command_bound[:,1])
            self.command_ranges[:, 2] =  torch.clip(self.command_ranges[:, 2] - 0.3 * self.curriculumupdate, self.command_bound[:,2], self.command_bound[:,3])
            self.command_ranges[:, 3] =  torch.clip(self.command_ranges[:, 3] + 0.3 * self.curriculumupdate, self.command_bound[:,2], self.command_bound[:,3])

            # for i in range(self.num_dof):

            #     m = (self.hard_dof_pos_limits[i, 0] + self.hard_dof_pos_limits[i, 1]) / 2
            #     r = self.hard_dof_pos_limits[i, 1] - self.hard_dof_pos_limits[i, 0]
            #     self.curriculum_dof_pos_limits[i, 0] = m - 0.5 * r * torch.clip((1.1 - 0.2 * self.curriculumupdate), 0.9, 1.1)
            #     self.curriculum_dof_pos_limits[i, 1] = m + 0.5 * r * torch.clip((1.1 - 0.2 * self.curriculumupdate), 0.9, 1.1)

            self.last_step_counter = self.common_step_counter


        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.



        if "eereach" in self.reward_scales:
            self.reward_scales["eereach"] = self.eereach_init * (1 + 0.5 * self.curriculumupdate)
        if "success" in self.reward_scales:
            self.reward_scales["success"] = self.success_init * (1 + 0.5 * self.curriculumupdate)
        if "stopball" in self.reward_scales:
            self.reward_scales["stopball"] = self.stop_init * (1 + 0.5 * self.curriculumupdate)

        if self.curriculumupdate > 1.0:
             self.reward_scales["dof_pos_limits"] = self.dof_pos_init * 2.0
             self.reward_scales["torque_limits"]  = self.torque_init  * 2.0

        if self.curriculumupdate > 2.0:
             self.reward_scales["dof_pos_limits"] = self.dof_pos_init * 3.0
             self.reward_scales["torque_limits"]  = self.torque_init  * 3.0


        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """

        hand_pos = self._get_hand_pos()
        hand_pos_l, hand_pos_r = quat_rotate_inverse(self.base_quat, hand_pos[:,0,:]- self.torso_pos), quat_rotate_inverse(self.base_quat, hand_pos[:,1,:]- self.torso_pos)

        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        end_target_local = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos) * initial_vanish        


        flying = ((end_target_local[:,0] > 0.05) & (end_target_local[:,0] < 3.4) & (end_target_local[:,1] > -2.0) & (end_target_local[:,1] < 2.0) & (end_target_local[:,2] < 1.8) & (self.catchstep > 0.) & ((end_target_local[:,0] < self.ball_last[:,0]) | (self.ball_last[:,0] == 0.))).view(-1, 1)
        random_vanish = (self.catchstep > self.vanish_step).view(-1, 1)
        self.ball_last = end_target_local

        current_obs = torch.cat((   
                                    end_target_local, # * ball_in_camera_fov,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.end_regions.unsqueeze(-1) / 3.,
                                    quat_rotate_inverse(self.base_quat, self.end_target - self.torso_pos),
                                    quat_rotate_inverse(self.base_quat, self.ball_states[:,7:10]) * self.obs_scales.ball_vel,
                                    hand_pos_r,
                                    hand_pos_l,
                                    self.dist.unsqueeze(-1),
                                    ),dim=-1)

        # add noise if needed
        current_actor_obs = torch.clone(current_obs[:, :self.num_one_step_obs])

        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[:(self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions)]
            current_actor_obs[:, :self.num_ballobs] =  current_actor_obs[:, :self.num_ballobs] * flying * random_vanish
        else:
            current_actor_obs[:, :self.num_ballobs] =  current_actor_obs[:, :self.num_ballobs] * flying
        # Tier A DR: ball obs noise
        if getattr(self.cfg.domain_rand, 'randomize_ball_obs_noise', False) and not self.add_noise:
            noise_scale = self.cfg.domain_rand.ball_obs_noise
            current_actor_obs[:, :self.num_ballobs] += (2.0 * torch.rand_like(current_actor_obs[:, :self.num_ballobs]) - 1.0) * noise_scale
        # Tier A DR: ball obs dropout
        if getattr(self.cfg.domain_rand, 'randomize_ball_obs_dropout', False):
            dropout_pct = self.cfg.domain_rand.ball_obs_dropout_pct
            mask = (torch.rand(self.num_envs, 1, device=self.device) > dropout_pct).float()
            current_actor_obs[:, :self.num_ballobs] *= mask

        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)

        # Stage 1: history ball frame dropout
        if getattr(self.cfg.domain_rand, 'randomize_history_ball_dropout', False):
            h_dropout_pct = self.cfg.domain_rand.history_ball_dropout_pct
            # In obs_buf shape (N, 10*frames), ball obs is at position [0:3] in each frame
            for h in range(self.actor_history_length - 1):  # don't drop current frame
                mask = (torch.rand(self.num_envs, 1, device=self.device) > h_dropout_pct).float()
                start = h * self.num_one_step_obs
                self.obs_buf[:, start:start + self.num_ballobs] *= mask

        self.privileged_obs_buf = current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        
        hand_pos = self._get_hand_pos()
        hand_pos_l, hand_pos_r = quat_rotate_inverse(self.base_quat, hand_pos[:,0,:]- self.torso_pos), quat_rotate_inverse(self.base_quat, hand_pos[:,1,:]- self.torso_pos)
        

        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        end_target_local = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos) * initial_vanish

        current_obs = torch.cat((   
                                    end_target_local, # * ball_in_camera_fov,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.end_regions.unsqueeze(-1) / 3.,
                                    quat_rotate_inverse(self.base_quat, self.end_target - self.torso_pos),
                                    quat_rotate_inverse(self.base_quat, self.ball_states[:,7:10]) * self.obs_scales.ball_vel,
                                    hand_pos_r,
                                    hand_pos_l,
                                    self.dist.unsqueeze(-1),
                                    ),dim=-1)



        return current_obs[env_ids]
    
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        start = time()
        print("*"*80)
        print("Start creating ground...")
        self._create_ground_plane()
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

        

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]
                
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.curriculum_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

                m = (self.hard_dof_pos_limits[i, 0] + self.hard_dof_pos_limits[i, 1]) / 2
                r = self.hard_dof_pos_limits[i, 1] - self.hard_dof_pos_limits[i, 0]
                self.curriculum_dof_pos_limits[i, 0] = m - 0.5 * r * 1.1
                self.curriculum_dof_pos_limits[i, 1] = m + 0.5 * r * 1.1


        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def refresh_actor_rigid_body_props(self, env_ids):
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload[env_ids] = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (len(env_ids), 1), device=self.device)
            
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement[env_ids] = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (len(env_ids), 3), device=self.device)
            
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            rigid_body_props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            rigid_body_props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
            
            if self.cfg.domain_rand.randomize_link_mass:
                rng = self.cfg.domain_rand.link_mass_range
                for i in range(1, len(rigid_body_props)):
                    scale = np.random.uniform(rng[0], rng[1])
                    rigid_body_props[i].mass = scale * self.default_rigid_body_mass[i]
            
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, rigid_body_props, recomputeInertia=True)



    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self._randomize_balls()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        # Q1 feet separation tracking (config-driven)
        if getattr(self.cfg.rewards, 'feet_sep_enabled', False):
            self._update_feet_separation()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller

        actions_scaled = actions * self.action_scale_vec
        self.joint_pos_target = self.default_dof_poses + actions_scaled
                
        self.joint_pos_target[self.catchstep > self.startstep] = self.init_dof_pos[self.catchstep > self.startstep]
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques = torques * self.motor_strength
        torques = torques + self.actuation_offset + self.joint_injection
        # Tier A DR: torque noise
        if getattr(self.cfg.domain_rand, 'randomize_torque_noise', False):
            noise_pct = self.cfg.domain_rand.torque_noise_pct
            noise = (2.0 * torch.rand_like(torques) - 1.0) * noise_pct * self.torque_limits.unsqueeze(0)
            torques = torques + noise
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)

        if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
            self.dof_pos[env_ids] = self.dof_pos[torch.randint(0, self.num_envs, (len(env_ids),), device=self.dof_pos.device)]
        else:    
            if self.cfg.domain_rand.randomize_initial_joint_pos:
                init_dos_pos = self.standpos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
                init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
                self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
            else:
                self.dof_pos[env_ids] = self.standpos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        # Stage 1: hip_pitch / ankle_pitch initial position noise
        if getattr(self.cfg.domain_rand, 'randomize_backward_lean_reset', False):
            for jidx in self._hp_indices:
                noise = self.cfg.domain_rand.hip_pitch_init_noise
                self.dof_pos[env_ids, jidx] += torch_rand_float(-noise, noise, (len(env_ids),), device=self.device)
            for jidx in self._ap_indices:
                noise = self.cfg.domain_rand.ankle_pitch_init_noise
                self.dof_pos[env_ids, jidx] += torch_rand_float(-noise, noise, (len(env_ids),), device=self.device)
            self.dof_pos[env_ids] = torch.clip(self.dof_pos[env_ids], dof_lower, dof_upper)

        self.init_dof_pos[env_ids] = self.dof_pos[env_ids].clone()


        self.dof_vel[env_ids] = 0.

        env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)

        env_ids_int32 = 2 * env_ids.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # self.root_states[env_ids, 1:2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
        # self.root_states[env_ids, 2:3] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        # base velocities (config-driven: Q1 goalkeeper zeros them)
        if getattr(self.cfg.domain_rand, 'randomize_reset_velocity', True):
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=self.device)
        else:
            self.root_states[env_ids, 7:13] = 0.0
        # Stage 1: backward lean reset DR — root pitch noise with backward bias
        if getattr(self.cfg.domain_rand, 'randomize_backward_lean_reset', False):
            pitch_deg = self.cfg.domain_rand.root_pitch_noise_deg
            pitch_rad_range = pitch_deg * 3.14159 / 180.0
            bias = self.cfg.domain_rand.root_pitch_backward_bias
            # Sample pitch: biased toward backward (positive pitch = backward lean)
            pitch = torch_rand_float(-pitch_rad_range, pitch_rad_range, (len(env_ids),), device=self.device)
            # bias: flip some samples to the backward side
            backward_mask = torch.rand(len(env_ids), device=self.device) < bias
            pitch[backward_mask] = torch.abs(pitch[backward_mask])  # always backward
            # Convert pitch to quaternion (rotation about Y axis)
            half_pitch = pitch * 0.5
            q_w = torch.cos(half_pitch)
            q_y = torch.sin(half_pitch)
            # quaternion: w, x, y, z
            self.root_states[env_ids, 3] = q_w
            self.root_states[env_ids, 4] = 0.0
            self.root_states[env_ids, 5] = q_y
            self.root_states[env_ids, 6] = 0.0
            # Root pitch angular velocity noise
            pv = self.cfg.domain_rand.root_pitch_vel_noise
            self.root_states[env_ids, 10] = torch_rand_float(-pv, pv, (len(env_ids),), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        ball_ids = env_ids
        self.ball_states[ball_ids] = self.base_init_state

        ball_vel = self.assign_ball_states(ball_ids)

        self.ball_states[ball_ids, :3] = self.ball_start[ball_ids, :]

        self.ball_vel[ball_ids] = ball_vel[:,0]
        self.has_in_air[ball_ids] = False
        self.static_obs[ball_ids] *= 0. 
        self.stop_flag[ball_ids] = 0.
        self.success_flag[ball_ids] = 0.
        self.dist[ball_ids] = 5.0
        
        self.ball_last[ball_ids] = 0.
        self.vanish_step[ball_ids] = torch.randint(low=0, high=30, size=(len(ball_ids),), device=self.device)

        # move towards the robot
        self.ball_states[ball_ids, 7:13] *= 0. 
        self.ball_states[ball_ids, 7:10] = ball_vel
        
        all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
        env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(all_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))

    def _randomize_balls(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        
        airforce = self.compute_drag_force()
        airforce += torch.empty_like(airforce).uniform_(-0.5, 0.5)
        force_tensor = torch.zeros([self.num_envs, self.num_bodies + 1, 3], device=self.device)
        force_tensor[:, -1, :3] = airforce
        force_tensor = gymtorch.unwrap_tensor(force_tensor)
        self.gym.apply_rigid_body_force_tensors(self.sim, force_tensor)

        if (self.common_step_counter % self.cfg.domain_rand.ball_interval == 0):
            max_vel = self.cfg.domain_rand.max_ball_vel
            self.ball_states[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y
            all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))



    def compute_drag_force(self, rho=1.225, drag_coeff=0.47, radius=0.1):
        velocity = self.ball_states[:, 7:10]
        cross_area = torch.pi * (radius ** 2)
        speed = torch.norm(velocity, dim=1, keepdim=True)  # Magnitude of velocity
        return -0.5 * rho * drag_coeff * cross_area * speed * velocity


    def assign_ball_states(self, ball_ids, g=9.81):

        dtype = torch.float
        device = self.ball_start.device

        self.catchstep[ball_ids] = 50 * torch.ones(len(ball_ids), dtype = torch.int, device = self.device)
                
        ball_start_local = torch.stack([
            2.0 * torch.rand(len(ball_ids), dtype=dtype, device=device) + 3.0,
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.init_ranges[1] - self.init_ranges[0]) + self.init_ranges[0],
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.init_ranges[3] - self.init_ranges[2]) + self.init_ranges[2]
        ], dim=1)

        ball_end_local = torch.stack([
            -0.5 * torch.rand(len(ball_ids), dtype=dtype, device=device) - 0.1,
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.command_ranges[ball_ids, 1] - self.command_ranges[ball_ids, 0]) + self.command_ranges[ball_ids, 0],
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.command_ranges[ball_ids, 3] - self.command_ranges[ball_ids, 2]) + self.command_ranges[ball_ids, 2]
        ], dim=1)

        # in world frame
        self.ball_start[ball_ids,:] = ball_start_local + self.env_origins[ball_ids]  # convert the local target cord in world frame 
        self.ball_start[ball_ids, 2] = ball_start_local[:, 2]

        self.ball_end[ball_ids,:] = ball_end_local + self.env_origins[ball_ids] # convert the local target cord in world frame 
        self.ball_end[ball_ids, 2] = ball_end_local[:, 2]

        catch_prop = (0.1 - ball_start_local[:,0:1]) / (ball_end_local[:,0:1] - ball_start_local[:,0:1])

        # Compute velocity
        delta_pos = self.ball_end[ball_ids,:] - self.ball_start[ball_ids,:]


        t_flight = 0.4 + 0.6 * torch.rand((1), dtype=dtype, device=device)

        if self.play:
            t_flight = 0.5 + 0.5 * torch.rand((1), dtype=dtype, device=device)     

        self.end_target[ball_ids,:] = self.ball_start[ball_ids,:] + delta_pos * catch_prop

        ball_vel = torch.empty_like(delta_pos)

        ball_vel[:, 0:2] = delta_pos[:, 0:2] / t_flight
        ball_vel[:, 2] = (delta_pos[:, 2] + 0.5 * g * t_flight**2) / t_flight

        return ball_vel

    def _get_hand_pos(self):
        """Return hand positions with optional body-frame offset (e.g., forearm tip instead of COM)."""
        hand_com = self.rigid_body_states[:, self.hand_indices, :3].clone()  # (N, 2, 3)
        offset_cfg = getattr(self.cfg.asset, 'hand_offset', None)
        if offset_cfg is not None:
            offset = torch.tensor(offset_cfg, device=self.device, dtype=hand_com.dtype)  # (3,)
            if torch.any(offset != 0.0):
                hand_quat = self.rigid_body_states[:, self.hand_indices, 3:7]  # (N, 2, 4)
                hand_com = hand_com + quat_rotate(hand_quat, offset.view(1, 1, 3))
        return hand_com


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\

        noise_vec = torch.zeros( self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:self.num_ballobs] = noise_scales.ball * noise_level
        noise_vec[self.num_ballobs:self.num_ballobs + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[self.num_ballobs + 3: self.num_ballobs + 6] = noise_scales.gravity * noise_level
        noise_vec[self.num_ballobs + 6:(self.num_ballobs + 6 + self.num_dof)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(self.num_ballobs + 6 + self.num_dof):(self.num_ballobs + 6 + 2 * self.num_dof)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(self.num_ballobs + 6 + 2 * self.num_dof):(self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions)] = 0. # previous actions
        return noise_vec




    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        all_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2,13)
        self.root_states, self.ball_states = all_states[:, 0, :], all_states[:,1, :]

    

        all_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_states = all_body_states[:, :-1, :]

        all_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies + 1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = all_contact_forces[:, :-1, :]

        self.ball_contact_forces = all_contact_forces[:, -1:, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.last_step_counter = 0
        self.extras = {}
        # Stage 2: ball-visible pitch disturbance state
        self._ball_visible_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._ball_was_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ball_visible_push_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])
        
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]

        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        

        self.ball_traj = [[], [], [], []]
        self.parabola = [[], [], [], []]

        self.end_target = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)

        mode_weights = getattr(self.cfg.env, 'mode_weights', [1, 1, 1, 1, 1, 1])
        total_w = sum(mode_weights)
        counts = [max(1, int(self.num_envs * w / total_w)) for w in mode_weights]
        diff = self.num_envs - sum(counts)
        for i in range(abs(diff)):
            counts[i % 6] += 1 if diff > 0 else -1
        regions = []
        for m in range(6):
            regions.append(torch.full((counts[m],), m, dtype=torch.long, device=self.device))
        self.end_regions = torch.cat(regions)
        pcts = [f"{c/self.num_envs*100:.1f}%" for c in counts]
        print(f"[MODE_DIST] weights={mode_weights} → counts={counts} → pct={pcts}")

        command_dict = class_to_dict(self.cfg.commands)

        print(f"[MODE_SPEC] mode | dir   | height_z(m)   | width_y(m)")
        for m in range(6):
            r = command_dict[f"ranges_{m}"]
            print(f"[MODE_SPEC]   {m}  | {'right' if m%2==0 else 'left '} | [{r['height'][0]:.2f}, {r['height'][1]:.2f}]      | [{r['width'][0]:.2f}, {r['width'][1]:.2f}]")

        # Initialize an empty tensor for command ranges
        num_envs = self.num_envs
        self.command_ranges = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.command_bound  = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.init_ranges  = torch.zeros((4), dtype=torch.float32, device=self.device)
        # For each environment, set the appropriate ranges based on end_region
        for env_idx in range(num_envs):
            region = self.end_regions[env_idx].item() if env_idx < len(self.end_regions) else 0
            region_key = f"ranges_{region}"
            
            # Get the ranges for this region
            region_ranges = command_dict[region_key]
            
            # Assign height and width ranges
            self.command_ranges[env_idx, 0] = region_ranges["width"][0]   # width_0
            self.command_ranges[env_idx, 1] = region_ranges["width"][1]   # width_1
            self.command_ranges[env_idx, 2] = region_ranges["height"][0]  # height_0
            self.command_ranges[env_idx, 3] = region_ranges["height"][1]  # height_1

            if self.play:
                self.command_ranges[env_idx, 0] = region_ranges["evalw"][0]   # width_0
                self.command_ranges[env_idx, 1] = region_ranges["evalw"][1]   # width_1
                self.command_ranges[env_idx, 2] = region_ranges["evalh"][0]  # height_0
                self.command_ranges[env_idx, 3] = region_ranges["evalh"][1]  # height_1



            self.command_bound[env_idx, 0] = region_ranges["maxw"][0]   # width_0
            self.command_bound[env_idx, 1] = region_ranges["maxw"][1]   # width_1
            self.command_bound[env_idx, 2] = region_ranges["maxh"][0]  # height_0
            self.command_bound[env_idx, 3] = region_ranges["maxh"][1]  # height_1
        
        self.init_ranges[0] = command_dict["ranges_1"]["maxw"][0]
        self.init_ranges[1] = command_dict["ranges_0"]["maxw"][1]
        self.init_ranges[2] = command_dict["ranges_4"]["maxh"][0]
        self.init_ranges[3] = command_dict["ranges_2"]["maxh"][1]




        self.catchstep = 50 * torch.ones(self.num_envs, dtype = torch.int, device= self.device)
        self.dist = 5 * torch.ones(self.num_envs, dtype = torch.float, device= self.device)
        self.ball_vel = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.has_in_air = torch.zeros(self.num_envs, dtype = torch.bool, device= self.device)
        self.static_obs = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.stop_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.success_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.success_rate = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)


        self.ball_start = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.ball_end = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)        
        self.ball_last = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)   
        self.vanish_step = torch.randint(low=0, high=30, size=(self.num_envs,), device=self.device)
        self.sr = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)  
        self.startstep = 50 - random.randint(3, 10)


        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        # self.standpos = self.default_dof_pos
        self.standpos = torch.tensor([self.cfg.init_state.init_pos], dtype=torch.float32, device=self.default_dof_pos.device)
        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs,1)
        self.init_dof_pos = self.default_dof_poses.clone()

        # Per-joint action_scale (config-driven, default = uniform)
        self.action_scale_vec = torch.ones(self.num_dof, dtype=torch.float, device=self.device) * self.cfg.control.action_scale
        per_joint_scales = getattr(self.cfg.control, 'per_joint_action_scale', None)
        if per_joint_scales is not None:
            for joint_name, scale in per_joint_scales.items():
                if joint_name in self.dof_names:
                    idx = self.dof_names.index(joint_name)
                    self.action_scale_vec[idx] = scale
                    print(f"[per_joint_action_scale] {joint_name}: {self.cfg.control.action_scale:.4f} -> {scale:.4f}")

        # Q1 goalkeeper smoke: capture actual simulator dof_pos as PD target
        # (avoids target/actual mismatch that causes feet-lift on reset)
        if getattr(self.cfg.init_state, 'capture_default_dof_pos_from_sim', False):
            captured = self.dof_pos[0].detach().clone()
            # Apply per-joint overrides from config (e.g. force symmetry)
            override = getattr(self.cfg.init_state, 'default_dof_pos_override', {})
            for name, val in override.items():
                if name in self.dof_names:
                    idx = self.dof_names.index(name)
                    captured[idx] = val
            self.standpos = captured.unsqueeze(0)
            self.default_dof_pos = captured.unsqueeze(0)
            self.default_dof_poses = captured.unsqueeze(0).repeat(self.num_envs, 1)
            self.init_dof_pos = self.default_dof_poses.clone()
            # Apply torque limit scaling AFTER capture (config-driven)
            tq_scale = getattr(self.cfg.init_state, 'torque_limits_scale', {})
            for name, scale in tq_scale.items():
                if name in self.dof_names:
                    self.torque_limits[self.dof_names.index(name)] *= scale
            print(f"[capture_default_dof_pos] knee: L={captured[3]:.4f} R={captured[9]:.4f} "
                  f"hip: L={captured[0]:.4f} R={captured[6]:.4f} "
                  f"ankle: L={captured[4]:.4f} R={captured[10]:.4f} "
                  f"overrides={list(override.keys()) if override else 'none'} "
                  f"tq_scale={list(tq_scale.keys()) if tq_scale else 'none'} "
                  f"root_z={self.cfg.init_state.pos[2]:.4f}")



        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dof), device=self.device)
        if getattr(self.cfg.domain_rand, 'randomize_motor_strength', False):
            low, high = self.cfg.domain_rand.motor_strength_range
            self.motor_strength = torch_rand_float(low, high, (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)

        # --- DR log (once) ---
        dr = self.cfg.domain_rand
        print(f"[DR] randomize_friction={dr.randomize_friction} range={list(dr.friction_range)}")
        print(f"[DR] randomize_payload_mass={dr.randomize_payload_mass} range={list(dr.payload_mass_range)}")
        print(f"[DR] push_robots={dr.push_robots} interval={dr.push_interval_s} max_push_vel_xy={dr.max_push_vel_xy}")
        if getattr(dr, 'randomize_motor_strength', False):
            print(f"[DR] randomize_motor_strength=True range={list(dr.motor_strength_range)}")
        if dr.randomize_kp:
            print(f"[DR] randomize_pd_gain=True kp={list(dr.kp_range)} kd={list(dr.kd_range)}")
        print(f"[DR] motor_strength min={self.motor_strength.min():.3f} max={self.motor_strength.max():.3f} mean={self.motor_strength.mean():.3f}")
        print(f"[DR] Kp_factors min={self.Kp_factors.min():.3f} max={self.Kp_factors.max():.3f} mean={self.Kp_factors.mean():.3f}")
        print(f"[DR] Kd_factors min={self.Kd_factors.min():.3f} max={self.Kd_factors.max():.3f} mean={self.Kd_factors.mean():.3f}")

        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)


        # create mocap dataset
        self.init_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.init_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_base_pos_xy[:] = self.base_init_state[:2] + self.env_origins[:, 0:2]
        self.init_base_quat[:] = self.base_init_state[3:7]



        multidataset, mapping = load_imitation_dataset(self.cfg.dataset.folder.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),self.cfg.dataset.joint_mapping.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.motions = {}
        self.motion_ids = {}
        self.motion_time = {}
        self.motion_dict = {}
        for key in multidataset.keys():
            # Here 'key' will be the dataset's key name, and 'dataset' is the actual data

            # Initialize the MotionLib class for the given dataset
            self.motions[key] = MotionLib(multidataset[key], mapping, self.dof_names, self.keyframe_names,
                                        self.cfg.dataset.frame_rate, self.cfg.dataset.min_time, self.device,
                                        self.amp_obs_type)
            
    

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue

            if name == "eereach":
                self.eereach_init =  self.reward_scales["eereach"]
            if name == "success":
                self.success_init = self.reward_scales["success"]
            if name == "stopball":
                self.stop_init = self.reward_scales["stopball"]
       

            if name == "torque_limits":
                self.torque_init = self.reward_scales["torque_limits"]
            if name == "dof_pos_limits":
                self.dof_pos_init = self.reward_scales["dof_pos_limits"]

            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        # Q1 feet separation tracking (config-driven, disabled by default)
        self.feet_sep_enabled = getattr(self.cfg.rewards, 'feet_sep_enabled', False)
        if self.feet_sep_enabled:
            self.min_foot_sep = float(getattr(self.cfg.rewards, 'min_foot_sep', 0.12))
            self.feet_cross_threshold = float(getattr(self.cfg.rewards, 'feet_cross_threshold', 0.06))
            self.feet_too_close_threshold = float(getattr(self.cfg.rewards, 'feet_too_close_threshold', 0.10))
            # Episode tracking buffers
            self.ep_feet_too_close = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.ep_feet_cross = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.ep_min_foot_sep = torch.ones(self.num_envs, dtype=torch.float, device=self.device) * 999.0
            self.ep_foot_sep_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.ep_steps_foot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            # Per-step foot sep (for reward)
            self.foot_sep_y = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.ball_gravity = self.cfg.env.ball_gravity
        self.num_ballobs = self.cfg.env.num_ballobs
        self.play = self.cfg.env.play

        ### load ball ###
        asset_options.disable_gravity = False
        ball_path = self.cfg.asset.ballfile.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ball_root = os.path.dirname(ball_path)
        ball_file = os.path.basename(ball_path)
        ball_asset = self.gym.load_asset(self.sim, ball_root, ball_file, asset_options)

    

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        # Stage 1/2/3: hip_pitch and ankle_pitch joint indices for targeted DR
        self._hp_indices = [i for i, n in enumerate(self.dof_names) if 'hip_pitch' in n]
        self._ap_indices = [i for i, n in enumerate(self.dof_names) if 'ankle_pitch' in n]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        hand_names  = [s for s in body_names if self.cfg.asset.hand_name in s]

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.ball_handles = []
        self.envs = []
        self.curriculumupdate = 0.
        self.curriculumsigma = self.cfg.rewards.catch_sigma
    
        self.task_scale = 0.
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0.3, 0.3, (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.actor_handles.append(actor_handle)

            ballpos = pos
            ballpos[:0] += torch_rand_float(0.5, 1.0, (1,1), device=self.device).squeeze(1)
            ballpos[:1] += torch_rand_float(-0.5, 0.5, (1,1), device=self.device).squeeze(1)
            ballpos[2] = 1.5
            
            start_pose.p = gymapi.Vec3(*ballpos)
            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pose, "ball", i, 0, 1)
            c = 0.5 + 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.ball_handles.append(ball_handle)

            self.envs.append(env_handle)



        self.left_hip_joint_indices = torch.zeros(len(self.cfg.control.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.control.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_names[i])

        self.camera_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "d435_link")
        
        contact_feet_names = [s for s in body_names if self.cfg.asset.contact_foot_names in s]
        
        self.contact_feet_indices = torch.zeros(len(contact_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(contact_feet_names)):
            self.contact_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], contact_feet_names[i])


        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
        self.curriculum_dof_indices = torch.zeros(len(self.cfg.control.curriculum_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.curriculum_joints)):
            self.curriculum_dof_indices[i] = self.dof_names.index(self.cfg.control.curriculum_joints[i])
            
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.control.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.control.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
            
        self.left_arm_joint_indices = torch.zeros(len(self.cfg.control.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.control.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.right_arm_joints[i])
            
        self.elbow_joint_indices = torch.zeros(len(self.cfg.control.elbow_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.elbow_joints)):
            self.elbow_joint_indices[i] = self.dof_names.index(self.cfg.control.elbow_joints[i])

        self.wrist_joint_indices = torch.zeros(len(self.cfg.control.wrist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.wrist_joints)):
            self.wrist_joint_indices[i] = self.dof_names.index(self.cfg.control.wrist_joints[i])



        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))
            
        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])


        self.upper_body_joint_indices = torch.cat([self.elbow_joint_indices, self.wrist_joint_indices])

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.upper_body_link)
        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.torso_link)
            
        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """

        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(-num_rows//2, num_rows//2), torch.arange(-num_cols//2, num_cols//2))
    
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.ball_interval = np.ceil(self.cfg.domain_rand.ball_interval_s / self.dt)

        
    def _get_base_heights(self, env_ids=None):

        return self.root_states[:, 2].clone()


    #------------ reward functions----------------


    def _reward_eereach(self):

        phase1 = ((self.ball_states[:, 0] - self.env_origins[:, 0] > 1.5) & (self.ball_states[:,7] - self.ball_vel < 2.0)).nonzero(as_tuple=False).flatten()
        
        taskrew = torch.zeros(self.num_envs, dtype = torch.float, device = self.device)

        end_target_local = self.end_target - self.torso_pos

        
        asidegoal = torch.clip(end_target_local[:, 1], -1.0, 1.0)
        asidegoal[torch.abs(asidegoal) < 0.3] = 0.

        verticalgoal = torch.clip(self.torso_pos[:, 2] - torch.clip(self.end_target[:,2], 0.3, 1.2), 0.0, 1.0)

        phase1_rew = 1.0 - (verticalgoal + torch.abs(asidegoal)) / 2.0
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        jump_scale =  3.0 + 3.0 * self.curriculumupdate

        vel_sigma = taskrew.clone()
        vel_sigma[self.end_regions == 0] = 1 + 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 0, self.upper_body_index, 8], 0.0, 3.0)
        vel_sigma[self.end_regions == 1] = 1 - 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 1, self.upper_body_index, 8], -3.0, 0.0)
        vel_sigma[self.end_regions == 4] = 1 + 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 4, self.upper_body_index, 8], 0.0, 3.0)
        vel_sigma[self.end_regions == 5] = 1 - 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 5, self.upper_body_index, 8], -3.0, 0.0)

        vel_sigma[self.end_regions == 2] = 1 + jump_scale * torch.clip(self.rigid_body_states[self.end_regions == 2, self.upper_body_index, 9], 0.0, 3.0)
        vel_sigma[self.end_regions == 3] = 1 + jump_scale * torch.clip(self.rigid_body_states[self.end_regions == 3, self.upper_body_index, 9], 0.0, 3.0)

        vel_sigma[behind] = 2.0

        taskrew =(1 - 1 / (1 + torch.exp(-self.curriculumsigma * (self.dist - self.cfg.rewards.reach_th))))


        taskrew *= vel_sigma

        taskrew[phase1] = phase1_rew[phase1]


        return taskrew * (1 - torch.clip(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), 0., 1.0))

    def _reward_success(self):
        return (self.success_flag + 1.0) * (self.dist < self.cfg.rewards.strict_th)

    

    def _reward_stopball(self):

        changevel = (self.ball_states[:,7] - self.ball_vel > 2.0) & ((self.ball_states[:, 0] - self.env_origins[:, 0]) > 0.0)
        stopped_ids = (changevel |((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.0)).nonzero(as_tuple=False).flatten()
        success_ids = ((self.stop_flag == 0) & changevel) .nonzero(as_tuple=False).flatten()
        rew_stop = 1.0 * (self.stop_flag == 0) * changevel
        self.success_flag[success_ids] = 1.0
        self.stop_flag[stopped_ids] = 1.0

        return rew_stop

    def _reward_feetorientaion(self):

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]  
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]  
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)
        return torch.exp(feet_orientation * -5)

    def _reward_airfeetorientation(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
        inair = torch.sum(foot_contact_forces_z, dim=-1) < 1.

        falling = (self.rigid_body_states[:, self.upper_body_index, 9] < 0.0) & inair
        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]  
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]  
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)

        return torch.exp(feet_orientation * -3) * falling * jump_ids



    def _reward_successland(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
        
        jump = self.root_states[:,2] > getattr(self.cfg.rewards, 'successland_jump_threshold', 0.55)
        
        self.has_in_air = torch.logical_or(self.has_in_air, jump)
        
        has_contact = (foot_contact_forces_z[:, 0] > 1.) & (foot_contact_forces_z[:, 1] > 1.)

        one_feet_contact = (((foot_contact_forces_z[:, 0] >  1.) & (foot_contact_forces_z[:, 1] < 1.)) | ((foot_contact_forces_z[:, 0] <  1.) & (foot_contact_forces_z[:, 1] > 1.))) & (self.has_in_air)

        successful_landings = torch.logical_and(has_contact, self.has_in_air)
        
        air_reward = self.has_in_air.float()  
        landing_reward = successful_landings.float() * 5.0  

        one_feet_punish = one_feet_contact.float() * -1.0

        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)
        
        return (air_reward + landing_reward + one_feet_punish) * jump_ids


    def _reward_feet_slippage(self):
        
        foot_vel = self.rigid_body_states[:,  self.contact_feet_indices, 7:10]  
        contactvel = torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1) > 1.), dim=1)
        return torch.exp(contactvel * -10)

    def _reward_penalize_sharpcontact(self):

        return (torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) >  self.cfg.rewards.max_contact_force) * 1.0 


    def _reward_postorientation(self):
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -3) * behind


    def _reward_penalize_kneeheight(self):
        return (torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim = -1).values < 0.15) * 1.0

    def _reward_postangvel(self):
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -3)* behind

    def _reward_postlinvel(self):
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, 0:1]), dim=1) * -3)* behind

    def _reward_postupperdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.default_dof_pos[:, self.upper_body_joint_indices]), dim=-1)
        reward = torch.exp(mse * -1) 
        return reward * behind

    def _reward_postwaistdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.waist_joint_indices] - self.default_dof_pos[:, self.waist_joint_indices]), dim=-1)
        reward = torch.exp(-3 * mse) 
        return reward * behind

    def _reward_stayonline(self):
        distance = torch.clip(torch.abs(self.torso_pos[:,0] - self.env_origins[:, 0]), 0.2, 1.2) - 0.2   
        return distance

    def _reward_noretreat(self):
        return -1 * torch.clip(self.base_lin_vel[:, 0], -1.0, 0.0)  
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    
    def _reward_deviation_waist_pitch_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.waist_joint_indices[2]], dim=-1)

    # --- Q1 Goalkeeper feet separation (config-driven) ---

    def _update_feet_separation(self):
        """Compute base-frame foot separation for Q1 goalkeeper."""
        base_pos = self.root_states[:, 0:3]
        base_quat = self.root_states[:, 3:7]
        lf_pos = self.rigid_body_states[:, self.contact_feet_indices[0], 0:3]
        rf_pos = self.rigid_body_states[:, self.contact_feet_indices[1], 0:3]

        lf_base = quat_rotate_inverse(base_quat, lf_pos - base_pos)
        rf_base = quat_rotate_inverse(base_quat, rf_pos - base_pos)
        self.foot_sep_y = torch.abs(lf_base[:, 1] - rf_base[:, 1])

        # Episode tracking
        too_close = (self.foot_sep_y < self.feet_too_close_threshold).float()
        crossed = (self.foot_sep_y < self.feet_cross_threshold).float()
        self.ep_feet_too_close += too_close
        self.ep_feet_cross = torch.maximum(self.ep_feet_cross, crossed)
        self.ep_min_foot_sep = torch.minimum(self.ep_min_foot_sep, self.foot_sep_y)
        self.ep_foot_sep_sum += self.foot_sep_y
        self.ep_steps_foot += 1.0

    def _reward_penalty_feet_separation(self):
        """Penalize feet being too close together (Q1 goalkeeper)."""
        viol = torch.clamp(self.min_foot_sep - self.foot_sep_y, min=0.0)
        return viol * viol

    # --- Stage 1/2/3: Failure-proxy penalty rewards ---

    def _reward_root_pitch_penalty(self):
        """Penalize excessive backward root pitch (positive pitch = backward lean)."""
        thresh = getattr(self.cfg.rewards, 'root_pitch_penalty_threshold', 0.3)
        excess = torch.clamp(self.pitch - thresh, min=0.0)
        return excess * excess

    def _reward_pitch_vel_penalty(self):
        """Penalize high pitch angular velocity."""
        thresh = getattr(self.cfg.rewards, 'pitch_vel_threshold', 2.0)
        pitch_vel = torch.abs(self.base_ang_vel[:, 1])  # Y-axis = pitch
        excess = torch.clamp(pitch_vel - thresh, min=0.0)
        return excess

    def _reward_hip_pitch_action_penalty(self):
        """Penalize large hip_pitch actions."""
        if not self._hp_indices:
            return torch.zeros(self.num_envs, device=self.device)
        hp_actions = torch.stack([self.actions[:, i] for i in self._hp_indices], dim=1)
        return torch.sum(torch.abs(hp_actions), dim=1)

    def _reward_hip_pitch_rate_penalty(self):
        """Penalize large hip_pitch action rate (difference from last action)."""
        if not self._hp_indices:
            return torch.zeros(self.num_envs, device=self.device)
        hp_curr = torch.stack([self.actions[:, i] for i in self._hp_indices], dim=1)
        hp_prev = torch.stack([self.last_actions[:, i] for i in self._hp_indices], dim=1)
        return torch.sum(torch.abs(hp_curr - hp_prev), dim=1)

    def _reward_foot_contact_keep(self):
        """Bonus for maintaining foot contact after ball is visible."""
        ball_visible = self._ball_was_visible.float()
        foot_contact_z = self.contact_forces[:, self.contact_feet_indices, 2]
        both_feet = ((foot_contact_z[:, 0] > 1.) & (foot_contact_z[:, 1] > 1.)).float()
        return both_feet * ball_visible

    def _reward_foot_slip_penalty(self):
        """Penalize foot slip (foot velocity when in contact)."""
        foot_vel = self.rigid_body_states[:, self.contact_feet_indices, 7:9]  # x,y vel
        foot_contact_z = self.contact_forces[:, self.contact_feet_indices, 2]
        in_contact = (foot_contact_z > 1.).float().unsqueeze(-1)
        slip = torch.norm(foot_vel, dim=-1) * in_contact.squeeze(-1)
        return torch.sum(slip, dim=1)

    def _reward_upright_penalty(self):
        """Penalise pelvic tilt toward gravity_threshold.

        upright = -projected_gravity[:,2]  → 1.0=vertical, 0.8=terminate.
        danger = clamp((threshold - upright) / deadzone, 0, 1).
        penalty = danger ** 2, masked by ball-visible if configured.
        """
        upright = -self.projected_gravity[:, 2]
        thresh = getattr(self.cfg.rewards, 'upright_threshold', 0.95)
        deadzone = getattr(self.cfg.rewards, 'upright_deadzone', 0.15)
        danger = torch.clamp((thresh - upright) / deadzone, 0.0, 1.0)
        penalty = danger * danger

        if getattr(self.cfg.rewards, 'upright_ball_visible_only', False):
            end_rel = quat_rotate_inverse(self.base_quat, self.ball_states[:, :3] - self.torso_pos)
            ball_flying = ((end_rel[:, 0] > 0.05) & (end_rel[:, 0] < 3.4) &
                           (end_rel[:, 1] > -2.0) & (end_rel[:, 1] < 2.0) &
                           (end_rel[:, 2] < 1.8) & (self.catchstep > 0))
            penalty = penalty * ball_flying.float()

        return penalty

    def _get_highball_active_mask(self):
        """Mask for high-ball modes (2,3) where ball is visible/approaching."""
        highball = (self.end_regions == 2) | (self.end_regions == 3)
        end_rel = quat_rotate_inverse(self.base_quat, self.ball_states[:, :3] - self.torso_pos)
        ball_flying = ((end_rel[:, 0] > 0.05) & (end_rel[:, 0] < 3.4) &
                       (end_rel[:, 1] > -2.0) & (end_rel[:, 1] < 2.0) &
                       (end_rel[:, 2] < 1.8) & (self.catchstep > 0))
        return (highball & ball_flying).float()

    def _reward_highball_jump_height(self):
        """Reward root_z above threshold for high-ball modes when ball is visible."""
        mask = self._get_highball_active_mask()
        if mask.sum() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        thresh = getattr(self.cfg.rewards, 'highball_jump_z_threshold', 0.45)
        rng = getattr(self.cfg.rewards, 'highball_jump_z_range', 0.15)
        jump = torch.clamp((self.root_states[:, 2] - thresh) / rng, 0.0, 1.0)
        return jump * mask

    def _reward_highball_upward_velocity(self):
        """Reward upward root velocity for high-ball modes when ball is visible."""
        mask = self._get_highball_active_mask()
        if mask.sum() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        up_vel = torch.clamp(self.root_states[:, 9], 0.0, 1.0)  # root_vz
        return up_vel * mask

    def _reward_highball_upright_penalty(self):
        """Extra upright penalty for high-ball modes only."""
        mask = self._get_highball_active_mask()
        if mask.sum() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        upright = -self.projected_gravity[:, 2]
        thresh = getattr(self.cfg.rewards, 'upright_threshold', 0.95)
        deadzone = getattr(self.cfg.rewards, 'upright_deadzone', 0.15)
        danger = torch.clamp((thresh - upright) / deadzone, 0.0, 1.0)
        return (danger * danger) * mask
