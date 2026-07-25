"""Ball-free ready-stance environment shared by Q1 and G1.

The goalkeeper base environment owns the simulator setup and domain
randomisation.  This subclass intentionally keeps that infrastructure but
removes ball observations/rewards and the pre-shot PD lock from the control
path, so the learned policy is responsible for holding the ready stance.
"""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class ReadyStandRobot(LeggedRobot):
    """Train a robust, fixed-position goalkeeper ready stance."""

    def _get_noise_scale_vec(self, cfg):
        """Noise vector for the ball-free actor observation layout."""
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        vec = torch.zeros(self.num_one_step_obs, device=self.device)
        vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        vec[3:6] = noise_scales.gravity * noise_level
        dof_start = 6
        vel_start = dof_start + self.num_dof
        vec[dof_start:vel_start] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        vec[vel_start:vel_start + self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        return vec

    def compute_observations(self):
        """Build [ang_vel, gravity, q-q_default, dq, last_action] history."""
        current_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        current_actor_obs = current_obs
        if self.add_noise:
            current_actor_obs = current_actor_obs + (
                2 * torch.rand_like(current_actor_obs) - 1
            ) * self.noise_scale_vec

        self.obs_buf = torch.cat((
            self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length],
            current_actor_obs,
        ), dim=-1)
        self.privileged_obs_buf = current_obs

    def compute_termination_observations(self, env_ids):
        """Return the ready-stand critic observation at terminal states."""
        current_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        return current_obs[env_ids]

    def _compute_torques(self, actions):
        """PD control without the goalkeeper's pre-shot joint-target override."""
        actions_scaled = actions * self.action_scale_vec
        self.joint_pos_target = self.default_dof_poses + actions_scaled
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                self.dof_vel - self.last_dof_vel
            ) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strength + self.actuation_offset + self.joint_injection
        if getattr(self.cfg.domain_rand, "randomize_torque_noise", False):
            noise_pct = self.cfg.domain_rand.torque_noise_pct
            torques += (2.0 * torch.rand_like(torques) - 1.0) * noise_pct * self.torque_limits.unsqueeze(0)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_upright(self):
        return torch.clamp(-self.projected_gravity[:, 2], min=0.0, max=1.0)

    def _reward_base_height(self):
        target = float(self.cfg.rewards.target_base_height)
        sigma = float(self.cfg.rewards.base_height_sigma)
        height = self.rigid_body_states[:, self.upper_body_index, 2]
        return torch.exp(-torch.square(height - target) / sigma)

    def _reward_ready_pose(self):
        sigma = float(self.cfg.rewards.ready_pose_sigma)
        error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
        return torch.exp(-error / sigma)

    def _reward_lin_vel_xy(self):
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=-1)

    def _reward_foot_contact(self):
        force_z = self.contact_forces[:, self.contact_feet_indices, 2]
        return torch.mean((force_z > 5.0).float(), dim=-1)

    def _reward_feet_slip(self):
        foot_vel = self.rigid_body_states[:, self.contact_feet_indices, 7:9]
        force_z = self.contact_forces[:, self.contact_feet_indices, 2]
        contact = (force_z > 5.0).float()
        return torch.sum(torch.square(foot_vel) * contact.unsqueeze(-1), dim=(1, 2))
