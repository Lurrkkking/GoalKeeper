"""G1 Dive Save Config — oracle ball shooting to high-corner target, save primitive.

Upgrades dive_reach from "reach a fixed target" to "save a ball shot at the target zone".
Ball trajectory passes through the target at the catch plane; robot must intercept.

KEY DIFFERENCES from g1_dive_reach:
  - Ball shoots TOWARD target (oracle), not random direction.
  - target = ball's catch-plane position (coupled, not independent).
  - time_remaining from ETA (catchstep), not fixed 0.8s deadline.
  - Rewards focus on contact + deflection + save, not just reaching.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.g1.g1_dive_reach_config import DiveReachRobot
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float
from isaacgym import gymtorch

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class G1DiveSaveCfg(LeggedRobotCfg):
    """G1 dive save — oracle ball to high corner, save primitive."""

    class env(LeggedRobotCfg.env):
        num_envs = 6144
        num_actor_history = 10
        num_actions = 29
        num_dofs = 29
        num_ballobs = 3
        num_target_obs = 4        # local_target_y, target_z, target_side, time_remaining

        num_one_step_observations = 6 + num_ballobs + num_target_obs + num_dofs * 2 + num_actions  # 100
        num_privileged_obs = num_one_step_observations + 3 + 1 + 3 + 3 + 3 + 3 + 1  # 117
        num_observations = num_actor_history * num_one_step_observations  # 1000

        env_spacing = 5.
        send_timeouts = True
        episode_length_s = 3.0
        ball_gravity = True
        play = False
        mode_weights = [1, 1, 0, 0, 0, 0]

    class commands:
        class ranges_0:
            height = [0.25, 0.60]; width = [1.25, 1.45]
            maxh = [0.25, 0.60]; maxw = [1.25, 1.45]
            evalh = [0.25, 0.60]; evalw = [1.25, 1.45]
        class ranges_1:
            height = [0.25, 0.60]; width = [-1.45, -1.25]
            maxh = [0.25, 0.60]; maxw = [-1.45, -1.25]
            evalh = [0.25, 0.60]; evalw = [-1.45, -1.25]
        class ranges_2:
            height = [0.25, 0.60]; width = [1.25, 1.45]
            maxh = [0.25, 0.60]; maxw = [1.25, 1.45]
            evalh = [0.25, 0.60]; evalw = [1.25, 1.45]
        class ranges_3:
            height = [0.25, 0.60]; width = [-1.45, -1.25]
            maxh = [0.25, 0.60]; maxw = [-1.45, -1.25]
            evalh = [0.25, 0.60]; evalw = [-1.45, -1.25]
        class ranges_4:
            height = [0.25, 0.60]; width = [1.25, 1.45]
            maxh = [0.25, 0.60]; maxw = [1.25, 1.45]
            evalh = [0.25, 0.60]; evalw = [1.25, 1.45]
        class ranges_5:
            height = [0.25, 0.60]; width = [-1.45, -1.25]
            maxh = [0.25, 0.60]; maxw = [-1.45, -1.25]
            evalh = [0.25, 0.60]; evalw = [-1.45, -1.25]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.1, 'left_hip_roll_joint': 0.2, 'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3, 'left_ankle_pitch_joint': -0.2, 'left_ankle_roll_joint': -0.2,
            'right_hip_pitch_joint': -0.1, 'right_hip_roll_joint': -0.2, 'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3, 'right_ankle_pitch_joint': -0.2, 'right_ankle_roll_joint': 0.2,
            'waist_yaw_joint': 0.0, 'waist_roll_joint': 0.0, 'waist_pitch_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0, 'left_shoulder_roll_joint': 0.5, 'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2, 'left_wrist_roll_joint': 0.0, 'left_wrist_pitch_joint': 0.0, 'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0, 'right_shoulder_roll_joint': -0.5, 'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2, 'right_wrist_roll_joint': 0.0, 'right_wrist_pitch_joint': 0.0, 'right_wrist_yaw_joint': 0.0,
        }
        init_pos = [-0.34930936, -0.03763366, -0.22198406,  0.93093884, -0.50943524, -0.08583859,
            0.13749947, -0.44516975, -0.06791031,  0.11570476, -0.17351833,  0.34241587,
            -0.00869134,  0.00670955,  0.01293622,  0.00395479,  0.49003497, -0.00168978,
            1.2062242,  -0.01060604,  0.00490874, -0.00869134,  0.00319979, -0.4975251,
            -0.00450607,  1.20307243,  0.00536893,  0.0053766,   0.00324437]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_yaw': 150, 'hip_roll': 150, 'hip_pitch': 150, 'knee': 300,
                     'ankle': 40, 'shoulder': 150, 'elbow': 150, 'waist': 150, 'wrist': 20}
        damping = {'hip_yaw': 2, 'hip_roll': 2, 'hip_pitch': 2, 'knee': 4, 'ankle': 2,
                   'shoulder': 2, 'elbow': 2, 'waist': 2, 'wrist': 0.5}
        action_scale = 0.25
        decimation = 4
        curriculum_joints = ['waist_yaw_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                             'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
                           'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
                            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                           'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        elbow_joints = ['left_elbow_joint', 'right_elbow_joint']
        wrist_joints = ['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        upper_body_link = "pelvis"
        torso_link = "torso_link"
        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']

    class termination:
        knee_height_threshold = -999.0
        gravity_threshold = 999.0
        enable_dive_window = False

    class terrain:
        static_friction = 1.0; dynamic_friction = 1.0; restitution = 0.

    class normalization:
        class obs_scales:
            lin_vel = 2.0; ang_vel = 0.25; dof_pos = 1.0; dof_vel = 0.05
            ball_vel = 0.2; ball_pos = 0.3; height_measurements = 5.0
            target = 1.0
        clip_observations = 100.; clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = True; noise_level = 1.0
        class noise_scales:
            ball = 0.08; dof_pos = 0.01; dof_vel = 1.5
            lin_vel = 0.1; ang_vel = 0.2; gravity = 0.05; height_measurements = 0.1

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29.urdf'
        ballfile = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/urdf/ball.urdf'
        name = "g1"
        foot_name = "ankle_pitch"
        contact_foot_names = "ankle_roll_link"
        hand_name = "rubber_hand"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        ankle_joints = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        keyframe_name = "keyframe"
        disable_gravity = False; collapse_fixed_joints = False; fix_base_link = False
        default_dof_drive_mode = 3; self_collisions = 0
        replace_cylinder_with_capsule = True; flip_visual_attachments = False
        density = 0.001; angular_damping = 0.01; linear_damping = 0.01
        max_angular_velocity = 1000.; max_linear_velocity = 1000.
        armature = 0.01; thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_joint_injection = False; joint_injection_range = [-0.01, 0.01]
        randomize_actuation_offset = False; actuation_offset_range = [-0.01, 0.01]
        randomize_payload_mass = False; payload_mass_range = [-5, 10]
        randomize_com_displacement = False; com_displacement_range = [-0.1, 0.1]
        randomize_link_mass = False; link_mass_range = [0.8, 1.2]
        randomize_friction = True; friction_range = [0.8, 1.2]
        randomize_restitution = False; restitution_range = [0.0, 1.0]
        randomize_kp = False; kp_range = [0.8, 1.2]
        randomize_kd = False; kd_range = [0.8, 1.2]
        randomize_initial_joint_pos = False
        initial_joint_pos_scale = [0.5, 1.5]; initial_joint_pos_offset = [-0.1, 0.1]
        continue_keep = True
        push_robots = False; push_interval_s = 15; max_push_vel_xy = 1.5
        ball_interval_s = 999.0; max_ball_vel = 0.0
        delay = False

    class rewards:
        class scales:
            # ── Old goalkeeper rewards: ALL ZERO ──
            eereach = 0.0; success = 0.0; stopball = 0.0
            stayonline = 0.0; noretreat = 0.0
            successland = 0.0; feetorientaion = 0.0
            penalize_sharpcontact = 0.0; penalize_kneeheight = 0.0; feet_slippage = 0.0
            postorientation = 0.0; postangvel = 0.0
            postupperdofpos = 0.0; postwaistdofpos = 0.0; postlinvel = 0.0

            # ── Regularization ──
            smoothness = -0.01
            torques = -1e-5
            dof_acc = -1e-7
            dof_pos_limits = -0.5
            dof_vel_limits = -0.2
            torque_limits = -0.2

            # ── Dive save rewards ──
            target_hand_reach = 9.0        # dense reach, lower weight, ETA window only
            launch_gate_bonus = 1.0        # small lateral launch gate
            hand_ball_distance_eta = 5.0   # hand-to-ball dist in ETA window
            target_hand_contact = 20.0     # target hand contacts ball (one-time)
            ball_deflection = 10.0         # ball deflected away from goal after contact
            success_dive_save = 50.0       # sparse: ball did not enter goal (one-time)
            # fast_reach_bonus disabled (not applicable for save task)

        only_positive_rewards = False
        catch_th = 0.5; handheight_th = 1.0; reach_th = 0.2; strict_th = 0.15
        target_dof_pos_sigma = -20; tracking_sigma = 0.25; catch_sigma = 5.0
        soft_dof_pos_limit = 0.9; soft_dof_vel_limit = 0.9; soft_torque_limit = 0.95
        max_contact_force = 1000.

        # ── Dive save parameters ──
        dive_deadline_s = 0.8             # compat with parent buffers (not used as active deadline)
        goal_x = -0.6                     # goal plane x (ball passes through target AT goal line)
        target_hand_reach_sigma = 0.5     # exp sigma for reach reward
        hand_ball_sigma = 0.3             # exp sigma for hand-ball distance
        eta_window_half = 0.15            # [s] ETA window half-width around arrival
        success_hand_dist_threshold = 0.15
        success_root_lateral_disp_min = 0.4
        success_root_lateral_vel_min = 0.5  # logging only
        hand_x_window = 0.5
        hand_z_min = 0.0; hand_z_max = 0.8
        # Goal area (world-size): ball inside these bounds = conceded
        goal_half_width = 1.5           # ±1.5m covers target at ±1.45
        goal_height = 1.8               # covers target_z up to 1.65
        goal_x_crossing = -0.6          # ball x < this = crossed goal plane

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30; min_time = 0.1

    class amp:
        obs_type = 'dof'; num_obs = 29 * 2; amp_coef = 0.0; num_steps = 2


class G1DiveSaveCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        experiment_name = 'g1_dive_save'
        run_name = ''
        wandb_project = 'legged_gym'
        max_iterations = 500
        save_interval = 100
        empirical_normalization = False
        resume = False; load_run = -1; checkpoint = -1
    amp = G1DiveSaveCfg.amp


# ═══════════════════════════════════════════════════════════════
# Environment Class
# ═══════════════════════════════════════════════════════════════

class DiveSaveRobot(DiveReachRobot):
    """G1 dive save environment.

    Inherits from DiveReachRobot. Key differences:
      - assign_ball_states() shoots ball through target (oracle).
      - time_remaining uses catchstep ETA, not fixed 0.8s.
      - Rewards: contact + deflection + save, not just reach.
    """

    @property
    def _dr_cfg(self):
        return self.cfg.rewards

    # ═══════════════════════════════════════════════════════
    # Target zone sampling (override: narrower range + would_score)
    # ═══════════════════════════════════════════════════════

    def _sample_target_zone(self, env_ids):
        """Sample high-corner target inside goal frame + set would_score flag."""
        n = len(env_ids)
        if n == 0:
            return

        goal_x = float(self._dr_cfg.goal_x)
        regions = self.end_regions[env_ids].long()
        is_pos_y = ((regions == 0) | (regions == 2) | (regions == 4))
        is_neg_y = ((regions == 1) | (regions == 3) | (regions == 5))

        target_y_local = torch.zeros(n, dtype=torch.float, device=self.device)
        if is_pos_y.any():
            n_pos = is_pos_y.sum().item()
            target_y_local[is_pos_y] = torch.rand(n_pos, device=self.device) * 0.2 + 1.25  # [1.25, 1.45]
        if is_neg_y.any():
            n_neg = is_neg_y.sum().item()
            target_y_local[is_neg_y] = -(torch.rand(n_neg, device=self.device) * 0.2 + 1.25)  # [-1.45, -1.25]

        target_z = torch.rand(n, device=self.device) * 0.35 + 0.25  # [0.25, 0.60]

        self.local_target_y[env_ids] = target_y_local
        self.target_side[env_ids] = torch.sign(target_y_local)
        self.target_pos[env_ids, 0] = goal_x + self.env_origins[env_ids, 0]
        self.target_pos[env_ids, 1] = target_y_local + self.env_origins[env_ids, 1]
        self.target_pos[env_ids, 2] = target_z

        # would_score: target is within goal bounds (oracle — ball WOULD score if unblocked)
        hw = float(self._dr_cfg.goal_half_width)
        gh = float(self._dr_cfg.goal_height)
        self.would_score[env_ids] = (target_y_local.abs() <= hw) & (target_z >= 0) & (target_z <= gh)

    # ═══════════════════════════════════════════════════════
    # Init — add save-specific buffers
    # ═══════════════════════════════════════════════════════

    def _init_buffers(self):
        num_envs = self.num_envs
        device = self.device
        # Create BEFORE super()._init_buffers() because parent calls _sample_target_zone
        self.would_score = torch.zeros(num_envs, dtype=torch.bool, device=device)
        super()._init_buffers()

        # Ball contact tracking (distance-based, not force-based)
        self.ball_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hand_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._prev_hand_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ball_deflected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ball_conceded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ball_crossed_goal_plane = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initial_ball_vel_x = torch.zeros(num_envs, dtype=torch.float, device=device)

        # ETA tracking
        self.initial_catchstep = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Hand-ball distance (ETA-windowed AND episode-wide)
        self.hand_ball_dist = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_min_hand_ball_dist_eta = torch.ones(num_envs, dtype=torch.float, device=device) * 999.0
        self.ep_min_hand_ball_dist_ep = torch.ones(num_envs, dtype=torch.float, device=device) * 999.0
        self.ep_min_target_hand_dist_eta = torch.ones(num_envs, dtype=torch.float, device=device) * 999.0

        # Per-side success tracking
        self.ep_left_success = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_right_success = torch.zeros(num_envs, dtype=torch.float, device=device)

        # ETA window mask for current step
        self._in_eta_window = torch.zeros(num_envs, dtype=torch.bool, device=device)

        print("[DIVE_SAVE] Oracle ball shooting to target zone")
        print(f"[DIVE_SAVE] ETA window: ±{self._dr_cfg.eta_window_half}s around arrival")
        print(f"[DIVE_SAVE] Goal: half_width={self._dr_cfg.goal_half_width}, height={self._dr_cfg.goal_height}")
        print(f"[DIVE_SAVE] Target: y∈±[1.25,1.45], z∈[1.35,1.65] (inside goal)")

    # ═══════════════════════════════════════════════════════
    # Oracle ball shooting — ball goes through target
    # ═══════════════════════════════════════════════════════

    def assign_ball_states(self, ball_ids, g=9.81):
        """Oracle: ball start → target at catch plane, with random t_flight.

        target = self.target_pos (already sampled by _sample_target_zone before this call).
        Ball velocity computed so ball passes through target at t_flight.
        """
        n = len(ball_ids)
        if n == 0:
            return torch.zeros(0, 3, device=self.device)

        dtype = torch.float
        device = self.device

        # ── Ball start: random in front, narrow y, low z ──
        start_x = torch.rand(n, dtype=dtype, device=device) * 2.0 + 3.0       # [3, 5]
        start_y = torch.rand(n, dtype=dtype, device=device) * 0.8 - 0.4       # [-0.4, 0.4]
        start_z = torch.rand(n, dtype=dtype, device=device) * 0.2 + 0.25      # [0.25, 0.45]

        ball_start_local = torch.stack([start_x, start_y, start_z], dim=1)
        self.ball_start[ball_ids, :] = ball_start_local + self.env_origins[ball_ids]
        self.ball_start[ball_ids, 2] = ball_start_local[:, 2]

        # ── Ball end = target_pos (world frame) ──
        # target_pos was already set by _sample_target_zone with world coords
        self.ball_end[ball_ids, :] = self.target_pos[ball_ids].clone()
        self.ball_end[ball_ids, 2] = self.target_pos[ball_ids, 2]

        # ── Flight time ──
        t_flight = torch.rand(n, dtype=dtype, device=device) * 0.3 + 0.55     # [0.55, 0.85]

        # ── Compute velocity to pass through target at t_flight ──
        delta = self.ball_end[ball_ids] - self.ball_start[ball_ids]
        ball_vel = torch.zeros(n, 3, dtype=dtype, device=device)
        ball_vel[:, 0:2] = delta[:, 0:2] / t_flight.unsqueeze(-1)
        ball_vel[:, 2] = (delta[:, 2] + 0.5 * g * t_flight ** 2) / t_flight

        # ── Set catchstep from t_flight ──
        catchstep_val = torch.round(t_flight / self.dt).int()
        self.catchstep[ball_ids] = catchstep_val
        self.initial_catchstep[ball_ids] = catchstep_val.float()
        self.initial_ball_vel_x[ball_ids] = ball_vel[:, 0]

        # ── end_target = ball_end (interception point, for compatibility) ──
        self.end_target[ball_ids, :] = self.ball_end[ball_ids].clone()

        return ball_vel

    # ═══════════════════════════════════════════════════════
    # Reset root states — use oracle assign_ball_states
    # ═══════════════════════════════════════════════════════

    def _reset_root_states(self, env_ids):
        """Override: oracle ball launch through target."""
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if getattr(self.cfg.domain_rand, 'randomize_reset_velocity', True):
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=self.device)
        else:
            self.root_states[env_ids, 7:13] = 0.0

        # Oracle ball launch
        ball_vel = self.assign_ball_states(env_ids)
        self.ball_states[env_ids] = self.base_init_state
        self.ball_states[env_ids, :3] = self.ball_start[env_ids, :]
        self.ball_states[env_ids, 7:10] = ball_vel
        self.ball_states[env_ids, 10:13] = 0.0
        self.ball_vel[env_ids] = ball_vel[:, 0]

        # Clear per-episode flags
        self.has_in_air[env_ids] = False
        self.stop_flag[env_ids] = 0.0
        self.success_flag[env_ids] = 0.0
        self.t_success[env_ids] = self._dr_cfg.dive_deadline_s
        self._first_success_this_step[env_ids] = False
        self.ball_contacted[env_ids] = False
        self.hand_contacted[env_ids] = False
        self._prev_hand_contacted[env_ids] = False
        self.ball_deflected[env_ids] = False
        self.ball_conceded[env_ids] = False
        self.ball_crossed_goal_plane[env_ids] = False
        self.dist[env_ids] = 5.0
        self.ball_last[env_ids] = 0.0
        self.vanish_step[env_ids] = 0

        # Write to sim
        all_states = torch.cat((self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(all_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # ═══════════════════════════════════════════════════════
    # Metrics — add ball tracking
    # ═══════════════════════════════════════════════════════

    def _compute_dive_reach_metrics(self):
        """Extend parent: hand-ball distance (ETA-gated), contact (distance-based), conceded."""
        super()._compute_dive_reach_metrics()

        # Target-side hand position
        left_pos = self.rigid_body_states[:, self._left_hand_idx, 0:3]
        right_pos = self.rigid_body_states[:, self._right_hand_idx, 0:3]
        use_left = self.target_side > 0
        use_right = self.target_side < 0
        target_hand_pos = torch.zeros_like(left_pos)
        target_hand_pos[use_left] = left_pos[use_left]
        target_hand_pos[use_right] = right_pos[use_right]
        neutral = ~(use_left | use_right)
        if neutral.any():
            target_hand_pos[neutral] = left_pos[neutral]

        ball_pos = self.ball_states[:, :3]
        self.hand_ball_dist = torch.norm(target_hand_pos - ball_pos, dim=-1)

        # ETA window
        eta_half_steps = int(self._dr_cfg.eta_window_half / self.dt)
        self._in_eta_window = (self.catchstep.abs() <= eta_half_steps)

        # ETA-gated distance tracking
        in_eta = self._in_eta_window
        self.ep_min_hand_ball_dist_eta = torch.where(
            in_eta, torch.minimum(self.ep_min_hand_ball_dist_eta, self.hand_ball_dist), self.ep_min_hand_ball_dist_eta)
        self.ep_min_hand_ball_dist_ep = torch.minimum(self.ep_min_hand_ball_dist_ep, self.hand_ball_dist)
        self.ep_min_target_hand_dist_eta = torch.where(
            in_eta, torch.minimum(self.ep_min_target_hand_dist_eta, self.hand_dist), self.ep_min_target_hand_dist_eta)

        # Contact detection: DISTANCE-based (hand-ball < ball_radius + hand_radius ≈ 0.15m)
        hand_contact_now = self.hand_ball_dist < 0.20
        # Detect NEW contacts (transition from False to True)
        new_contact = hand_contact_now & ~self._prev_hand_contacted
        self.hand_contacted[new_contact] = True
        self.ball_contacted[new_contact] = True
        self._prev_hand_contacted = hand_contact_now

        # Deflection: ball x-velocity reversed after contact
        ball_vx = self.ball_states[:, 7]
        deflected = self.ball_contacted & (ball_vx > 0.0)
        self.ball_deflected[deflected] = True

        # Conceded: ball crossed goal plane INSIDE goal bounds
        ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
        ball_local_y = self.ball_states[:, 1] - self.env_origins[:, 1]
        hw = float(self._dr_cfg.goal_half_width)
        gh = float(self._dr_cfg.goal_height)
        gx = float(self._dr_cfg.goal_x_crossing)
        crossed_goal_plane = ball_local_x < gx
        self.ball_crossed_goal_plane[crossed_goal_plane] = True
        inside_goal_bounds = (ball_local_y.abs() <= hw) & (self.ball_states[:, 2] <= gh)
        self.ball_conceded[crossed_goal_plane & inside_goal_bounds] = True

    # ═══════════════════════════════════════════════════════
    # Observations — ETA-based time_remaining
    # ═══════════════════════════════════════════════════════

    def compute_observations(self):
        """Override: time_remaining from catchstep ETA, not fixed deadline."""
        # ETA-based time remaining (normalized by max flight time)
        max_catch = self.initial_catchstep.clamp(min=1.0)
        time_remaining = torch.clamp(self.catchstep.float() / max_catch, min=0.0, max=1.0)

        ball_local = quat_rotate_inverse(self.base_quat, self.ball_states[:, :3] - self.torso_pos)
        local_target_y = self.local_target_y.unsqueeze(-1)
        target_z = self.target_pos[:, 2:3]

        current_actor_obs = torch.cat((
            ball_local,
            local_target_y,
            target_z,
            self.target_side.unsqueeze(-1),
            time_remaining.unsqueeze(-1),
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.obs_buf = torch.cat(
            (self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)

        target_local = quat_rotate_inverse(self.base_quat, self.target_pos - self.torso_pos)
        left_hand_pos = self.rigid_body_states[:, self._left_hand_idx, 0:3]
        right_hand_pos = self.rigid_body_states[:, self._right_hand_idx, 0:3]
        hand_pos_r_local = quat_rotate_inverse(self.base_quat, right_hand_pos - self.torso_pos)
        hand_pos_l_local = quat_rotate_inverse(self.base_quat, left_hand_pos - self.torso_pos)

        self.privileged_obs_buf = torch.cat((
            current_actor_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.end_regions.unsqueeze(-1).float() / 3.0,
            target_local,
            self.ball_states[:, 7:10] * self.obs_scales.ball_vel,
            hand_pos_r_local,
            hand_pos_l_local,
            self.hand_dist.unsqueeze(-1),
        ), dim=-1)

        if not torch.isfinite(self.obs_buf).all():
            raise RuntimeError("Non-finite obs_buf in g1_dive_save")
        if not torch.isfinite(self.privileged_obs_buf).all():
            raise RuntimeError("Non-finite privileged_obs_buf in g1_dive_save")

    # ═══════════════════════════════════════════════════════
    # Rewards
    # ═══════════════════════════════════════════════════════

    def _reward_target_hand_reach(self):
        """Dense reach reward, ETA window only, lower weight."""
        sigma = float(self._dr_cfg.target_hand_reach_sigma)
        rew = torch.exp(-self.hand_dist / sigma) * self._in_eta_window.float()
        return rew

    def _reward_launch_gate_bonus(self):
        """Small lateral launch gate, ETA window only."""
        close = self.hand_dist < 0.3
        launched = self.root_lateral_disp_toward_target > 0.25
        return (close & launched).float() * self._in_eta_window.float()

    def _reward_hand_ball_distance_eta(self):
        """Hand-to-ball distance in ETA window."""
        sigma = float(self._dr_cfg.hand_ball_sigma)
        rew = torch.exp(-self.hand_ball_dist / sigma) * self._in_eta_window.float()
        return rew

    def _reward_target_hand_contact(self):
        """One-time reward: target hand first contacts ball (distance-based, ETA window)."""
        # hand_contacted just flipped True this step → contact_now=True, _prev_hand_contacted (pre-update)=False
        # We detect this via: hand_contacted=True AND the contact happened in ETA window
        # Since _prev_hand_contacted is updated AFTER, a new contact = hand_contacted & ~_prev_hand_contacted (old)
        # Simpler: reward when hand_contacted is True AND we're in ETA window AND not yet succeeded
        just_contacted = self.hand_contacted & self._in_eta_window & (self.success_flag == 0)
        return just_contacted.float()

    def _reward_ball_deflection(self):
        """Ball deflected away from goal after contact. Per-step, ETA window."""
        ball_vx = self.ball_states[:, 7]
        deflected = (ball_vx > 0.0).float() * self.ball_contacted.float()
        return deflected * self._in_eta_window.float()

    def _reward_success_dive_save(self):
        """Sparse one-time save reward. Requires:
          - would_score (ball target was inside goal)
          - robot contacted ball (hand or body)
          - ball did NOT enter goal (not conceded)
        Fires on first frame where ball has crossed goal plane after contact.
        """
        # Success = would_score + contacted + not conceded + ball past/crossed plane
        ball_local_x = self.ball_states[:, 0] - self.env_origins[:, 0]
        ball_past = ball_local_x < float(self._dr_cfg.goal_x_crossing)
        is_timeout = self.episode_length_buf >= self.max_episode_length

        success_now = self.would_score & self.ball_contacted & ~self.ball_conceded & (ball_past | is_timeout) & (self.success_flag == 0)
        self.success_flag[success_now] = 1.0
        self.t_success[success_now] = self.episode_length_buf[success_now].float() * self.dt
        self._first_success_this_step = success_now

        self.ep_left_success[success_now & (self.target_side < 0)] = 1.0
        self.ep_right_success[success_now & (self.target_side > 0)] = 1.0

        return success_now.float()

    # ═══════════════════════════════════════════════════════
    # Reset — add save-specific metrics
    # ═══════════════════════════════════════════════════════

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        n_succ = self.success_flag[env_ids]
        captured = {
            "local_target_y": self.local_target_y[env_ids].mean().item(),
            "target_z": self.target_pos[env_ids, 2].mean().item(),
            "would_score_rate": self.would_score[env_ids].float().mean().item(),
            "success_rate": n_succ.float().mean().item(),
            "time_to_success": self.t_success[env_ids][n_succ.bool()].mean().item() if n_succ.any() else -1.0,
            "ball_contact_rate": self.ball_contacted[env_ids].float().mean().item(),
            "hand_contact_rate_eta": self.hand_contacted[env_ids].float().mean().item(),
            "deflection_rate": self.ball_deflected[env_ids].float().mean().item(),
            "conceded_rate": self.ball_conceded[env_ids].float().mean().item(),
            "crossed_goal_plane_rate": self.ball_crossed_goal_plane[env_ids].float().mean().item(),
            "min_hand_ball_dist_eta": self.ep_min_hand_ball_dist_eta[env_ids].mean().item(),
            "min_hand_ball_dist_ep": self.ep_min_hand_ball_dist_ep[env_ids].mean().item(),
            "min_target_hand_dist_eta": self.ep_min_target_hand_dist_eta[env_ids].mean().item(),
            "min_target_hand_dist_ep": self.ep_min_hand_dist[env_ids].mean().item(),
            "root_lateral_disp": self.ep_root_lateral_disp[env_ids].mean().item(),
            "peak_lateral_vel": self.ep_peak_root_lateral_vel[env_ids].mean().item(),
            "ep_length": self.episode_length_buf[env_ids].float().mean().item(),
            "timeout": self.time_out_buf[env_ids].float().mean().item(),
            "left_success": self.ep_left_success[env_ids].mean().item(),
            "right_success": self.ep_right_success[env_ids].mean().item(),
        }

        self._sample_target_zone(env_ids)
        self.init_root_y[env_ids] = self.root_states[env_ids, 1].clone()
        super(DiveReachRobot, self).reset_idx(env_ids)

        eps = self.extras.setdefault("episode", {})
        eps["local_target_y_mean"] = captured["local_target_y"]
        eps["target_z_mean"] = captured["target_z"]
        eps["would_score_rate"] = captured["would_score_rate"]
        eps["success_dive_save_rate"] = captured["success_rate"]
        eps["time_to_success"] = captured["time_to_success"]
        eps["ball_contact_rate"] = captured["ball_contact_rate"]
        eps["hand_contact_rate_eta"] = captured["hand_contact_rate_eta"]
        eps["deflection_rate"] = captured["deflection_rate"]
        eps["conceded_rate"] = captured["conceded_rate"]
        eps["crossed_goal_plane_rate"] = captured["crossed_goal_plane_rate"]
        eps["min_hand_ball_dist_eta"] = captured["min_hand_ball_dist_eta"]
        eps["min_hand_ball_dist_ep"] = captured["min_hand_ball_dist_ep"]
        eps["min_target_hand_dist_eta"] = captured["min_target_hand_dist_eta"]
        eps["min_target_hand_dist_ep"] = captured["min_target_hand_dist_ep"]
        eps["root_lateral_displacement_toward_target"] = captured["root_lateral_disp"]
        eps["peak_root_lateral_velocity_toward_target"] = captured["peak_lateral_vel"]
        eps["episode_length"] = captured["ep_length"]
        eps["timeout_rate"] = captured["timeout"]
        eps["left_success_rate"] = captured["left_success"]
        eps["right_success_rate"] = captured["right_success"]

        # Clear episode buffers
        self.ep_min_hand_dist[env_ids] = 999.0
        self.ep_min_hand_ball_dist_eta[env_ids] = 999.0
        self.ep_min_hand_ball_dist_ep[env_ids] = 999.0
        self.ep_min_target_hand_dist_eta[env_ids] = 999.0
        self.ep_success[env_ids] = 0.0
        self.ep_time_to_success[env_ids] = 0.0
        self.t_success[env_ids] = self._dr_cfg.dive_deadline_s
        self.ep_peak_root_lateral_vel[env_ids] = 0.0
        self.ep_root_lateral_disp[env_ids] = 0.0
        self.ep_workspace_reset[env_ids] = 0.0
        self.ep_nan_reset[env_ids] = 0.0
        self.ep_timeout[env_ids] = 0.0
        self.ep_left_success[env_ids] = 0.0
        self.ep_right_success[env_ids] = 0.0
        self._first_success_this_step[env_ids] = False
