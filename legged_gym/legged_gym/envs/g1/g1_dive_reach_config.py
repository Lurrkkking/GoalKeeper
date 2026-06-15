"""G1 Dive Reach Primitive Config — standalone lateral dive to high-corner target zone.

NOT a goalkeeper task. No ball blocking, no landing, no recovery.
Pure primitive: from goalkeeper ready stance, lateral-launch dive to reach
a high-corner target zone with the correct-side rubber hand within deadline.

Target zone: goal plane high corners (y=±0.9~1.2, z=1.4~1.6).
Success requires lateral root displacement + velocity, not just arm reach.

THREE CONCEPTS ARE SEPARATE:
  1. ball     — real physics ball, normal goalkeeper launch, NOT a target marker
  2. target_pos — virtual goal-plane point, no physics, used for all rewards
  3. target marker — visualization only (not yet implemented; ball != marker)
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float
from isaacgym import gymtorch

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class G1DiveReachCfg(LeggedRobotCfg):
    """G1 dive reach primitive — lateral dive to high-corner target zone."""

    class env(LeggedRobotCfg.env):
        num_envs = 6144
        num_actor_history = 10
        num_actions = 29
        num_dofs = 29
        num_ballobs = 3           # real ball local pos in actor obs
        num_target_obs = 4        # local_target_y, target_z, target_side, time_remaining

        num_one_step_observations = 6 + num_ballobs + num_target_obs + num_dofs * 2 + num_actions  # 100
        num_privileged_obs = num_one_step_observations + 3 + 1 + 3 + 3 + 3 + 3 + 1  # 117
        num_observations = num_actor_history * num_one_step_observations  # 1000

        env_spacing = 5.
        send_timeouts = True
        episode_length_s = 3.0     # longer rollout to see post-dive state; success still gated by 0.8s deadline
        ball_gravity = True        # real ball, normal physics
        play = False
        mode_weights = [1, 1, 0, 0, 0, 0]  # only modes 0 (right) and 1 (left)
        # ball launches normally (default ~3-5m start_x); reward ignores ball

    # ── Target zone sampling: high corners only (v3: wider y) ──
    # Mode 0 = right high corner, Mode 1 = left high corner
    class commands:
        class ranges_0:
            # Right high corner
            height = [1.4, 1.6]; width = [1.4, 1.6]
            maxh = [1.4, 1.6]; maxw = [1.4, 1.6]
            evalh = [1.4, 1.6]; evalw = [1.4, 1.6]
        class ranges_1:
            # Left high corner
            height = [1.4, 1.6]; width = [-1.6, -1.4]
            maxh = [1.4, 1.6]; maxw = [-1.6, -1.4]
            evalh = [1.4, 1.6]; evalw = [-1.6, -1.4]
        class ranges_2:
            height = [1.4, 1.6]; width = [1.4, 1.6]
            maxh = [1.4, 1.6]; maxw = [1.4, 1.6]
            evalh = [1.4, 1.6]; evalw = [1.4, 1.6]
        class ranges_3:
            height = [1.4, 1.6]; width = [-1.6, -1.4]
            maxh = [1.4, 1.6]; maxw = [-1.6, -1.4]
            evalh = [1.4, 1.6]; evalw = [-1.6, -1.4]
        class ranges_4:
            height = [1.4, 1.6]; width = [1.4, 1.6]
            maxh = [1.4, 1.6]; maxw = [1.4, 1.6]
            evalh = [1.4, 1.6]; evalw = [1.4, 1.6]
        class ranges_5:
            height = [1.4, 1.6]; width = [-1.6, -1.4]
            maxh = [1.4, 1.6]; maxw = [-1.6, -1.4]
            evalh = [1.4, 1.6]; evalw = [-1.6, -1.4]

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

    # ── Termination: extremely lenient ──
    class termination:
        knee_height_threshold = -999.0   # effectively disabled
        gravity_threshold = 999.0        # effectively disabled
        enable_dive_window = False       # no dive window needed

    class terrain:
        static_friction = 1.0; dynamic_friction = 1.0; restitution = 0.

    class normalization:
        class obs_scales:
            lin_vel = 2.0; ang_vel = 0.25; dof_pos = 1.0; dof_vel = 0.05
            ball_vel = 0.2; ball_pos = 0.3; height_measurements = 5.0
            target = 1.0  # target y/z scale
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
        hand_name = "rubber_hand"          # matches left_rubber_hand / right_rubber_hand
        penalize_contacts_on = []          # no contact penalization
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

    # ── DR: mostly OFF for skill discovery ──
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

    # ── Dive Reach Rewards ──
    class rewards:
        class scales:
            # ── Old goalkeeper rewards: ALL ZEROED ──
            eereach = 0.0; success = 0.0; stopball = 0.0
            stayonline = 0.0; noretreat = 0.0
            successland = 0.0; feetorientaion = 0.0
            penalize_sharpcontact = 0.0; penalize_kneeheight = 0.0; feet_slippage = 0.0
            postorientation = 0.0; postangvel = 0.0
            postupperdofpos = 0.0; postwaistdofpos = 0.0; postlinvel = 0.0

            # ── Small regularization (keep sim stable) ──
            smoothness = -0.01
            torques = -1e-5
            dof_acc = -1e-7
            dof_pos_limits = -0.5
            dof_vel_limits = -0.2
            torque_limits = -0.2

            # ── Dive Reach rewards ──
            target_hand_reach = 12.0       # dense: exp(-dist/sigma), sigma=0.5
            success_dive_reach = 50.0     # sparse: one-time success
            fast_reach_bonus = 5.0        # bonus for early success
            launch_gate_bonus = 5.0       # small bonus when close + lateral launch

        only_positive_rewards = False
        catch_th = 0.5; handheight_th = 1.0; reach_th = 0.2; strict_th = 0.15
        target_dof_pos_sigma = -20; tracking_sigma = 0.25; catch_sigma = 5.0
        soft_dof_pos_limit = 0.9; soft_dof_vel_limit = 0.9; soft_torque_limit = 0.95
        max_contact_force = 1000.

        # ── Dive reach parameters ──
        dive_deadline_s = 0.8              # must reach within 0.8s
        target_hand_reach_sigma = 0.5      # exp sigma for dense reach reward
        success_hand_dist_threshold = 0.15  # [m] hand must be within this distance
        success_root_lateral_disp_min = 0.4   # [m] min root lateral displacement (v3: tighter gate for real dive)
        success_root_lateral_vel_min = 0.5    # [m/s] logging-only; NOT a success condition
        hand_x_window = 0.5                # [m] abs(hand_x - target_x) must be < this
        hand_z_min = 1.2; hand_z_max = 1.8

        # Target zone x coordinate (catch/interception plane)
        # Original G1 goalkeeper: end_target_x = start_x + (end_x-start_x) * (0.1-start_x)/(end_x-start_x) = 0.1
        # This is the plane where eereach reward computes hand-to-target distance.
        # ball_end_x ∈ [-0.6, -0.1] is the goal line (behind robot), NOT the reach target.
        goal_x = 0.1

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30; min_time = 0.1

    class amp:
        obs_type = 'dof'; num_obs = 29 * 2; amp_coef = 0.0; num_steps = 2  # AMP disabled for primitive discovery


class G1DiveReachCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        experiment_name = 'g1_dive_reach'
        run_name = ''
        wandb_project = 'legged_gym'
        max_iterations = 500
        save_interval = 100
        empirical_normalization = False
        resume = False; load_run = -1; checkpoint = -1
    amp = G1DiveReachCfg.amp


# ═══════════════════════════════════════════════════════════════
# Environment Class
# ═══════════════════════════════════════════════════════════════

class DiveReachRobot(LeggedRobot):
    """G1 dive reach primitive environment.

    Ball: normal goalkeeper launch (real physics), NOT a target marker.
    target_pos: virtual goal-plane point, no physics, used for all rewards.
    """

    # ── Dive reach config accessors ──
    @property
    def _dr_cfg(self):
        return self.cfg.rewards

    # ═══════════════════════════════════════════════════════
    # Init
    # ═══════════════════════════════════════════════════════

    def _init_buffers(self):
        """Add dive-reach tracking buffers, then defer to parent."""
        super()._init_buffers()
        num_envs = self.num_envs
        device = self.device

        # Target zone — local (robot-frame) and world
        self.local_target_y = torch.zeros(num_envs, dtype=torch.float, device=device)  # [-1.2,-0.9] or [0.9,1.2]
        self.target_pos = torch.zeros(num_envs, 3, dtype=torch.float, device=device)
        self.target_side = torch.zeros(num_envs, dtype=torch.float, device=device)  # sign(local_target_y)
        self.init_root_y = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Per-episode tracking
        self.ep_min_hand_dist = torch.ones(num_envs, dtype=torch.float, device=device) * 999.0
        self.ep_success = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_time_to_success = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.t_success = torch.full((num_envs,), self._dr_cfg.dive_deadline_s, dtype=torch.float, device=device)  # first-success time; init=deadline to avoid NaN
        self.ep_peak_root_lateral_vel = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_root_lateral_disp = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_target_hand_dist = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_workspace_reset = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_nan_reset = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.ep_timeout = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Per-step
        self.hand_dist = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.root_lateral_disp_toward_target = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.root_lateral_vel_toward_target = torch.zeros(num_envs, dtype=torch.float, device=device)

        # Rubber hand body indices (resolved after _create_envs)
        self._left_hand_idx = None
        self._right_hand_idx = None
        self._target_hand_resolved = False
        self._first_success_this_step = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Sample initial target zones (ball launches independently via assign_ball_states)
        all_env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
        self._sample_target_zone(all_env_ids)
        self.init_root_y[:] = self.root_states[:, 1].clone()

        print("=" * 70)
        print("[DIVE_REACH] Buffers initialized")
        print(f"  num_one_step_observations = {self.num_one_step_obs}")
        print(f"  num_actor_history          = {self.actor_history_length}")
        print(f"  num_observations           = {self.cfg.env.num_observations}")
        print(f"  num_actions                = {self.cfg.env.num_actions}")
        print(f"  dive_deadline_s            = {self._dr_cfg.dive_deadline_s}")
        print(f"  episode_length_s           = {self.cfg.env.episode_length_s}")
        print(f"  target_obs in actor obs    = True (local_target_y, target_z, target_side, time_remaining)")
        print(f"  goal_x (catch plane)       = {self._dr_cfg.goal_x}")
        print(f"  Hand mapping: local_target_y > 0 → left_rubber_hand, < 0 → right_rubber_hand")
        print("=" * 70)

    def _resolve_target_hands(self):
        """Find left_rubber_hand and right_rubber_hand body indices."""
        if self._target_hand_resolved:
            return

        all_names = list(self._all_body_names)
        print("=" * 70)
        print("[DIVE_REACH] Body name resolution:")
        print(f"  All body names ({len(all_names)}): {all_names}")

        requested = ["left_rubber_hand", "right_rubber_hand"]
        print(f"  Requested target hand bodies: {requested}")

        left_idx = None
        right_idx = None
        for i, name in enumerate(all_names):
            if name == "left_rubber_hand":
                left_idx = i
            if name == "right_rubber_hand":
                right_idx = i

        matched = []
        missing = []
        if left_idx is not None:
            matched.append(f"left_rubber_hand (idx={left_idx})")
            self._left_hand_idx = left_idx
        else:
            missing.append("left_rubber_hand")
        if right_idx is not None:
            matched.append(f"right_rubber_hand (idx={right_idx})")
            self._right_hand_idx = right_idx
        else:
            missing.append("right_rubber_hand")

        print(f"  Matched target hand bodies: {matched if matched else 'NONE'}")
        print(f"  Missing target hand bodies: {missing if missing else 'NONE'}")

        if missing:
            raise RuntimeError(
                f"[DIVE_REACH] FATAL: Cannot find target hand bodies: {missing}. "
                f"Available bodies: {all_names}. "
                f"Check URDF collapse_fixed_joints setting (must be False)."
            )

        self._target_hand_resolved = True
        print("=" * 70)

    # ═══════════════════════════════════════════════════════
    # Noise scale (extended for target obs)
    # ═══════════════════════════════════════════════════════

    def _get_noise_scale_vec(self, cfg):
        """Extend noise vector for target observation dims (no noise on target)."""
        num_target = self.cfg.env.num_target_obs
        base_len = self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions
        noise_vec = torch.zeros(base_len + num_target, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:self.num_ballobs] = noise_scales.ball * noise_level
        # target obs at [num_ballobs : num_ballobs+num_target] — no noise
        offset = self.num_ballobs + num_target
        noise_vec[offset:offset + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[offset + 3:offset + 6] = noise_scales.gravity * noise_level
        noise_vec[offset + 6:offset + 6 + self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[offset + 6 + self.num_dof:offset + 6 + 2 * self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[offset + 6 + 2 * self.num_dof:offset + 6 + 2 * self.num_dof + self.num_actions] = 0.
        return noise_vec

    # ═══════════════════════════════════════════════════════
    # Target zone sampling
    # ═══════════════════════════════════════════════════════

    def _sample_target_zone(self, env_ids):
        """Sample high-corner target zone for reset envs.

        Sets local_target_y, target_side, and target_pos (world frame).
        """
        n = len(env_ids)
        if n == 0:
            return

        goal_x = float(self._dr_cfg.goal_x)

        # target_y: right [1.4, 1.6], left [-1.6, -1.4]
        # Use end_regions to determine side: 0,2,4 = pos y; 1,3,5 = neg y
        regions = self.end_regions[env_ids].long()
        is_pos_y = ((regions == 0) | (regions == 2) | (regions == 4))
        is_neg_y = ((regions == 1) | (regions == 3) | (regions == 5))

        target_y_local = torch.zeros(n, dtype=torch.float, device=self.device)
        if is_pos_y.any():
            n_pos = is_pos_y.sum().item()
            target_y_local[is_pos_y] = torch.rand(n_pos, device=self.device) * 0.2 + 1.4  # [1.4, 1.6]
        if is_neg_y.any():
            n_neg = is_neg_y.sum().item()
            target_y_local[is_neg_y] = -(torch.rand(n_neg, device=self.device) * 0.2 + 1.4)  # [-1.6, -1.4]

        # target_z: [1.4, 1.6]
        target_z = torch.rand(n, device=self.device) * 0.2 + 1.4

        # Local target buffers (actor obs uses these, no env_origins)
        self.local_target_y[env_ids] = target_y_local
        self.target_side[env_ids] = torch.sign(target_y_local)

        # World-frame target for distance computation
        self.target_pos[env_ids, 0] = goal_x + self.env_origins[env_ids, 0]
        self.target_pos[env_ids, 1] = target_y_local + self.env_origins[env_ids, 1]
        self.target_pos[env_ids, 2] = target_z

    # ═══════════════════════════════════════════════════════
    # Post physics step override
    # ═══════════════════════════════════════════════════════

    def post_physics_step(self):
        """Override: remove ball approach logic, freeze ball at target, new rewards/obs."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.catchstep -= 1

        # Base state
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = self._euler_from_quat_tensors()

        upper_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        self.base_lin_vel = quat_rotate_inverse(upper_quat, self.rigid_body_states[:, self.upper_body_index, 7:10])
        self.base_ang_vel = quat_rotate_inverse(upper_quat, self.rigid_body_states[:, self.upper_body_index, 10:13])

        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        self.projected_gravity[:] = quat_rotate_inverse(upper_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        # Joint powers
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        self._post_physics_step_callback()

        # ── Compute per-step dive reach metrics ──
        self._compute_dive_reach_metrics()

        # ── end_target for logging compatibility (set to target_pos) ──
        self.end_target[:] = self.target_pos.clone()

        # ── dist for compatibility ──
        self.dist[:] = self.hand_dist

        # Compute rewards
        self.compute_reward()

        # Check termination
        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)

        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def _euler_from_quat_tensors(self):
        """Inline euler_from_quaternion avoiding external import."""
        qx, qy, qz, qw = self.base_quat[:, 0], self.base_quat[:, 1], self.base_quat[:, 2], self.base_quat[:, 3]
        roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return roll, pitch, yaw

    # ═══════════════════════════════════════════════════════
    # Dive reach metrics
    # ═══════════════════════════════════════════════════════

    def _compute_dive_reach_metrics(self):
        """Compute per-step hand distance, lateral metrics."""
        self._resolve_target_hands()

        # Get target hand position
        left_pos = self.rigid_body_states[:, self._left_hand_idx, 0:3]
        right_pos = self.rigid_body_states[:, self._right_hand_idx, 0:3]

        # Target hand: original GK convention: pos y → LEFT hand, neg y → RIGHT hand
        use_left_hand = self.target_side > 0   # local_target_y > 0 → left_rubber_hand
        use_right_hand = self.target_side < 0  # local_target_y < 0 → right_rubber_hand

        hand_pos = torch.zeros_like(left_pos)
        hand_pos[use_left_hand] = left_pos[use_left_hand]
        hand_pos[use_right_hand] = right_pos[use_right_hand]
        # For target_side == 0 (shouldn't happen), use nearest
        neutral = ~(use_left_hand | use_right_hand)
        if neutral.any():
            dist_l = torch.norm(left_pos[neutral] - self.target_pos[neutral], dim=-1)
            dist_r = torch.norm(right_pos[neutral] - self.target_pos[neutral], dim=-1)
            hand_pos[neutral] = torch.where(
                (dist_l < dist_r).unsqueeze(-1), left_pos[neutral], right_pos[neutral])

        self.hand_dist = torch.norm(hand_pos - self.target_pos, dim=-1)

        # Root lateral displacement toward target
        root_dy = self.root_states[:, 1] - self.init_root_y
        self.root_lateral_disp_toward_target = root_dy * self.target_side

        # Root lateral velocity toward target
        root_vy = self.root_states[:, 8]  # world-frame y velocity
        self.root_lateral_vel_toward_target = root_vy * self.target_side

        # Episode tracking
        self.ep_min_hand_dist = torch.minimum(self.ep_min_hand_dist, self.hand_dist)
        self.ep_peak_root_lateral_vel = torch.maximum(
            self.ep_peak_root_lateral_vel,
            torch.relu(self.root_lateral_vel_toward_target))
        self.ep_root_lateral_disp = torch.maximum(
            self.ep_root_lateral_disp,
            torch.relu(self.root_lateral_disp_toward_target))
        self.ep_target_hand_dist = self.hand_dist  # latest

    # ═══════════════════════════════════════════════════════
    # Termination (lenient)
    # ═══════════════════════════════════════════════════════

    def check_termination(self):
        """Lenient termination: only timeout, NaN, extreme out-of-bounds."""
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        # NaN / Inf
        invalid_state = ~torch.isfinite(self.root_states).all(dim=1) | ~torch.isfinite(self.ball_states).all(dim=1)

        # Workspace boundary: root too far
        root_far = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1]) > 2.5  # |y| > 2.5
        root_far |= torch.abs(self.root_states[:, 0] - self.env_origins[:, 0]) > 3.0  # |x| > 3.0

        # root_z < 0.05 AND numerical anomaly (NaN in dof_pos)
        root_below = (self.root_states[:, 2] < 0.05) & (~torch.isfinite(self.dof_pos).all(dim=1))

        # No knee height, no gravity, no contact termination
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= invalid_state
        self.reset_buf |= root_far
        self.reset_buf |= root_below

    # ═══════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════

    def reset_idx(self, env_ids):
        """Reset envs: sample target zone, track metrics, parent reset."""
        if len(env_ids) == 0:
            return

        # Capture metrics BEFORE parent reset clears episode state
        n_succ = self.success_flag[env_ids]
        captured = {
            "local_target_y": self.local_target_y[env_ids].mean().item(),
            "target_z": self.target_pos[env_ids, 2].mean().item(),
            "hand_dist": self.ep_target_hand_dist[env_ids].mean().item(),
            "min_hand_dist": self.ep_min_hand_dist[env_ids].mean().item(),
            "success_rate": n_succ.float().mean().item(),
            "time_to_success": self.t_success[env_ids][n_succ.bool()].mean().item() if n_succ.any() else -1.0,
            "root_lateral_disp": self.ep_root_lateral_disp[env_ids].mean().item(),
            "peak_lateral_vel": self.ep_peak_root_lateral_vel[env_ids].mean().item(),
            "ep_length": self.episode_length_buf[env_ids].float().mean().item(),
            "timeout": self.time_out_buf[env_ids].float().mean().item(),
            "workspace_reset": (
                (torch.abs(self.root_states[env_ids, 1] - self.env_origins[env_ids, 1]) > 2.5) |
                (torch.abs(self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]) > 3.0)
            ).float().mean().item(),
            "nan_reset": (
                ~torch.isfinite(self.root_states[env_ids]).all(dim=1) |
                ~torch.isfinite(self.ball_states[env_ids]).all(dim=1)
            ).float().mean().item(),
        }

        # Sample target zone BEFORE parent resets ball
        self._sample_target_zone(env_ids)

        # Track init root y
        self.init_root_y[env_ids] = self.root_states[env_ids, 1].clone()

        # Parent reset (handles DOF, root state, ball state, extras init, episode sum reset)
        super().reset_idx(env_ids)

        # Now add dive-reach metrics to extras (parent just initialized extras["episode"] = {})
        eps = self.extras.setdefault("episode", {})
        eps["local_target_y_mean"] = captured["local_target_y"]
        eps["target_z_mean"] = captured["target_z"]
        eps["target_hand_dist"] = captured["hand_dist"]
        eps["min_target_hand_dist_episode"] = captured["min_hand_dist"]
        eps["success_dive_reach_rate"] = captured["success_rate"]
        eps["time_to_success"] = captured["time_to_success"]
        eps["root_lateral_displacement_toward_target"] = captured["root_lateral_disp"]
        eps["peak_root_lateral_velocity_toward_target"] = captured["peak_lateral_vel"]
        eps["episode_length"] = captured["ep_length"]
        eps["timeout_rate"] = captured["timeout"]
        eps["workspace_reset_rate"] = captured["workspace_reset"]
        eps["nan_reset_rate"] = captured["nan_reset"]

        # Clear dive-reach episode buffers
        self.ep_min_hand_dist[env_ids] = 999.0
        self.ep_success[env_ids] = 0.0
        self.ep_time_to_success[env_ids] = 0.0
        self.t_success[env_ids] = self._dr_cfg.dive_deadline_s
        self.ep_peak_root_lateral_vel[env_ids] = 0.0
        self.ep_root_lateral_disp[env_ids] = 0.0
        self.ep_workspace_reset[env_ids] = 0.0
        self.ep_nan_reset[env_ids] = 0.0
        self.ep_timeout[env_ids] = 0.0
        self._first_success_this_step[env_ids] = False

    # ═══════════════════════════════════════════════════════
    # Reset root states override
    # ═══════════════════════════════════════════════════════

    def _reset_root_states(self, env_ids):
        """Override: normal ball launch (not pinned to target), dive-reach specific buffers."""
        # Reset robot root state (same as parent)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if getattr(self.cfg.domain_rand, 'randomize_reset_velocity', True):
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=self.device)
        else:
            self.root_states[env_ids, 7:13] = 0.0

        # Normal ball launch (goalkeeper trajectory) — NOT placed at target_pos
        ball_vel = self.assign_ball_states(env_ids)
        self.ball_states[env_ids] = self.base_init_state
        self.ball_states[env_ids, :3] = self.ball_start[env_ids, :]
        self.ball_states[env_ids, 7:10] = ball_vel
        self.ball_states[env_ids, 10:13] = 0.0
        self.ball_vel[env_ids] = ball_vel[:, 0]

        # Dive-reach specific: clear per-episode flags
        self.has_in_air[env_ids] = False
        self.stop_flag[env_ids] = 0.0
        self.success_flag[env_ids] = 0.0
        self.t_success[env_ids] = self._dr_cfg.dive_deadline_s
        self._first_success_this_step[env_ids] = False
        self.dist[env_ids] = 5.0
        self.ball_last[env_ids] = 0.0
        self.vanish_step[env_ids] = 0
        self.catchstep[env_ids] = 0

        # Write to sim
        all_states = torch.cat((self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(all_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # ═══════════════════════════════════════════════════════
    # Randomize balls: no-op (ball trajectory irrelevant for dive-reach task)
    # ═══════════════════════════════════════════════════════

    def _randomize_balls(self):
        """No-op: ball forces don't matter for dive-reach reward."""
        pass

    # ═══════════════════════════════════════════════════════
    # Observations
    # ═══════════════════════════════════════════════════════

    def compute_observations(self):
        """Compute observations with target info in actor obs."""
        num_target = self.cfg.env.num_target_obs

        # Elapsed time and time remaining
        elapsed = self.episode_length_buf.float() * self.dt
        deadline = self._dr_cfg.dive_deadline_s
        time_remaining = torch.clamp(deadline - elapsed, min=0.0) / deadline  # normalized [0, 1]

        # Ball local (real ball position, normal goalkeeper launch; reward ignores ball)
        ball_local = quat_rotate_inverse(self.base_quat, self.ball_states[:, :3] - self.torso_pos)

        # Target info — LOCAL coordinates only, no env_origins
        local_target_y = self.local_target_y.unsqueeze(-1)
        target_z = self.target_pos[:, 2:3]  # z is absolute (ground at z=0)

        # Build actor obs: [ball_local(3), local_target_y(1), target_z(1), target_side(1), time_remaining(1),
        #                   ang_vel(3), gravity(3), dof_pos(29), dof_vel(29), actions(29)]
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

        # Add noise if enabled
        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        # History buffer: shift old, append new
        self.obs_buf = torch.cat(
            (self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)

        # Privileged obs: add base_lin_vel, end_region, target_local, ball_vel, hand positions, hand_dist
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

        # NaN guard
        if not torch.isfinite(self.obs_buf).all():
            bad = ~torch.isfinite(self.obs_buf)
            print("[NaN Obs] count:", bad.sum().item())
            print("[NaN Obs] rows:", bad.any(dim=1).nonzero()[:10].flatten().tolist())
            raise RuntimeError("Non-finite obs_buf in g1_dive_reach")
        if not torch.isfinite(self.privileged_obs_buf).all():
            bad = ~torch.isfinite(self.privileged_obs_buf)
            print("[NaN PrivObs] count:", bad.sum().item())
            raise RuntimeError("Non-finite privileged_obs_buf in g1_dive_reach")

    # ═══════════════════════════════════════════════════════
    # Reward accounting debug
    # ═══════════════════════════════════════════════════════

    def compute_reward(self):
        """Override: call parent, then compare visible sum vs actual rew_buf."""
        # Call parent's compute_reward
        super().compute_reward()

        # DEBUG: compare sum of logged reward terms vs actual rew_buf
        visible_sum = torch.zeros(self.num_envs, device=self.device)
        for name in self.reward_names:
            scale = self.reward_scales[name]
            fn = getattr(self, '_reward_' + name)
            visible_sum += fn() * scale

        diff = self.rew_buf - visible_sum
        abs_diff = diff.abs()
        if abs_diff.max() > 0.01:
            print(f"[REWARD_ACCT] WARN: max|rew_buf - visible_sum| = {abs_diff.max().item():.4f}")
            print(f"[REWARD_ACCT]       mean diff = {diff.mean().item():.4f}")
            print(f"[REWARD_ACCT]       reward_names = {self.reward_names}")
            # Check termination
            if "termination" in self.reward_scales:
                t = self._reward_termination() * self.reward_scales["termination"]
                print(f"[REWARD_ACCT]       termination_rew = mean {t.mean().item():.4f}")
            # Check if there's an extra term we're missing
            extra = self.rew_buf - visible_sum
            if "termination" in self.reward_scales:
                extra = extra - self._reward_termination() * self.reward_scales["termination"]
            if extra.abs().max() > 0.01:
                print(f"[REWARD_ACCT]       UNEXPLAINED extra: mean={extra.mean().item():.4f}, max={extra.max().item():.4f}")

    # ═══════════════════════════════════════════════════════
    # Rewards
    # ═══════════════════════════════════════════════════════

    def _reward_target_hand_reach(self):
        """Dense reward: exp(-hand_dist / sigma). Only active before deadline."""
        sigma = float(self._dr_cfg.target_hand_reach_sigma)
        deadline = self._dr_cfg.dive_deadline_s
        elapsed = self.episode_length_buf.float() * self.dt
        before_deadline = (elapsed <= deadline).float()
        rew = torch.exp(-self.hand_dist / sigma) * before_deadline
        if not torch.isfinite(rew).all():
            raise RuntimeError(f"NaN in target_hand_reach: hand_dist has NaN={torch.isnan(self.hand_dist).any()}")
        return rew

    def _reward_success_dive_reach(self):
        """Sparse ONE-TIME success reward (first success frame only).

        Conditions (ALL must be met):
          A. hand_dist < 0.15m (using correct-side rubber hand)
          B. elapsed_time < dive_deadline_s (0.8s)
          C. root_lateral_disp_toward_target > 0.4m
          D. hand position sanity checks (x window, z range, same side)
        peak_root_lateral_vel is logging-only, NOT a success condition.
        """
        deadline = self._dr_cfg.dive_deadline_s
        elapsed = self.episode_length_buf.float() * self.dt

        cond_a = self.hand_dist < self._dr_cfg.success_hand_dist_threshold
        cond_b = elapsed < deadline
        cond_c = self.ep_root_lateral_disp > self._dr_cfg.success_root_lateral_disp_min
        # peak_root_lateral_vel is logging-only, NOT a success condition

        # Hand sanity: use correct-side hand (pos y → left, neg y → right)
        left_pos = self.rigid_body_states[:, self._left_hand_idx, 0:3]
        right_pos = self.rigid_body_states[:, self._right_hand_idx, 0:3]
        use_left_hand = self.target_side > 0
        use_right_hand = self.target_side < 0
        hand_pos = torch.zeros_like(left_pos)
        hand_pos[use_left_hand] = left_pos[use_left_hand]
        hand_pos[use_right_hand] = right_pos[use_right_hand]
        neutral = ~(use_left_hand | use_right_hand)
        if neutral.any():
            dist_l = torch.norm(left_pos[neutral] - self.target_pos[neutral], dim=-1)
            dist_r = torch.norm(right_pos[neutral] - self.target_pos[neutral], dim=-1)
            hand_pos[neutral] = torch.where(
                (dist_l < dist_r).unsqueeze(-1), left_pos[neutral], right_pos[neutral])

        hand_x_ok = torch.abs(hand_pos[:, 0] - self.target_pos[:, 0]) < self._dr_cfg.hand_x_window
        hand_z_ok = (hand_pos[:, 2] > self._dr_cfg.hand_z_min) & (hand_pos[:, 2] < self._dr_cfg.hand_z_max)
        hand_y_ok = (hand_pos[:, 1] * self.target_side) > 0  # same side as target

        cond_e = hand_x_ok & hand_z_ok & hand_y_ok

        # First success only: current conditions met AND not already succeeded this episode
        first_success = cond_a & cond_b & cond_c & cond_e & (self.success_flag == 0)
        self.success_flag[first_success] = 1.0
        self.t_success[first_success] = elapsed[first_success]
        self._first_success_this_step = first_success  # consumed by _reward_fast_reach_bonus

        rew = first_success.float()
        if not torch.isfinite(rew).all():
            raise RuntimeError("NaN in success_dive_reach")
        return rew

    def _reward_fast_reach_bonus(self):
        """ONE-TIME bonus for early success: (deadline - elapsed_now) / deadline.

        Uses _first_success_this_step set by _reward_success_dive_reach.
        Computes bonus from current elapsed_time on the first-success frame,
        NOT from self.t_success (which is NaN for non-success envs → 0*NaN = NaN).
        """
        deadline = self._dr_cfg.dive_deadline_s
        elapsed_now = self.episode_length_buf.float() * self.dt
        bonus_now = torch.clamp((deadline - elapsed_now) / deadline, min=0.0, max=1.0)
        # torch.where avoids 0.0 * NaN → NaN for non-success envs
        rew = torch.where(self._first_success_this_step, bonus_now, torch.zeros_like(bonus_now))
        if not torch.isfinite(rew).all():
            raise RuntimeError("NaN in fast_reach_bonus")
        return rew

    def _reward_launch_gate_bonus(self):
        """Small bonus when hand is close AND root has launched laterally.

        Only active before deadline.
        """
        deadline = self._dr_cfg.dive_deadline_s
        elapsed = self.episode_length_buf.float() * self.dt
        before_deadline = (elapsed <= deadline).float()
        close = self.hand_dist < 0.3
        launched = self.root_lateral_disp_toward_target > 0.25
        rew = (close & launched).float() * before_deadline
        if not torch.isfinite(rew).all():
            raise RuntimeError("NaN in launch_gate_bonus")
        return rew

    # Override old goalkeeper reward methods to return zero
    # (they won't be called since scales are 0, but safe)
    def _reward_eereach(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_success(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_stopball(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_stayonline(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_noretreat(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_successland(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_feetorientaion(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_slippage(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_penalize_sharpcontact(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_penalize_kneeheight(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_postorientation(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_postangvel(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_postlinvel(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_postupperdofpos(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_postwaistdofpos(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_deviation_waist_pitch_joint(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_reach_ball(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_lateral_launch(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_lateral_displacement(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_hand_target(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_contact_proxy(self):
        return torch.zeros(self.num_envs, device=self.device)

    # ═══════════════════════════════════════════════════════
    # Compute termination observations (override for target)
    # ═══════════════════════════════════════════════════════

    def compute_termination_observations(self, env_ids):
        """Return privileged obs for terminated envs."""
        return self.privileged_obs_buf[env_ids]
