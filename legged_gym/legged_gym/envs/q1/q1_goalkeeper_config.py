"""Q1 (22-DOF) Goalkeeper Config — adapts G1 pipeline to Q1."""
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Import G1 for commands reference
from legged_gym.envs.g1.g1_29_config import G129Cfg


class Q1GoalkeeperCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # smoke: smaller scale
        num_actor_history = 10
        num_actions = 22        # Q1: 22 DOF (no waist_pitch, no wrist)
        num_dofs = 22
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions
        num_privileged_obs = num_one_step_observations + 3 + 1 + 6 + 6 + 1
        num_observations = num_actor_history * num_one_step_observations
        env_spacing = 5.
        send_timeouts = True
        episode_length_s = 3
        ball_gravity = True
        play = False
        mode_weights = [1, 1, 1, 1, 1, 1]  # uniform, set e.g. [1,1,2,2,1,1] for high-ball oversample

    # Q1 commands: G1 ranges scaled by 0.65 (height-based)
    class commands:
        class ranges_0:
            height = [0.26, 0.78]; width = [0.13, 0.78]
            maxh = [0.20, 0.98]; maxw = [0.0, 1.17]
            evalh = [0.20, 0.98]; evalw = [0.0, 0.98]
        class ranges_1:
            height = [0.26, 0.78]; width = [-0.78, -0.13]
            maxh = [0.20, 0.98]; maxw = [-1.17, 0.0]
            evalh = [0.20, 0.98]; evalw = [-0.98, 0.0]
        class ranges_2:
            height = [0.78, 1.04]; width = [0.0, 0.65]
            maxh = [0.78, 1.17]; maxw = [0.0, 0.98]
            evalh = [0.78, 1.17]; evalw = [0.0, 0.98]
        class ranges_3:
            height = [0.78, 1.04]; width = [-0.65, 0.0]
            maxh = [0.78, 1.17]; maxw = [-0.98, 0.0]
            evalh = [0.78, 1.17]; evalw = [-0.98, 0.0]
        class ranges_4:
            height = [0.07, 0.20]; width = [0.13, 0.78]
            maxh = [0.07, 0.20]; maxw = [0.0, 1.17]
            evalh = [0.07, 0.20]; evalw = [0.0, 0.98]
        class ranges_5:
            height = [0.07, 0.20]; width = [-0.78, -0.13]
            maxh = [0.07, 0.20]; maxw = [-1.17, 0.0]
            evalh = [0.07, 0.20]; evalw = [-0.98, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.415]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.087,  'left_hip_roll_joint': 0.0,    'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.175,        'left_ankle_pitch_joint': -0.087, 'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.087, 'right_hip_roll_joint': 0.0,    'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.175,       'right_ankle_pitch_joint': -0.087, 'right_ankle_roll_joint': 0.0,
            'waist_roll_joint': 0.0,        'waist_yaw_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,  'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,    'left_elbow_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0, 'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,   'right_elbow_joint': 0.0,
        }
        init_pos = [-0.087,0,0,0.175,-0.087,0,-0.087,0,0,0.175,-0.087,0,0,0,0,0,0,0,0,0,0,0]
        # Boost anti-spin torque (applied after env init)
        torque_limits_scale = {
            'left_hip_yaw_joint': 2.0, 'right_hip_yaw_joint': 2.0,
            'left_ankle_roll_joint': 1.8, 'right_ankle_roll_joint': 1.8,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # Normal Q1 PD stiffness (from q1_22dof_rl_collision)
        stiffness = {'hip_yaw': 30, 'hip_roll': 30, 'hip_pitch': 30, 'knee': 30, 'ankle': 20,
                     'waist': 80, 'shoulder': 20, 'elbow': 20}
        damping = {'hip_yaw': 1.5, 'hip_roll': 1.5, 'hip_pitch': 1.5, 'knee': 1.5, 'ankle': 1.0,
                   'waist': 2.0, 'shoulder': 1.0, 'elbow': 1.0}
        action_scale = 0.25
        # Q1 per-joint action scale: reduce crossing-prone joints, boost arms
        per_joint_action_scale = {
            # legs: reduce
            'left_hip_yaw_joint': 0.10, 'right_hip_yaw_joint': 0.10,
            'left_hip_roll_joint': 0.12, 'right_hip_roll_joint': 0.12,
            'left_ankle_roll_joint': 0.10, 'right_ankle_roll_joint': 0.10,
            # arms: boost (was 0.25 global → 0.50)
            'left_shoulder_pitch_joint': 0.50,  'right_shoulder_pitch_joint': 0.50,
            'left_shoulder_roll_joint': 0.50,   'right_shoulder_roll_joint': 0.50,
            'left_shoulder_yaw_joint': 0.50,    'right_shoulder_yaw_joint': 0.50,
            'left_elbow_joint': 0.50,           'right_elbow_joint': 0.50,
        }
        decimation = 4
        curriculum_joints = ['waist_yaw_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                             'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
                           'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
                            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                           'left_shoulder_yaw_joint', 'left_elbow_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                            'right_shoulder_yaw_joint', 'right_elbow_joint']
        elbow_joints = ['left_elbow_joint', 'right_elbow_joint']
        wrist_joints = []  # Q1: no wrist
        upper_body_link = "pelvis"
        torso_link = "torso_link"
        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']

    class termination:
        knee_height_threshold = 0.06
        gravity_threshold = 0.8

    class terrain:
        static_friction = 1.0; dynamic_friction = 1.0; restitution = 0.

    class normalization:
        class obs_scales:
            lin_vel = 2.0; ang_vel = 0.25; dof_pos = 1.0; dof_vel = 0.05
            ball_vel = 0.2; ball_pos = 0.3; height_measurements = 5.0
        clip_observations = 100.; clip_actions = 100.

    class noise(G129Cfg.noise):
        pass

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/q1/q1_box.urdf'
        ballfile = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/urdf/ball.urdf'
        name = "q1"
        foot_name = "ankle_roll_link"
        contact_foot_names = "ankle_roll_link"
        hand_name = "elbow_link"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "elbow", "pelvis", "head"]
        terminate_after_contacts_on = []
        waist_joints = ["waist_roll_joint", "waist_yaw_joint"]
        ankle_joints = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        imu_link = "torso_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        keyframe_name = "torso_link"
        disable_gravity = False; collapse_fixed_joints = True; fix_base_link = False
        default_dof_drive_mode = 3; self_collisions = 1  # collision filter: must be 1 for ground contact!
        replace_cylinder_with_capsule = True; flip_visual_attachments = False
        density = 1.0; angular_damping = 0.0; linear_damping = 0.0
        max_angular_velocity = 1000.; max_linear_velocity = 1000.
        armature = 0.004; thickness = 0.01
        hand_offset = [0.14, 0.0, 0.0]   # push reference point from elbow COM toward forearm tip (inertia: L≈24cm, COM@9.8cm, tip@14cm from COM)

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_joint_injection = False; randomize_actuation_offset = False
        randomize_payload_mass = True; randomize_com_displacement = False
        randomize_link_mass = False
        randomize_friction = True
        friction_range = [0.6, 1.4]
        randomize_restitution = False
        restitution_range = [0.0, 0.1]
        randomize_kp = True; randomize_kd = True
        kp_range = [0.85, 1.15]; kd_range = [0.85, 1.15]
        randomize_initial_joint_pos = False
        continue_keep = False  # needed by leg ddl robot reset
        push_robots = False; push_interval_s = 4.0; max_push_vel_xy = 0.25
        ball_interval_s = 999.0; max_ball_vel = 0.0  # disabled: ball obs noise is sufficient DR
        randomize_reset_velocity = False  # Q1 goalkeeper: zero reset velocity
        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]
        payload_mass_range = [-0.5, 1.0]

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            eereach = 10.0; success = 5.0; stopball = 120.0
            stayonline = -2.0; noretreat = -2.0
            successland = 5.0; feetorientaion = 3.0
            penalize_sharpcontact = -100.; penalize_kneeheight = -100.; feet_slippage = 3.0
            postorientation = 3.0; postangvel = 3.0; postupperdofpos = 1.0; postwaistdofpos = 1.0; postlinvel = 1.0
            ang_vel_xy = -0.1; dof_acc = -2.5e-7; smoothness = -0.1
            torques = -1e-5; dof_vel = -5e-4
            dof_pos_limits = -3.0; dof_vel_limits = -2.0; torque_limits = -2.0
            penalty_feet_separation = 0.0  # disabled: must match feet_sep_enabled=False
            upright_penalty = -3.0         # proximity penalty for approaching gravity threshold (set 0 to disable)
            highball_jump_height = 0.0       # reward root_z > 0.45 on high-ball modes (set 0 to disable)
            highball_upward_velocity = 0.0   # reward root_vz > 0 on high-ball modes
            highball_upright_penalty = -3.0  # extra upright penalty on high-ball modes only
        only_positive_rewards = False
        successland_jump_threshold = 0.53    # root_z above this triggers 'has_in_air' (G1 default=0.55)
        highball_jump_z_threshold = 0.45     # root_z above this starts earning jump reward
        highball_jump_z_range = 0.15         # maps threshold→threshold+range to reward 0→1
        catch_th = 0.5; handheight_th = 1.0; reach_th = 0.2; strict_th = 0.15
        target_dof_pos_sigma = -20; tracking_sigma = 0.25; catch_sigma = 5.0
        soft_dof_pos_limit = 0.9; soft_dof_vel_limit = 0.9; soft_torque_limit = 0.95
        # Upright penalty: penalises tilt toward gravity_threshold (0.8)
        upright_threshold = 0.95             # danger starts when upright < this
        upright_deadzone = 0.15              # maps threshold→gravity_threshold (0.8)
        upright_ball_visible_only = True     # only penalise when ball is in flight
        # Q1 feet separation tracking
        feet_sep_enabled = False
        min_foot_sep = 0.12
        feet_cross_threshold = 0.06
        feet_too_close_threshold = 0.10
        max_contact_force = 500.  # raised for box collision URDF

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30; min_time = 0.1

    class amp:
        obs_type = 'dof'; num_obs = 22 * 2; amp_coef = 0.4; num_steps = 2


class Q1GoalkeeperCfgHard(Q1GoalkeeperCfg):
    """Harder domain randomization for sim2sim robustness."""
    class domain_rand(Q1GoalkeeperCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 2.0]
        randomize_payload_mass = True
        payload_mass_range = [-1.5, 3.0]
        push_robots = False; push_interval_s = 3.0; max_push_vel_xy = 0.5
        ball_interval_s = 999.0; max_ball_vel = 0.0  # disabled
        randomize_kp = True; randomize_kd = True
        kp_range = [0.6, 1.5]; kd_range = [0.6, 1.5]
        randomize_motor_strength = True
        motor_strength_range = [0.6, 1.5]
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.05, 0.05]
        randomize_reset_velocity = True


# --- Tier A: Medium DR (extends Hard with control/contact/ball/recovery DR) ---
class Q1GoalkeeperCfgMediumDR(Q1GoalkeeperCfgHard):
    """Tier A: Medium DR — action delay, torque noise, joint damping, foot slip, ball dropout, root tilt."""
    class domain_rand(Q1GoalkeeperCfgHard.domain_rand):
        # Action DR
        randomize_action_delay = True
        action_delay_steps = 2         # 0-2 policy steps
        randomize_action_filter = True
        action_filter_alpha_min = 0.4  # low-pass alpha
        randomize_torque_noise = True
        torque_noise_pct = 0.03        # ±3% torque noise
        # Joint dynamics DR
        randomize_joint_damping = True
        joint_damping_scale = [0.5, 2.0]
        randomize_armature_scale = True
        armature_scale = [0.5, 2.0]
        # Contact DR
        push_during_ball = False
        push_during_ball_vel = 0.0  # disabled: ball obs noise is sufficient DR
        randomize_restitution = True
        restitution_range = [0.0, 0.05]
        # Ball obs DR
        randomize_ball_obs_noise = True
        ball_obs_noise = 0.03          # ±0.03m position noise
        randomize_ball_obs_dropout = True
        ball_obs_dropout_pct = 0.10    # 10% dropout
        # Recovery DR
        randomize_initial_root_tilt = True
        initial_root_tilt_deg = 5.0    # ±5° initial tilt
        # Side curriculum
        oversample_right_side = True
        right_side_ratio = 0.5         # 50% right-side shots


# --- Tier B: Contact DR (focus on foot-ground contact robustness) ---
class Q1GoalkeeperCfgContactDR(Q1GoalkeeperCfgMediumDR):
    """Tier B: Contact DR — ground slope, push at contact loss, ball obs jitter/delay."""
    class domain_rand(Q1GoalkeeperCfgMediumDR.domain_rand):
        randomize_ground_slope = True
        ground_slope_deg = 3.0         # ±3° ground tilt
        randomize_ground_height = True
        ground_height_noise = 0.01     # ±0.01m
        push_at_contact_loss = False
        push_at_contact_loss_vel = 0.0  # disabled
        randomize_ball_obs_delay = True
        ball_obs_delay_steps = 2       # 0-2 steps delay
        ball_vanish_jitter = True
        ball_vanish_jitter_steps = 2   # ±1-2 steps vanish jitter
        randomize_contact_friction = True
        contact_friction_range = [0.4, 1.6]
        # Stronger joint output DR
        torque_noise_pct = 0.05        # 3→5% torque noise
        action_delay_steps = 3         # 2→3 steps delay
        action_filter_alpha_min = 0.3  # 0.4→0.3 (stronger smoothing)


# --- Tier C: Failure-case finetune (target right-side asymmetry) ---
class Q1GoalkeeperCfgFinetuneDR(Q1GoalkeeperCfgContactDR):
    """Tier C: Failure-case finetune — right-side oversample, hip_roll penalty, foot contact keep, mirror aug."""
    class domain_rand(Q1GoalkeeperCfgContactDR.domain_rand):
        oversample_right_side = True
        right_side_ratio = 0.6         # 60% right-side (mode 1,3,5)
        side_wise_metrics = True       # track left/right success separately
        mirror_augmentation = True     # mirror left→right samples
        mirror_augmentation_pct = 0.3  # 30% of samples mirrored
        # Penalty DR
        hip_roll_action_penalty = True
        hip_roll_penalty_weight = 0.01
        foot_contact_keep_bonus = True
        foot_contact_keep_weight = 0.5
        root_pitch_penalty_after_ball = True
        root_pitch_penalty_weight = 0.05
        foot_slip_penalty = True
        foot_slip_penalty_weight = 0.01


# ============================================================================
# Weak Stage 1: Targeted hip-pitch / backward-fall DR (reduced intensity)
# ============================================================================

class Q1GoalkeeperCfgStage1(Q1GoalkeeperCfgContactDR):
    """Weak Stage 1: minimal backward-lean + hip actuator DR to preserve goalkeeper skill."""
    class domain_rand(Q1GoalkeeperCfgContactDR.domain_rand):
        # --- Backward lean reset DR (weak) ---
        randomize_backward_lean_reset = True
        root_pitch_noise_deg = 2.0           # ±2° (was 5°)
        root_pitch_backward_bias = 0.6
        root_pitch_vel_noise = 0.2           # ±0.2 rad/s (was 0.5)
        hip_pitch_init_noise = 0.04          # ±0.04 rad (was 0.10)
        ankle_pitch_init_noise = 0.03        # ±0.03 rad (was 0.08)
        # --- Hip-pitch actuator DR (weak) ---
        randomize_hip_pitch_actuator = True
        hip_pitch_motor_strength_scale = [0.85, 1.15]
        hip_pitch_kp_scale = [0.85, 1.15]
        hip_pitch_kd_scale = [0.9, 1.3]
        hip_pitch_action_delay_steps = 1     # 0-1 step
        hip_pitch_action_filter_alpha = [0.7, 1.0]
        hip_pitch_torque_noise_pct = 0.02    # 2%
        # --- Ball obs DR (weak) ---
        ball_obs_noise = 0.02               # ±0.02m (was 0.05)
        ball_obs_dropout_pct = 0.05         # 5% (was 15%)
        ball_obs_delay_steps = 1            # 0-1 step (was 3)
        ball_vanish_jitter_steps = 1        # 0-1 step (was 3)
        # Disabled for now
        randomize_history_ball_dropout = False
        randomize_ball_visible_pitch_disturb = False

    # Rewards: unchanged from contact_dr (no extra penalties)


class Q1GoalkeeperCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'  # use HIMPPO (only available algo)
        num_steps_per_env = 100
        max_iterations = 20000
        save_interval = 50
        run_name = 'q1_goalkeeper_smoke'
        experiment_name = 'q1'
        wandb_project = "q1_goalkeeper"
        logger = 'tensorboard'
        resume = False; load_run = -1; checkpoint = -1; resume_path = None
    amp = Q1GoalkeeperCfg.amp
