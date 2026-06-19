"""Q1 Extreme Dive Goalkeeper Config — Stage 1 hard-shot fine-tuning.

Copied from original goalkeeper task for extreme dive stage-1 experiments.
Hard lateral shots from 7m penalty spot, dive-friendly terminations,
reach + lateral launch rewards, reduced upright/standing constraints.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.g1.g1_29_config import G129Cfg


class Q1ExtremeDiveCfg(LeggedRobotCfg):
    """Q1 extreme dive goalkeeper — hard lateral shots, dive window, reach rewards."""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actor_history = 10
        num_actions = 22
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
        # Extreme dive: only 6 hard-lateral modes, curated heights
        mode_weights = [1, 1, 1, 1, 1, 1]
        # Stage 1 ball start distance from goal
        ball_start_distance = 7.0  # [m]

    # ── Hard-shot commands (6 modes: R/L × low/mid/high at edges) ──
    class commands:
        class ranges_0:
            height = [0.20, 0.40]; width = [0.6, 0.9]       # right, low
            maxh = [0.20, 0.40]; maxw = [0.6, 0.9]
            evalh = [0.20, 0.40]; evalw = [0.6, 0.9]
        class ranges_1:
            height = [0.20, 0.40]; width = [-0.9, -0.6]      # left, low
            maxh = [0.20, 0.40]; maxw = [-0.9, -0.6]
            evalh = [0.20, 0.40]; evalw = [-0.9, -0.6]
        class ranges_2:
            height = [0.40, 0.80]; width = [0.6, 0.9]       # right, mid
            maxh = [0.40, 0.80]; maxw = [0.6, 0.9]
            evalh = [0.40, 0.80]; evalw = [0.6, 0.9]
        class ranges_3:
            height = [0.40, 0.80]; width = [-0.9, -0.6]      # left, mid
            maxh = [0.40, 0.80]; maxw = [-0.9, -0.6]
            evalh = [0.40, 0.80]; evalw = [-0.9, -0.6]
        class ranges_4:
            height = [0.80, 1.20]; width = [0.9, 1.2]       # right, high (extreme)
            maxh = [0.80, 1.20]; maxw = [0.9, 1.2]
            evalh = [0.80, 1.20]; evalw = [0.9, 1.2]
        class ranges_5:
            height = [0.80, 1.20]; width = [-1.2, -0.9]      # left, high (extreme)
            maxh = [0.80, 1.20]; maxw = [-1.2, -0.9]
            evalh = [0.80, 1.20]; evalw = [-1.2, -0.9]

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
        torque_limits_scale = {
            'left_hip_yaw_joint': 2.0, 'right_hip_yaw_joint': 2.0,
            'left_ankle_roll_joint': 1.8, 'right_ankle_roll_joint': 1.8,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_yaw': 30, 'hip_roll': 30, 'hip_pitch': 30, 'knee': 30, 'ankle': 20,
                     'waist': 80, 'shoulder': 20, 'elbow': 20}
        damping = {'hip_yaw': 1.5, 'hip_roll': 1.5, 'hip_pitch': 1.5, 'knee': 1.5, 'ankle': 1.0,
                   'waist': 2.0, 'shoulder': 1.0, 'elbow': 1.0}
        action_scale = 0.25
        per_joint_action_scale = {
            'left_hip_yaw_joint': 0.10, 'right_hip_yaw_joint': 0.10,
            'left_hip_roll_joint': 0.12, 'right_hip_roll_joint': 0.12,
            'left_ankle_roll_joint': 0.10, 'right_ankle_roll_joint': 0.10,
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
        wrist_joints = []
        upper_body_link = "pelvis"
        torso_link = "torso_link"
        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']

    class termination:
        knee_height_threshold = 0.06
        gravity_threshold = 0.8
        enable_dive_window = True
        dive_window_buffer_time = 0.2         # [s] extra time after ball passes before early reset

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
        default_dof_drive_mode = 3; self_collisions = 1
        replace_cylinder_with_capsule = True; flip_visual_attachments = False
        density = 1.0; angular_damping = 0.0; linear_damping = 0.0
        max_angular_velocity = 1000.; max_linear_velocity = 1000.
        armature = 0.004; thickness = 0.01
        hand_offset = [0.14, 0.0, 0.0]

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
        continue_keep = False
        push_robots = False; push_interval_s = 4.0; max_push_vel_xy = 0.25
        ball_interval_s = 999.0; max_ball_vel = 0.0
        randomize_reset_velocity = False
        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]
        payload_mass_range = [-0.5, 1.0]

    # ── Extreme Dive Rewards ──
    # Lower upright/standing weights, add dive-specific rewards.
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # Task rewards
            eereach = 10.0; success = 5.0; stopball = 120.0
            # Move rewards (relaxed)
            stayonline = -1.0; noretreat = -1.0
            # Feet rewards
            successland = 5.0; feetorientaion = 1.0  # reduced
            penalize_sharpcontact = -100.; penalize_kneeheight = -100.; feet_slippage = 3.0
            # Post rewards (reduced upright constraints for diving)
            postorientation = 1.5; postangvel = 1.5
            postupperdofpos = 0.5; postwaistdofpos = 0.5; postlinvel = 1.0
            # Reg rewards (reduced)
            ang_vel_xy = -0.05; dof_acc = -2.5e-7; smoothness = -0.1
            torques = -1e-5; dof_vel = -5e-4
            dof_pos_limits = -3.0; dof_vel_limits = -2.0; torque_limits = -2.0
            penalty_feet_separation = 0.0
            upright_penalty = -1.0              # reduced from -3.0: allow diving tilt
            highball_jump_height = 0.0
            highball_upward_velocity = 0.0
            highball_upright_penalty = -1.0     # reduced
            # ── Dive-specific rewards ──
            reach_ball = 3.0                    # reward min(hand/forearm distance to ball)
            lateral_launch = 2.0                # lateral velocity toward target side
            lateral_displacement = 1.0          # base lateral displacement toward target
            hand_target = 2.0                   # hand/forearm proximity to ball target pos
            contact_proxy = 5.0                 # bonus when min_ball_body_dist < 0.12

        only_positive_rewards = False
        successland_jump_threshold = 0.53
        highball_jump_z_threshold = 0.45
        highball_jump_z_range = 0.15
        catch_th = 0.5; handheight_th = 1.0; reach_th = 0.2; strict_th = 0.15
        target_dof_pos_sigma = -20; tracking_sigma = 0.25; catch_sigma = 5.0
        soft_dof_pos_limit = 0.9; soft_dof_vel_limit = 0.9; soft_torque_limit = 0.95
        upright_threshold = 0.95
        upright_deadzone = 0.15
        upright_ball_visible_only = True
        feet_sep_enabled = False
        min_foot_sep = 0.12
        feet_cross_threshold = 0.06
        feet_too_close_threshold = 0.10
        max_contact_force = 500.
        # Dive reward config
        contact_proxy_threshold = 0.12         # [m] ball-body distance considered "contact"
        dive_blocker_bodies = ["left_hand_link", "right_hand_link",
                               "left_forearm_link", "right_forearm_link",
                               "left_elbow_link", "right_elbow_link"]

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30; min_time = 0.1

    class amp:
        obs_type = 'dof'; num_obs = 22 * 2; amp_coef = 0.4; num_steps = 2


class Q1ExtremeDiveCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        experiment_name = 'q1'
        run_name = ''
        wandb_project = 'legged_gym'
        max_iterations = 20000
        save_interval = 200
        empirical_normalization = False
        resume = False
        load_run = -1
        checkpoint = -1
    # Must reference amp config from env config (HIMPPO runner reads train_cfg["amp"])
    amp = Q1ExtremeDiveCfg.amp
