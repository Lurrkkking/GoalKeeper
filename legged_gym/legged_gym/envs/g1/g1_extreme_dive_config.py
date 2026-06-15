"""G1 Extreme Dive Goalkeeper Config — Stage 1a skill discovery.

Copied from official G1 goalkeeper (g1_29_config.py) for extreme dive experiments.
STAGE-1a: narrow mid-high lateral shots, light DR, upper-body-only blockers,
boosted dive rewards, minimal posture constraints.
Goal: make G1 learn lateral dive / arm reaching toward hard-zone targets.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1ExtremeDiveCfg(LeggedRobotCfg):
    """G1 extreme dive goalkeeper — hard lateral shots, dive window, reach rewards."""

    class env(LeggedRobotCfg.env):
        num_envs = 6144
        num_actor_history = 10
        num_actions = 29
        num_dofs = 29
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions  # 96
        num_privileged_obs = 6 + num_ballobs + num_dofs * 2 + num_actions + 3 + 1 + 6 + 6 + 1  # 113
        num_observations = num_actor_history * num_one_step_observations  # 960
        env_spacing = 5.
        send_timeouts = True
        episode_length_s = 3
        ball_gravity = True
        play = False
        mode_weights = [1, 1, 0, 0, 0, 0]  # stage-1a: right/left mid-high only
        ball_start_distance = 7.0  # [m] penalty spot

    # ── Stage-1a: narrow mid-high lateral shots (no low, no extreme ±1.2) ──
    class commands:
        class ranges_0:
            height = [0.5, 1.1]; width = [0.6, 0.9]        # right, mid-high
            maxh = [0.5, 1.1]; maxw = [0.6, 0.9]
            evalh = [0.5, 1.1]; evalw = [0.6, 0.9]
        class ranges_1:
            height = [0.5, 1.1]; width = [-0.9, -0.6]      # left, mid-high
            maxh = [0.5, 1.1]; maxw = [-0.9, -0.6]
            evalh = [0.5, 1.1]; evalw = [-0.9, -0.6]
        class ranges_2:
            height = [0.5, 1.1]; width = [0.6, 0.9]        # right (dup, mode_weights=0)
            maxh = [0.5, 1.1]; maxw = [0.6, 0.9]
            evalh = [0.5, 1.1]; evalw = [0.6, 0.9]
        class ranges_3:
            height = [0.5, 1.1]; width = [-0.9, -0.6]      # left (dup, mode_weights=0)
            maxh = [0.5, 1.1]; maxw = [-0.9, -0.6]
            evalh = [0.5, 1.1]; evalw = [-0.9, -0.6]
        class ranges_4:
            height = [0.5, 1.1]; width = [0.6, 0.9]        # right (dup, mode_weights=0)
            maxh = [0.5, 1.1]; maxw = [0.6, 0.9]
            evalh = [0.5, 1.1]; evalw = [0.6, 0.9]
        class ranges_5:
            height = [0.5, 1.1]; width = [-0.9, -0.6]      # left (dup, mode_weights=0)
            maxh = [0.5, 1.1]; maxw = [-0.9, -0.6]
            evalh = [0.5, 1.1]; evalw = [-0.9, -0.6]

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
        knee_height_threshold = 0.06
        gravity_threshold = 0.8
        enable_dive_window = True          # dive-friendly: suppress early resets during save attempt
        dive_window_buffer_time = 0.2      # [s] after ball passes goal line

    class terrain:
        static_friction = 1.0; dynamic_friction = 1.0; restitution = 0.

    class normalization:
        class obs_scales:
            lin_vel = 2.0; ang_vel = 0.25; dof_pos = 1.0; dof_vel = 0.05
            ball_vel = 0.2; ball_pos = 0.3; height_measurements = 5.0
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
        hand_name = "hand"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "elbow", "pelvis", "hand", "head"]
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

    # Stage-1a: light DR only — let skill discovery happen without heavy randomization
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_joint_injection = False; joint_injection_range = [-0.01, 0.01]
        randomize_actuation_offset = False; actuation_offset_range = [-0.01, 0.01]
        randomize_payload_mass = False; payload_mass_range = [-5, 10]
        randomize_com_displacement = False; com_displacement_range = [-0.1, 0.1]
        randomize_link_mass = False; link_mass_range = [0.8, 1.2]
        randomize_friction = True; friction_range = [0.8, 1.2]     # light friction DR only
        randomize_restitution = False; restitution_range = [0.0, 1.0]
        randomize_kp = False; kp_range = [0.8, 1.2]
        randomize_kd = False; kd_range = [0.8, 1.2]
        randomize_initial_joint_pos = False; initial_joint_pos_scale = [0.5, 1.5]; initial_joint_pos_offset = [-0.1, 0.1]
        continue_keep = True
        push_robots = False; push_interval_s = 15; max_push_vel_xy = 1.5
        ball_interval_s = 999.0; max_ball_vel = 0.0               # no random ball push
        delay = False

    # ── Stage-1a Dive Rewards: boost dive, minimize posture penalties ──
    class rewards:
        class scales:
            # Task rewards
            eereach = 10.0; success = 5.0; stopball = 100.0
            # Move rewards (relaxed)
            stayonline = -0.5; noretreat = -0.5
            # Feet rewards (heavily reduced — don't punish diving)
            successland = 0.5; feetorientaion = 0.5
            penalize_sharpcontact = -20.; penalize_kneeheight = -20.; feet_slippage = 1.0
            # Post rewards (minimal — allow full-body diving)
            postorientation = 0.5; postangvel = 0.5
            postupperdofpos = 0.3; postwaistdofpos = 0.3; postlinvel = 0.3
            # Reg rewards
            ang_vel_xy = -0.03; dof_acc = -2.5e-7; smoothness = -0.05
            torques = -1e-5; dof_vel = -5e-4
            dof_pos_limits = -1.0; dof_vel_limits = -1.0; torque_limits = -1.0
            deviation_waist_pitch_joint = -0.0005
            # ── Dive-specific rewards (boosted for stage-1a) ──
            reach_ball = 5.0
            lateral_launch = 4.0
            lateral_displacement = 3.0
            hand_target = 3.0
            contact_proxy = 5.0

        only_positive_rewards = False
        catch_th = 0.5; handheight_th = 1.0; reach_th = 0.2; strict_th = 0.15
        target_dof_pos_sigma = -20; tracking_sigma = 0.25; catch_sigma = 5.0
        soft_dof_pos_limit = 0.9; soft_dof_vel_limit = 0.9; soft_torque_limit = 0.95
        max_contact_force = 1000.
        # Dive reward config
        contact_proxy_threshold = 0.12
        # Stage-1a: upper-body only blockers (hands, wrists, elbows, shoulders)
        dive_blocker_bodies = [
            "left_rubber_hand", "right_rubber_hand",
            "left_wrist_roll_link", "right_wrist_roll_link",
            "left_elbow_link", "right_elbow_link",
            "left_shoulder_roll_link", "right_shoulder_roll_link",
        ]

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30; min_time = 0.1

    class amp:
        obs_type = 'dof'; num_obs = 29 * 2; amp_coef = 0.4; num_steps = 2


class G1ExtremeDiveCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        experiment_name = 'g1'
        run_name = ''
        wandb_project = 'legged_gym'
        max_iterations = 20000
        save_interval = 200
        empirical_normalization = False
        resume = False; load_run = -1; checkpoint = -1
    amp = G1ExtremeDiveCfg.amp
