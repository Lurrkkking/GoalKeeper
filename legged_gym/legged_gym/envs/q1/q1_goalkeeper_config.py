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
        push_robots = True; push_interval_s = 4.0; max_push_vel_xy = 0.25
        ball_interval_s = 0.5; max_ball_vel = 0.5
        randomize_reset_velocity = False  # Q1 goalkeeper: zero reset velocity
        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]
        payload_mass_range = [-0.5, 1.0]

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            eereach = 20.0; success = 20.0; stopball = 150.0
            stayonline = -1.0; noretreat = -1.0
            successland = 10.0; feetorientaion = 1.0
            penalize_sharpcontact = -100.; penalize_kneeheight = -100.; feet_slippage = 3.0
            postorientation = 3.0; postangvel = 3.0; postupperdofpos = 1.0; postwaistdofpos = 1.0; postlinvel = 1.0
            ang_vel_xy = -0.3; dof_acc = -2.5e-7; smoothness = -0.15
            torques = -1e-5; dof_vel = -5e-4
            dof_pos_limits = -3.0; dof_vel_limits = -2.0; torque_limits = -3.0
            penalty_feet_separation = 0.0  # disabled: must match feet_sep_enabled=False
        only_positive_rewards = False
        catch_th = 0.5; handheight_th = 1.0; reach_th = 0.2; strict_th = 0.15
        target_dof_pos_sigma = -20; tracking_sigma = 0.25; catch_sigma = 5.0
        soft_dof_pos_limit = 0.9; soft_dof_vel_limit = 0.9; soft_torque_limit = 0.95
        # Q1 feet separation tracking
        feet_sep_enabled = False
        min_foot_sep = 0.12
        feet_cross_threshold = 0.06
        feet_too_close_threshold = 0.10
        max_contact_force = 100000.  # raised for box collision URDF

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30; min_time = 0.1

    class amp:
        obs_type = 'dof'; num_obs = 22 * 2; amp_coef = 0.0; num_steps = 2


class Q1GoalkeeperCfgHard(Q1GoalkeeperCfg):
    """Harder domain randomization for sim2sim robustness."""
    class domain_rand(Q1GoalkeeperCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 2.0]           # wider: ice to sticky
        randomize_payload_mass = True
        payload_mass_range = [-1.5, 3.0]      # wider mass variation
        push_robots = True; push_interval_s = 3.0; max_push_vel_xy = 0.5  # stronger push
        ball_interval_s = 0.3; max_ball_vel = 1.0
        randomize_kp = True; randomize_kd = True
        kp_range = [0.6, 1.5]; kd_range = [0.6, 1.5]  # wider PD gain variation
        randomize_motor_strength = True
        motor_strength_range = [0.6, 1.5]     # wider motor strength
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.05, 0.05]
        randomize_reset_velocity = True        # random initial velocity


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
