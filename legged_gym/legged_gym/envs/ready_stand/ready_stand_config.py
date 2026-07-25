"""Ready-stance configs for the existing Q1 and G1 goalkeeper robots."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.g1.g1_29_config import G129Cfg
from legged_gym.envs.q1.q1_goalkeeper_config import Q1GoalkeeperCfg


class _ReadyStandCfgMixin:
    class env:
        num_actor_history = 10
        num_ballobs = 0
        episode_length_s = 8
        ball_gravity = False
        play = False

    class domain_rand:
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.95, 1.05]
        initial_joint_pos_offset = [-0.05, 0.05]
        continue_keep = False
        randomize_reset_velocity = True
        push_robots = True
        push_interval_s = 2.0
        max_push_vel_xy = 0.35
        ball_interval_s = 999.0
        max_ball_vel = 0.0

    class rewards:
        class scales:
            upright = 2.0
            base_height = 1.0
            ready_pose = 2.0
            foot_contact = 0.5
            lin_vel_xy = -0.5
            ang_vel_xy = -0.25
            feet_slip = -0.2
            dof_acc = -2.5e-7
            smoothness = -0.05
            torques = -1e-5
            dof_vel = -5e-4
            dof_pos_limits = -3.0
            dof_vel_limits = -2.0
            torque_limits = -2.0
        only_positive_rewards = False
        target_base_height = 0.0
        base_height_sigma = 0.02
        ready_pose_sigma = 0.08


class Q1ReadyStandCfg(_ReadyStandCfgMixin, Q1GoalkeeperCfg):
    class env(Q1GoalkeeperCfg.env):
        num_envs = 2048
        num_actor_history = 10
        num_actions = 22
        num_dofs = 22
        num_ballobs = 0
        num_one_step_observations = 6 + num_dofs * 3
        num_privileged_obs = num_one_step_observations
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 8
        ball_gravity = False

    class init_state(Q1GoalkeeperCfg.init_state):
        # The requested Q1 stand-v2 ready pose, in current Q1 joint order.
        pos = [0.0, 0.0, 0.415]
        init_pos = [-0.2, 0.0, 0.0, 0.5, -0.2, 0.0,
                    -0.2, 0.0, 0.0, 0.5, -0.2, 0.0,
                    0.0, 0.0, 0.0, 0.14, 0.0, 1.3,
                    0.0, -0.14, 0.0, 1.3]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.2, 'left_hip_roll_joint': 0.0, 'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.5, 'left_ankle_pitch_joint': -0.2, 'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.2, 'right_hip_roll_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.5, 'right_ankle_pitch_joint': -0.2, 'right_ankle_roll_joint': 0.0,
            'waist_roll_joint': 0.0, 'waist_yaw_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0, 'left_shoulder_roll_joint': 0.14,
            'left_shoulder_yaw_joint': 0.0, 'left_elbow_joint': 1.3,
            'right_shoulder_pitch_joint': 0.0, 'right_shoulder_roll_joint': -0.14,
            'right_shoulder_yaw_joint': 0.0, 'right_elbow_joint': 1.3,
        }

    class dataset(Q1GoalkeeperCfg.dataset):
        load_motions = False

    class amp(Q1GoalkeeperCfg.amp):
        amp_coef = 0.0

    class domain_rand(Q1GoalkeeperCfg.domain_rand, _ReadyStandCfgMixin.domain_rand):
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.95, 1.05]
        initial_joint_pos_offset = [-0.05, 0.05]
        continue_keep = False
        randomize_reset_velocity = True
        push_robots = True
        push_interval_s = 2.0
        max_push_vel_xy = 0.35
        ball_interval_s = 999.0
        max_ball_vel = 0.0

    class rewards(Q1GoalkeeperCfg.rewards, _ReadyStandCfgMixin.rewards):
        class scales(_ReadyStandCfgMixin.rewards.scales):
            pass
        target_base_height = 0.415
        base_height_sigma = 0.02
        ready_pose_sigma = 0.08


class G1ReadyStandCfg(_ReadyStandCfgMixin, G129Cfg):
    class env(G129Cfg.env):
        num_envs = 2048
        num_actor_history = 10
        num_actions = 29
        num_dofs = 29
        num_ballobs = 0
        num_one_step_observations = 6 + num_dofs * 3
        num_privileged_obs = num_one_step_observations
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 8
        ball_gravity = False

    class init_state(G129Cfg.init_state):
        # Match task 29's default joint-angle target and use it for reset too.
        init_pos = [-0.1, 0.2, 0.0, 0.45, -0.1, -0.2,
                    -0.1, -0.2, 0.0, 0.45, -0.1, 0.2,
                    0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0,
                    0.0, -0.5, 0.0, 1.2, 0.0, 0.0, 0.0]

    class dataset(G129Cfg.dataset):
        load_motions = False

    class amp(G129Cfg.amp):
        amp_coef = 0.0

    class domain_rand(G129Cfg.domain_rand, _ReadyStandCfgMixin.domain_rand):
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.95, 1.05]
        initial_joint_pos_offset = [-0.05, 0.05]
        continue_keep = False
        randomize_reset_velocity = True
        push_robots = True
        push_interval_s = 2.0
        max_push_vel_xy = 0.35
        ball_interval_s = 999.0
        max_ball_vel = 0.0

    class rewards(G129Cfg.rewards, _ReadyStandCfgMixin.rewards):
        class scales(_ReadyStandCfgMixin.rewards.scales):
            pass
        target_base_height = 0.8
        base_height_sigma = 0.03
        ready_pose_sigma = 0.10


class _ReadyStandCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        # HIMPPO ball and region supervision is goalkeeper-specific.
        use_auxiliary_estimators = False

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100
        max_iterations = 20000
        save_interval = 50
        logger = 'tensorboard'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


class Q1ReadyStandCfgPPO(_ReadyStandCfgPPO):
    class runner(_ReadyStandCfgPPO.runner):
        run_name = 'q1_ready_stand'
        experiment_name = 'q1_ready_stand'
        wandb_project = 'q1_ready_stand'

    amp = Q1ReadyStandCfg.amp


class G1ReadyStandCfgPPO(_ReadyStandCfgPPO):
    class runner(_ReadyStandCfgPPO.runner):
        run_name = 'g1_ready_stand'
        experiment_name = 'g1_ready_stand'
        wandb_project = 'g1_ready_stand'

    amp = G1ReadyStandCfg.amp
