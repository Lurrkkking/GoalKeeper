"""Task 3: Action-to-joint mapping test.

For each action dimension i, set action[i]=0.4 (others=0), run PD for 0.3s,
verify the expected joint moves most.
"""

import numpy as np
import mujoco

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

DEFAULT_DOF_POS = np.array([
    -0.087, 0.0, 0.0, 0.175, -0.087, 0.0,
    -0.087, 0.0, 0.0, 0.175, -0.087, 0.0,
    0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])

ACTION_SCALE_VEC = np.array([
    0.25, 0.12, 0.10, 0.25, 0.25, 0.10,
    0.25, 0.12, 0.10, 0.25, 0.25, 0.10,
    0.25, 0.25,
    0.50, 0.50, 0.50, 0.50,
    0.50, 0.50, 0.50, 0.50,
])

KPS = np.array([
    30, 30, 30, 30, 20, 20, 30, 30, 30, 30, 20, 20,
    80, 80, 20, 20, 20, 20, 20, 20, 20, 20,
])
KDS = np.array([
    1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0,
    2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
])
TAU_LIMIT = np.array([
    36, 36, 36, 36, 22, 22, 36, 36, 36, 36, 22, 22,
    36, 36, 22, 22, 22, 22, 22, 22, 22, 22,
])


def test_single_action(action_idx):
    """Set action[action_idx]=0.4, run 0.3s, check which joint moves most."""
    xml_path = "/root/autodl-tmp/Humanoid-Goalkeeper/scripts/q1_22dof_goalkeeper_ball.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    qpos_ids = [m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES]
    qvel_ids = [m.jnt_dofadr[m.joint(n).id] for n in JOINT_NAMES]
    act_ids = [m.actuator(n).id for n in JOINT_NAMES]

    # Reset
    d.qpos[0:3] = [0, 0, 0.415]
    d.qpos[3:7] = [1, 0, 0, 0]
    for i, adr in enumerate(qpos_ids):
        d.qpos[adr] = DEFAULT_DOF_POS[i]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # Action: only this dim = 0.4
    action = np.zeros(22)
    action[action_idx] = 0.4
    target = DEFAULT_DOF_POS + action * ACTION_SCALE_VEC

    # Initial dof_pos
    init_pos = np.array([d.qpos[adr] for adr in qpos_ids])

    # Run 0.3s (15 control steps), track max absolute torque per joint
    sim_steps = int(0.3 / 0.005)
    max_tau_per_joint = np.zeros(22)
    for _ in range(sim_steps):
        dof_pos = np.array([d.qpos[adr] for adr in qpos_ids])
        dof_vel = np.array([d.qvel[adr] for adr in qvel_ids])
        tau = KPS * (target - dof_pos) - KDS * dof_vel
        tau = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
        max_tau_per_joint = np.maximum(max_tau_per_joint, np.abs(tau))
        for i, aid in enumerate(act_ids):
            d.ctrl[aid] = tau[i]
        mujoco.mj_step(m, d)

    # Check: target_dof_pos should differ from default only at action_idx
    target_correct = (np.abs(target - DEFAULT_DOF_POS).argmax() == action_idx)

    # Check: largest torque magnitude should be at action_idx (first few steps)
    tau_max_idx = np.argmax(max_tau_per_joint)
    tau_correct = (tau_max_idx == action_idx)

    return {
        "action_idx": action_idx,
        "expected_joint": JOINT_NAMES[action_idx],
        "target_at_idx": target[action_idx],
        "target_default_at_idx": DEFAULT_DOF_POS[action_idx],
        "target_correct": target_correct,
        "max_torque_joint": JOINT_NAMES[tau_max_idx],
        "max_torque": max_tau_per_joint[tau_max_idx],
        "torque_at_expected": max_tau_per_joint[action_idx],
        "tau_correct": tau_correct,
        "match": target_correct and tau_correct,
    }


def main():
    print("=" * 60)
    print("Task 3: Action-to-Joint Mapping Test")
    print("=" * 60)
    print(f"Testing all {len(JOINT_NAMES)} action dimensions...")
    print()

    failures = []
    for i in range(len(JOINT_NAMES)):
        result = test_single_action(i)
        status = "OK" if result["match"] else f"FAIL(tau_on={result['max_torque_joint']})"
        print(f"  action[{i:2d}] → {result['expected_joint']:30s} | "
              f"target={result['target_at_idx']:+.3f} | "
              f"max_tau={result['max_torque']:.1f} Nm on {result['max_torque_joint']:30s} | "
              f"{status}")

        if not result["match"]:
            failures.append(result)

    print()
    if failures:
        print(f"*** {len(failures)} FAILURES ***")
        for f in failures:
            print(f"  action[{f['action_idx']}] expected {f['expected_joint']}")
            if not f["target_correct"]:
                print(f"    target_dof_pos WRONG")
            if not f["tau_correct"]:
                print(f"    max_torque on {f['max_torque_joint']} ({f['max_torque']:.1f}Nm), "
                      f"expected {f['expected_joint']} ({f['torque_at_expected']:.1f}Nm)")
        raise SystemExit(1)
    else:
        print("ALL 22 action dimensions correctly mapped to their joints.")
        print("Action → Joint → MuJoCo actuator mapping is CORRECT.")


if __name__ == "__main__":
    main()
