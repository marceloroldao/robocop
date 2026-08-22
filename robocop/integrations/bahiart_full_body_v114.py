from __future__ import annotations

import numpy as np

# BahiaRT actuator names and perceived HJ sensor keys are different namespaces.
# The audit confirmed a one-to-one semantic/order mapping across all 23 joints.
MOTOR_NAMES = (
    'he1','he2','lae1','lae2','lae3','lae4','rae1','rae2','rae3','rae4','te1',
    'lle1','lle2','lle3','lle4','lle5','lle6','rle1','rle2','rle3','rle4','rle5','rle6'
)
SENSOR_KEYS = (
    'q_hj1','q_hj2','q_laj1','q_laj2','q_laj3','q_laj4','q_raj1','q_raj2','q_raj3','q_raj4','q_tj1',
    'q_llj1','q_llj2','q_llj3','q_llj4','q_llj5','q_llj6','q_rlj1','q_rlj2','q_rlj3','q_rlj4','q_rlj5','q_rlj6'
)


def full_body_sensor_state(agent) -> dict:
    robot = agent.robot
    world = agent.world

    runtime_motors = tuple(robot.ROBOT_MOTORS)
    if runtime_motors != MOTOR_NAMES:
        raise RuntimeError(f'Unexpected BahiaRT ROBOT_MOTORS schema: {runtime_motors!r}')

    pos = robot.motor_positions
    spd = robot.motor_speeds
    missing_pos = [k for k in SENSOR_KEYS if k not in pos]
    missing_spd = [k for k in SENSOR_KEYS if k not in spd]
    if missing_pos or missing_spd:
        raise RuntimeError(f'Missing BahiaRT joint sensors: positions={missing_pos} speeds={missing_spd}')

    global_position = np.asarray(world.global_position, dtype=float).reshape(-1)[:3]
    quat = np.asarray(robot.global_orientation_quat, dtype=float).reshape(-1)[:4]
    euler = np.asarray(robot.global_orientation_euler, dtype=float).reshape(-1)[:3]
    gyro = np.asarray(robot.gyroscope, dtype=float).reshape(-1)[:3]
    accel = np.asarray(robot.accelerometer, dtype=float).reshape(-1)[:3]
    joint_position = np.asarray([float(pos[k]) for k in SENSOR_KEYS], dtype=float)
    joint_speed = np.asarray([float(spd[k]) for k in SENSOR_KEYS], dtype=float)

    vector = np.concatenate([
        global_position, quat, euler, gyro, accel, joint_position, joint_speed
    ]).astype(float)

    names = (
        ['global_x','global_y','global_z']
        + ['quat_x','quat_y','quat_z','quat_w']
        + ['roll_deg','pitch_deg','yaw_deg']
        + ['gyro_x_deg_s','gyro_y_deg_s','gyro_z_deg_s']
        + ['accel_x_m_s2','accel_y_m_s2','accel_z_m_s2']
        + [f'joint_pos_deg:{m}' for m in MOTOR_NAMES]
        + [f'joint_speed_deg_s:{m}' for m in MOTOR_NAMES]
    )
    if len(names) != vector.size:
        raise RuntimeError('V11.4 full-body schema/vector length mismatch')

    return {
        'version': 'v11-full-body-2',
        'names': names,
        'vector': vector.tolist(),
        'groups': {
            'global_position': global_position.tolist(),
            'orientation_quat': quat.tolist(),
            'orientation_euler_deg': euler.tolist(),
            'gyroscope_deg_s': gyro.tolist(),
            'accelerometer_m_s2': accel.tolist(),
            'joint_position_deg': joint_position.tolist(),
            'joint_speed_deg_s': joint_speed.tolist(),
        },
        'motor_names': list(MOTOR_NAMES),
        'joint_sensor_keys': list(SENSOR_KEYS),
    }
