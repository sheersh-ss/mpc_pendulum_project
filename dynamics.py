import numpy as np

def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angle_error_to_upright(theta: np.ndarray) -> np.ndarray:
    # upright is theta = 0
    return wrap_angle(theta)

def pendulum_step(state, action, cfg):
    theta, omega = state
    u = np.clip(action, -cfg.max_torque, cfg.max_torque)

    theta_ddot = (
        (3.0 * cfg.g / (2.0 * cfg.l)) * np.sin(theta)
        + (3.0 / (cfg.m * cfg.l**2)) * u
        - cfg.b * omega
    )

    omega_next = omega + cfg.dt * theta_ddot
    omega_next = np.clip(omega_next, -cfg.max_speed, cfg.max_speed)

    theta_next = theta + cfg.dt * omega_next
    theta_next = wrap_angle(theta_next)

    return np.array([theta_next, omega_next], dtype=np.float64)

def rollout_dynamics(initial_state, action_sequence, cfg):
    states = [np.array(initial_state, dtype=np.float64)]
    state = np.array(initial_state, dtype=np.float64)

    for u in action_sequence:
        state = pendulum_step(state, u, cfg)
        states.append(state.copy())

    return np.stack(states, axis=0)