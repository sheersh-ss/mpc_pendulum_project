import numpy as np
from dynamics import rollout_dynamics, angle_error_to_upright

def trajectory_cost(states, actions, mpc_cfg):
    theta = states[:-1, 0]
    omega = states[:-1, 1]

    theta_err = angle_error_to_upright(theta)

    state_cost = (
        mpc_cfg.q_theta * theta_err**2
        + mpc_cfg.q_omega * omega**2
    )
    action_cost = mpc_cfg.r_u * actions**2

    return np.sum(state_cost + action_cost)

def sample_action_sequences(num_candidates, horizon, max_torque, rng):
    return rng.uniform(
        low=-max_torque,
        high=max_torque,
        size=(num_candidates, horizon),
    )

def mpc_action(state, dyn_cfg, mpc_cfg, rng):
    candidates = sample_action_sequences(
        num_candidates=mpc_cfg.num_candidates,
        horizon=mpc_cfg.horizon,
        max_torque=dyn_cfg.max_torque,
        rng=rng,
    )

    best_cost = np.inf
    best_sequence = None

    for seq in candidates:
        pred_states = rollout_dynamics(state, seq, dyn_cfg)
        cost = trajectory_cost(pred_states, seq, mpc_cfg)

        if cost < best_cost:
            best_cost = cost
            best_sequence = seq

    return best_sequence[0], best_sequence, best_cost