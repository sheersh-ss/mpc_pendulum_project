import numpy as np
from dynamics import angle_error_to_upright

def compute_episode_metrics(states, actions, success_angle=0.2, success_omega=0.5, final_window=20):
    theta = states[:, 0]
    omega = states[:, 1]
    theta_err = np.abs(angle_error_to_upright(theta))

    final_angle_error = float(theta_err[-1])
    control_effort = float(np.sum(actions**2))

    start = max(0, len(theta) - final_window)
    stable_window = (
        np.all(theta_err[start:] < success_angle) and
        np.all(np.abs(omega[start:]) < success_omega)
    )
    success = int(stable_window)

    swingup_time = None
    for t in range(len(theta)):
        if theta_err[t] < success_angle and abs(omega[t]) < success_omega:
            swingup_time = t
            break

    return {
        "success": success,
        "final_angle_error": final_angle_error,
        "control_effort": control_effort,
        "swingup_time": swingup_time if swingup_time is not None else -1,
    }