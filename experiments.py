import os
import json
import numpy as np

from config import PendulumConfig, MPCConfig, HORIZONS, CONTROL_PENALTIES, SEEDS
from dynamics import pendulum_step
from mpc import mpc_action
from metrics import compute_episode_metrics

def run_episode(dyn_cfg, mpc_cfg, seed=0, initial_state=None):
    rng = np.random.default_rng(seed)

    if initial_state is None:
        # downward position near pi with small perturbation
        initial_state = np.array([np.pi + rng.normal(0, 0.05), rng.normal(0, 0.05)])

    state = initial_state.copy()
    states = [state.copy()]
    actions = []
    planning_costs = []

    for _ in range(mpc_cfg.episode_length):
        action, _, pred_cost = mpc_action(state, dyn_cfg, mpc_cfg, rng)
        state = pendulum_step(state, action, dyn_cfg)

        actions.append(action)
        states.append(state.copy())
        planning_costs.append(pred_cost)

    states = np.stack(states, axis=0)
    actions = np.array(actions, dtype=np.float64)
    planning_costs = np.array(planning_costs, dtype=np.float64)

    metrics = compute_episode_metrics(states, actions)
    metrics["mean_planning_cost"] = float(np.mean(planning_costs))

    return {
        "states": states,
        "actions": actions,
        "planning_costs": planning_costs,
        "metrics": metrics,
    }

def summarize_results(records):
    summary = {}
    keys = ["success", "final_angle_error", "control_effort", "swingup_time", "mean_planning_cost"]

    for key in keys:
        values = [r["metrics"][key] for r in records]
        valid = [v for v in values if v != -1] if key == "swingup_time" else values

        summary[f"{key}_mean"] = float(np.mean(valid)) if len(valid) > 0 else -1.0
        summary[f"{key}_std"] = float(np.std(valid)) if len(valid) > 0 else -1.0

    return summary

def run_grid_experiments(output_dir="results/raw"):
    os.makedirs(output_dir, exist_ok=True)

    dyn_cfg = PendulumConfig()
    all_summaries = []

    for horizon in HORIZONS:
        for r_u in CONTROL_PENALTIES:
            mpc_cfg = MPCConfig(horizon=horizon, r_u=r_u)
            records = []

            for seed in SEEDS:
                result = run_episode(dyn_cfg, mpc_cfg, seed=seed)
                records.append(result)

            summary = summarize_results(records)
            summary["horizon"] = horizon
            summary["r_u"] = r_u
            all_summaries.append(summary)

            save_path = os.path.join(output_dir, f"h{horizon}_r{r_u}.json")
            serializable = {
                "summary": summary,
                "per_seed_metrics": [r["metrics"] for r in records],
                # store one example trajectory only to keep files smaller
                "example_states": records[0]["states"].tolist(),
                "example_actions": records[0]["actions"].tolist(),
            }
            with open(save_path, "w") as f:
                json.dump(serializable, f, indent=2)

    return all_summaries