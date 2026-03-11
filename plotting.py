import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_result_files(results_dir="results/raw"):
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    data = []

    for fname in files:
        with open(os.path.join(results_dir, fname), "r") as f:
            data.append(json.load(f))

    return data

def plot_example_trajectory(result_json, save_path):
    states = np.array(result_json["example_states"])
    actions = np.array(result_json["example_actions"])
    t_states = np.arange(len(states))
    t_actions = np.arange(len(actions))

    plt.figure(figsize=(8, 3))
    plt.plot(t_states, states[:, 0])
    plt.xlabel("Time step")
    plt.ylabel("Angle")
    plt.tight_layout()
    plt.savefig(save_path.replace(".pdf", "_angle.pdf"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(t_states, states[:, 1])
    plt.xlabel("Time step")
    plt.ylabel("Angular velocity")
    plt.tight_layout()
    plt.savefig(save_path.replace(".pdf", "_omega.pdf"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(t_actions, actions)
    plt.xlabel("Time step")
    plt.ylabel("Torque")
    plt.tight_layout()
    plt.savefig(save_path.replace(".pdf", "_action.pdf"))
    plt.close()

def plot_success_vs_horizon(results, save_path):
    penalties = sorted(set(r["summary"]["r_u"] for r in results))

    plt.figure(figsize=(6, 4))
    for penalty in penalties:
        xs = []
        ys = []
        filtered = [r for r in results if r["summary"]["r_u"] == penalty]
        filtered = sorted(filtered, key=lambda x: x["summary"]["horizon"])

        for r in filtered:
            xs.append(r["summary"]["horizon"])
            ys.append(r["summary"]["success_mean"])

        plt.plot(xs, ys, marker="o", label=f"R={penalty}")

    plt.xlabel("Planning horizon")
    plt.ylabel("Success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()