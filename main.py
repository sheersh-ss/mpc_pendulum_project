import os
from experiments import run_grid_experiments
from plotting import load_result_files, plot_example_trajectory, plot_success_vs_horizon

def main():
    os.makedirs("results/figures", exist_ok=True)
    run_grid_experiments(output_dir="results/raw")

    results = load_result_files("results/raw")

    # pick one file as trajectory example
    example = sorted(results, key=lambda x: (x["summary"]["horizon"], x["summary"]["r_u"]))[-1]
    plot_example_trajectory(example, "results/figures/example_trajectory.pdf")
    plot_success_vs_horizon(results, "results/figures/success_vs_horizon.pdf")

if __name__ == "__main__":
    main()