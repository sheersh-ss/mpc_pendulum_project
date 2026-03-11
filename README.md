# MPC Pendulum Project

This project studies **Model Predictive Control (MPC)** on the **pendulum swing-up** task. The controller uses a known dynamics model and a random-shooting planner to choose torque actions that swing the pendulum to the upright position and stabilize it.

The main research question is:

**How do planning horizon and control penalty affect the performance of MPC on the pendulum swing-up task?**

## Project structure

```text
mpc_pendulum_project/
├── config.py
├── dynamics.py
├── experiments.py
├── main.py
├── metrics.py
├── mpc.py
├── plotting.py
├── utils.py
├── requirements.txt
├── README.md
├── notebooks/
│   ├── quick_analysis.ipynb
├── results/
│   ├── raw/
│   ├── figures/
│   └── tables/
└── report/
```

