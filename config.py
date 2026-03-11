from dataclasses import dataclass

@dataclass
class PendulumConfig:
    g: float = 9.81
    m: float = 1.0
    l: float = 1.0
    b: float = 0.05
    dt: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0

@dataclass
class MPCConfig:
    horizon: int = 20
    num_candidates: int = 256
    q_theta: float = 1.0
    q_omega: float = 0.1
    r_u: float = 0.01
    episode_length: int = 200

HORIZONS = [10, 20, 40]
CONTROL_PENALTIES = [0.001, 0.01, 0.1]
SEEDS = list(range(10))