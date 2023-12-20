# -*- coding:utf-8 -*-
import time
import numpy as np
from pathlib import Path
from ising import IsingModel


STATE_NUM = 50
BETAS = 1 / np.linspace(1.6, 2.4, STATE_NUM)
# BETAS = 1 / np.linspace(1.53, 3.28, STATE_NUM)
RELAX_STEPS = 20000
N_STEPS = 500000
SAMPLE_STEPS = 1000
N = 40  # N x N grid for the Ising model
PROCESSES = 5
figure_path = Path("./figures")
figure_path.mkdir(parents=True, exist_ok=True)

time_start = time.time()
m1 = IsingModel(N=N, beta=0.4)
energy_samples, spins_samples = m1.mcMove(RELAX_STEPS, SAMPLE_STEPS)
time_end = time.time()
print("relaxing time: ", time_end - time_start,
      f"energy_samples: {len(energy_samples)}, spins_samples: {len(spins_samples)}")
