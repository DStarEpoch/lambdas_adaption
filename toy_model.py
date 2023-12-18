# -*- coding:utf-8 -*-
import tqdm
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
from alchemlyb.estimators import MBAR
from util.ising_model import IsingModel
from alchemlyb.visualisation import plot_mbar_overlap_matrix
from util.calc_partial_overlap import calc_partial_overlap_matrix


STATE_NUM = 40
BETAS = 1 / np.linspace(1.6, 2.4, STATE_NUM)
# BETAS = 1 / np.linspace(1.53, 3.28, STATE_NUM)
RELAX_STEPS = 1000
N_STEPS = 2000
SAMPLE_STEPS = 50
N = 40  # N x N grid for the Ising model
PROCESSES = 8
figure_path = Path("./figures")
figure_path.mkdir(parents=True, exist_ok=True)


class ParaForwardContext:
    obj: IsingModel
    idx: int
    forward_steps: int
    sample_steps: int

    def __init__(self, idx: int, obj: IsingModel, forward_steps: int, sample_steps: int):
        self.idx = idx
        self.obj = obj
        self.forward_steps = forward_steps
        self.sample_steps = sample_steps


def parallel_forward(context: ParaForwardContext):
    obj = context.obj
    E_list = list()
    config_list = list()
    for i in range(context.forward_steps):
        obj.mcmove()
        E_list.append(obj.dimless_energy)
        if i % context.sample_steps == context.sample_steps - 1:
            config_list.append(obj.config)
    return {"idx": context.idx, "obj": obj, "E_list": E_list, "config_list": config_list}


models = [IsingModel(N=N, beta=beta) for beta in BETAS]
# relax
params = [ParaForwardContext(idx, m, RELAX_STEPS, SAMPLE_STEPS) for idx, m in enumerate(models)]
if PROCESSES > 1:
    with Pool(PROCESSES) as pool:
        result = list(tqdm.tqdm(pool.imap(parallel_forward, params), total=len(params), desc="relaxing"))
        result.sort(key=lambda r: r["idx"])
        for x in result:
            models[x["idx"]] = x["obj"]
else:
    for param in tqdm.tqdm(params, total=len(params), desc="relaxing"):
        result = parallel_forward(param)
        models[result["idx"]] = result["obj"]

# sampling run
params = [ParaForwardContext(idx, m, N_STEPS, SAMPLE_STEPS) for idx, m in enumerate(models)]
E = [[] for _ in BETAS]
config_traj_list = [[] for _ in BETAS]
if PROCESSES > 1:
    with Pool(PROCESSES) as pool:
        result = list(tqdm.tqdm(pool.imap(parallel_forward, params), total=len(params), desc="sampling"))
        result.sort(key=lambda r: r["idx"])
        for x in result:
            E[x["idx"]] = x["E_list"]
            config_traj_list[x["idx"]] = x["config_list"]
            models[x["idx"]] = x["obj"]
else:
    for param in tqdm.tqdm(params, total=len(params), desc="sampling"):
        result = parallel_forward(param)
        E[result["idx"]] = result["E_list"]
        config_traj_list[result["idx"]] = result["config_list"]
        models[x["idx"]] = x["obj"]

color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
for model_idx in range(len(E)):
    plt.hist(E[model_idx], bins=50, alpha=0.5,
             color=color_list[model_idx % len(color_list)],
             label=f"beta_{model_idx}", density=True)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.savefig(figure_path / f"Ising_energy_distribution_{STATE_NUM}_{N_STEPS}_{N}.png")
# plt.show()


# collect u_nks for MBAR estimation
class ParaCollectContext:
    idx: int
    config_list: List[np.ndarray]
    model_list: List[IsingModel]

    def __init__(self, idx: int, config_list: list, model_list: list):
        self.idx = idx
        self.model_list = model_list
        self.config_list = config_list


def parallel_calc_u_nk(context: ParaCollectContext):
    u_k = list()
    for model in context.model_list:
        p_on_j = list()
        for cfg in context.config_list:
            p_on_j.append(model.dimlessEnergyOnConfig(cfg))
        u_k.append(p_on_j)
    u_k = pd.DataFrame(np.asarray(u_k).T, columns=[f"state_{i}" for i in range(len(context.model_list))])
    # rename window list
    u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
    # add a column to df for using groupby
    u_k['lambda'] = f"lambda_{context.idx}"
    u_k['window'] = f"{context.idx}"
    # set lambda index for later groupby
    u_k = u_k.set_index(['lambda', 'window'])
    return {"idx": context.idx, "u_k": u_k}


org_u_nks = []
params = [ParaCollectContext(idx, config_list, models) for idx, config_list in enumerate(config_traj_list)]
if PROCESSES > 1:
    with Pool(PROCESSES) as pool:
        result = list(tqdm.tqdm(pool.imap(parallel_calc_u_nk, params), total=len(params), desc="collect u_nks"))
        result.sort(key=lambda r: r["idx"])
        for x in result:
            org_u_nks.append(x["u_k"])
else:
    for param in tqdm.tqdm(params, total=len(params), desc="collect u_nks"):
        result = parallel_calc_u_nk(param)
        org_u_nks.append(result["u_k"])

mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in org_u_nks]))
org_overlap_matrix = mbar_estimator.overlap_matrix
f_k = [0.0]
for i in range(len(mbar_estimator.delta_f_) - 1):
    f_k.append(mbar_estimator.delta_f_.iloc[i, i+1] + f_k[i])
    print(f"{i} -> {i+1}: {mbar_estimator.delta_f_.iloc[i, i+1]}, f_k: {f_k[i+1]}")
ax = plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
ax.figure.savefig(figure_path / f"Ising_overlap_matrix_{STATE_NUM}_{N_STEPS}_{N}.png")
# plt.show()

plot_data = {
    "x": [],
    "estimate": [],
    "real": [],
}
partial_overlap_matrix = calc_partial_overlap_matrix(mbar_estimator)
estimate_start_lambda_idx = 10
estimate_end_lambda_idx = 30
i = estimate_start_lambda_idx
for j in range(estimate_start_lambda_idx+1, estimate_end_lambda_idx):
    test_u_nks = []
    remove_lambda_list = list(range(estimate_start_lambda_idx+1, j))
    # remain_lambda_list = [l for l in range(STATE_NUM) if l not in
    #                       list(range(estimate_start_lambda_idx+1, j)) + [1, 3, 18]]
    remain_lambda_list = [l for l in range(STATE_NUM) if l not in remove_lambda_list]
    lc = -1
    for o in remain_lambda_list:
        lc += 1
        u_k = org_u_nks[o].drop(columns=[str(l) for l in remove_lambda_list])
        u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
        # add a column to df for using groupby
        u_k['lambda'] = f"lambda_{lc}"
        u_k['window'] = f"{lc}"
        # set lambda index for later groupby
        u_k = u_k.set_index(['lambda', 'window'])
        test_u_nks.append(u_k)
    test_start_lambda_idx = -1
    for i in range(estimate_start_lambda_idx+1):
        if i in remain_lambda_list:
            test_start_lambda_idx += 1
    test_u_nks = pd.concat([u_nk for u_nk in test_u_nks])
    test_mbar_estimator = MBAR(method="L-BFGS-B").fit(test_u_nks)
    test_overlap_matrix = test_mbar_estimator.overlap_matrix

    # C1 = sum([org_overlap_matrix[i, k] * org_overlap_matrix[j, k] /
    #           sum([org_overlap_matrix[l, k] for l in range(STATE_NUM)]) for k in range(STATE_NUM)])
    # C2 = sum([org_overlap_matrix[i, k] * org_overlap_matrix[j, k] /
    #           sum([org_overlap_matrix[l, k] for l in remain_lambda_list]) for k in range(STATE_NUM)])
    C1 = sum([partial_overlap_matrix[k][i][j] for k in remain_lambda_list])
    C2 = sum([partial_overlap_matrix[k][i][j] for k in range(STATE_NUM)])
    print("\nC1: {}, C2: {}".format(C1, C2))
    C = C2 / C1
    # C = np.exp(1 - C2 / C1)
    print(f"{estimate_start_lambda_idx}->{j}, C: {C}, "
          f"\nestimate overlap: {org_overlap_matrix[estimate_start_lambda_idx, j] * C}, "
          f"\nreal overlap {test_start_lambda_idx}->{test_start_lambda_idx+1}: "
          f"{test_overlap_matrix[test_start_lambda_idx, test_start_lambda_idx+1]}")
    plot_data["x"].append(f"{estimate_start_lambda_idx}->{j}")
    plot_data["estimate"].append(org_overlap_matrix[estimate_start_lambda_idx, j] * C)
    plot_data["real"].append(test_overlap_matrix[test_start_lambda_idx, test_start_lambda_idx + 1])

plt.close("all")
h1 = plt.plot(range(len(plot_data["estimate"])), plot_data["estimate"], color="red", marker="o")
h2 = plt.plot(range(len(plot_data["real"])), plot_data["real"], color="blue", marker="^")
plt.legend(handles=[h1[0], h2[0]], labels=["estimate", "real"], loc="best")
plt.xticks(range(len(plot_data["x"])), plot_data["x"], rotation=45)
plt.savefig(figure_path / f"Ising_estimate_{STATE_NUM}_{N_STEPS}_{N}.png")
# plt.show()
