# -*- coding:utf-8 -*-
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alchemlyb.estimators import MBAR
from util.ising_model import IsingModel
from alchemlyb.visualisation import plot_mbar_overlap_matrix
from util.calc_partial_overlap import calc_partial_overlap_matrix


state_num = 40
betas = 1 / np.linspace(1.8, 2.0, state_num)
# betas = 1 / np.linspace(1.53, 3.28, state_num)
N_STEPS = 2000
EXCHANGE_STEPS = 20
SAMPLE_STEPS = 10
N = 40
models = [IsingModel(N=N, beta=beta) for beta in betas]
E = [[] for _ in betas]
config_traj_list = [[] for _ in betas]
exchange_flag = True
for step in tqdm.tqdm(range(N_STEPS), desc="sampling"):
    for model_idx in range(len(models)):
        model = models[model_idx]
        model.mcmove()
        E[model_idx].append(model.dimless_energy)

    if step % SAMPLE_STEPS == (SAMPLE_STEPS - 1):
        for model_idx in range(len(models)):
            config_traj_list[model_idx].append(models[model_idx].config)
    # if step % EXCHANGE_STEPS == 0:
    #     for model_idx in range(len(models)):
    #         config_traj_list[model_idx].append(models[model_idx].config)
    #     exchange_flag = exchange_flag ^ True
    #     exchange_status = ""
    #     for k in range(state_num):
    #         if (exchange_flag and k % 2 == 0) or (not exchange_flag and k % 2 == 1):
    #             flag = models[k].swap(models[(k + 1) % state_num])
    #             exchange_status += f"{k}x" if flag else f"{k} "
    #         else:
    #             exchange_status += f"{k} "
    #     print(f"exchange:\n{exchange_status}")

color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
for model_idx in range(len(models)):
    plt.hist(E[model_idx], bins=50, alpha=0.5,
             color=color_list[model_idx % len(color_list)],
             label=f"beta_{model_idx}", density=True)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.show()

org_u_nks = []
for o in range(state_num):
    print(f"collect u_nks state: {o}")
    u_k = list()
    for ro in range(state_num):
        p_on_j = list()
        for cfg in config_traj_list[ro]:
            p_on_j.append(models[o].dimlessEnergyOnConfig(cfg))
        u_k.append(p_on_j)
    u_k = pd.DataFrame(np.asarray(u_k).T, columns=[f"state_{i}" for i in range(state_num)])
    # rename window list
    u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
    # add a column to df for using groupby
    u_k['lambda'] = f"lambda_{o}"
    u_k['window'] = f"{o}"
    # set lambda index for later groupby
    u_k = u_k.set_index(['lambda', 'window'])
    org_u_nks.append(u_k)

mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in org_u_nks]))
org_overlap_matrix = mbar_estimator.overlap_matrix
f_k = [0.0]
for i in range(len(mbar_estimator.delta_f_) - 1):
    f_k.append(mbar_estimator.delta_f_.iloc[i, i+1] + f_k[i])
    print(f"{i} -> {i+1}: {mbar_estimator.delta_f_.iloc[i, i+1]}, f_k: {f_k[i+1]}")
ax = plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
plt.show()



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
    remain_lambda_list = [l for l in range(state_num) if l not in remove_lambda_list]
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
    C2 = sum([partial_overlap_matrix[k][i][j] for k in range(state_num)])
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
plt.show()
