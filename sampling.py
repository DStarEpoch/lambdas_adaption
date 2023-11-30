import time
import numpy as np
import pandas as pd
from openmm import Vec3
from typing import List
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
from alchemlyb.estimators import MBAR as _MBAR
from alchemlyb.visualisation import plot_mbar_overlap_matrix
from state_obj import StateObj


class ParaForwardContext:
    obj_properties: dict
    n_steps: int
    sample_steps: int
    idx: int

    def __init__(self, idx: int, obj_properties: dict, n_steps: int, sample_steps: int):
        self.idx = idx
        self.obj_properties = obj_properties
        self.n_steps = n_steps
        self.sample_steps = sample_steps


def parallel_forward(context: ParaForwardContext):
    obj = StateObj.createFromProperties(context.obj_properties)
    obj.forward(context.n_steps, context.sample_steps)
    return {"idx": context.idx, "obj": obj.properties}


N_PROCESSES = 5
SAMPLE_STEPS = 1
EXCHANGE_STEPS = SAMPLE_STEPS * 500
N_STEPS = 10000
TEMPERATURE = 500   # K

color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
force_const_list = np.asarray([4, 3, 3, 2, 2, 2, 2, 1, 1.8, 1.6, 1.6, 1.8, 1, 2, 2, 2, 2, 3, 3, 4]) * 2
x0_list = np.asarray(range(20)) * 0.8
p0_list = np.asarray(range(20)) * 100.0
sample_objs: List[StateObj] = [StateObj(TEMPERATURE, fc, x0, p0) for fc, x0, p0 in
                               zip(force_const_list, x0_list, p0_list)]
STATE_NUM = len(sample_objs)

exchange_flag = True
complete_steps = 0
time_start = time.time()
for i in range(N_STEPS // EXCHANGE_STEPS):
    if N_PROCESSES > 1:
        params = [ParaForwardContext(idx, obj.properties, EXCHANGE_STEPS, SAMPLE_STEPS)
                  for idx, obj in enumerate(sample_objs)]
        with Pool(N_PROCESSES) as pool:
            result = list(pool.map(parallel_forward, params))
            result.sort(key=lambda x: x["idx"])
            sample_objs = [StateObj.createFromProperties(x["obj"]) for x in result]
    else:
        for obj in sample_objs:
            obj.forward(EXCHANGE_STEPS, SAMPLE_STEPS)
    complete_steps += EXCHANGE_STEPS

    exchange_flag = exchange_flag ^ True
    obj_idx = list(range(len(sample_objs)))
    exchange_status = ""
    for k in range(STATE_NUM):
        if (exchange_flag and k % 2 == 0) or (not exchange_flag and k % 2 == 1):
            flag = sample_objs[k].tryExchange(sample_objs[(k + 1) % STATE_NUM])
            exchange_status += f"{k}x" if flag else f"{k} "
        else:
            exchange_status += f"{k} "
    print("step: {}, exchange:\n{}".format(complete_steps, exchange_status))
time_end = time.time()
print("time cost: {}s".format(time_end - time_start))

for o in range(STATE_NUM):
    x_data = [x[0].x for x in sample_objs[o].traj]
    plt.hist(x_data, bins=EXCHANGE_STEPS, color=color_list[o % len(color_list)], alpha=0.5)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
Path("./figures").mkdir(parents=True, exist_ok=True)
plt.savefig(Path(f"./figures/hist_ex_{EXCHANGE_STEPS}_samples_{N_STEPS}_states_{STATE_NUM}.png"))

org_u_nks = []
for o in range(STATE_NUM):
    print(f"collect u_nks state: {o}")
    u_k = list()
    for ro in range(STATE_NUM):
        p_on_j = sample_objs[o].calcRelativePotentialOnTraj(sample_objs[ro].traj, trunk_step=EXCHANGE_STEPS)
        u_k.append(p_on_j)
    u_k = pd.DataFrame(np.asarray(u_k).T, columns=[f"state_{i}" for i in range(STATE_NUM)])
    # rename window list
    u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
    # add a column to df for using groupby
    u_k['lambda'] = f"lambda_{o}"
    u_k['window'] = f"{o}"
    # set lambda index for later groupby
    u_k = u_k.set_index(['lambda', 'window'])
    org_u_nks.append(u_k)

org_u_nks = pd.concat([u_nk for u_nk in org_u_nks])
mbar_estimator = _MBAR(method="L-BFGS-B").fit(org_u_nks)
f_k = [0.0]
for i in range(len(mbar_estimator.delta_f_) - 1):
    f_k.append(mbar_estimator.delta_f_.iloc[i, i+1] + f_k[i])
    print(f"{i} -> {i+1}: {mbar_estimator.delta_f_.iloc[i, i+1]}, f_k: {f_k[i+1]}")
ax = plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
ax.figure.savefig(Path("./figures/overlap_matrix.png"))
org_overlap_matrix = mbar_estimator.overlap_matrix

balance_density_matrix = np.zeros((STATE_NUM, STATE_NUM))
for i in range(STATE_NUM):
    for j in range(STATE_NUM):
        u_ij = sample_objs[i].calcRelativePotentialOnTraj([Vec3(x0_list[j], 0, 0), ])[0]
        balance_density_matrix[i, j] = np.exp(f_k[i]-u_ij)
# print("balance density matrix: \n{}".format(balance_density_matrix))

estimate_start_lambda_idx = 6
estimate_end_lambda_idx = 16
i = estimate_start_lambda_idx
for j in range(estimate_start_lambda_idx+1, estimate_end_lambda_idx):
    test_u_nks = []
    # remain_lambda_list = [l for l in range(STATE_NUM) if l not in
    #                       list(range(estimate_start_lambda_idx+1, j)) + [1, 3, 18]]
    remain_lambda_list = [l for l in range(STATE_NUM) if l not in list(range(estimate_start_lambda_idx+1, j))]
    lc = -1
    for o in remain_lambda_list:
        lc += 1
        u_k = list()
        for ro in remain_lambda_list:
            p_on_j = sample_objs[o].calcRelativePotentialOnTraj(sample_objs[ro].traj, trunk_step=EXCHANGE_STEPS)
            u_k.append(p_on_j)
        u_k = pd.DataFrame(np.asarray(u_k).T, columns=[f"state_{i}" for i in range(len(remain_lambda_list))])
        # rename window list
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
    test_mbar_estimator = _MBAR(method="L-BFGS-B").fit(test_u_nks)
    test_overlap_matrix = test_mbar_estimator.overlap_matrix
    ax = plot_mbar_overlap_matrix(test_mbar_estimator.overlap_matrix)
    ax.figure.savefig(Path(f"./figures/test_real_overlap_matrix_{j}.png"))

    C1 = sum([org_overlap_matrix[i, k] * org_overlap_matrix[j, k] /
              sum([org_overlap_matrix[l, k] for l in range(STATE_NUM)]) for k in range(STATE_NUM)])
    C2 = sum([org_overlap_matrix[i, k] * org_overlap_matrix[j, k] /
              sum([org_overlap_matrix[l, k] for l in remain_lambda_list]) for k in range(STATE_NUM)])
    # C1 = sum([balance_density_matrix[i, k] * balance_density_matrix[j, k] /
    #           sum([balance_density_matrix[l, k] for l in range(STATE_NUM)]) for k in range(STATE_NUM)])
    # C2 = sum([balance_density_matrix[i, k] * balance_density_matrix[j, k] /
    #           sum([balance_density_matrix[l, k] for l in remain_lambda_list]) for k in range(STATE_NUM)])
    print("\nC1: {}, C2: {}".format(C1, C2))
    C = np.exp(1 - C2 / C1)
    print(f"{estimate_start_lambda_idx}->{j}, C: {C}, "
          f"\nestimate overlap: {org_overlap_matrix[estimate_start_lambda_idx, j] * C,}, "
          f"\nreal overlap {test_start_lambda_idx}->{test_start_lambda_idx+1}: "
          f"{test_overlap_matrix[test_start_lambda_idx, test_start_lambda_idx+1]}")

