# -*- coding:utf-8 -*-
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from alchemlyb.estimators import MBAR
from util.real_data_handler import RealDataHandler
from alchemlyb.visualisation import plot_mbar_overlap_matrix

fig_path = Path("./figures")
fig_path.mkdir(parents=True, exist_ok=True)

system_output_path = Path("./real_data/ejm_31")
sim_type = "abfe"
org_u_nks = RealDataHandler.get_files_from_directory(directory=system_output_path / sim_type,
                                                     energy_file_name="prod_npt.csv").u_nks
org_mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in org_u_nks]))
org_overlap_matrix = org_mbar_estimator.overlap_matrix
f_k = [0.0]
for i in range(len(org_mbar_estimator.delta_f_) - 1):
    f_k.append(org_mbar_estimator.delta_f_.iloc[i, i + 1] + f_k[i])
    print(f"{i} -> {i + 1}: {org_mbar_estimator.delta_f_.iloc[i, i + 1]}, f_k: {f_k[i + 1]}")


estimate_start_lambda_idx = 15
for interval in range(1, 10):
    neighbour_lambda_idx = estimate_start_lambda_idx + interval + 1
    insert_lambda_idx = round((estimate_start_lambda_idx + neighbour_lambda_idx) / 2)

    # real overlap estimation
    remove_lambda_idx = list(range(estimate_start_lambda_idx+1, estimate_start_lambda_idx+interval+1))
    remove_lambda_idx.remove(insert_lambda_idx)
    remain_lambda_list = sorted([l for l in range(len(org_u_nks)) if l not in remove_lambda_idx])
    lc = -1
    real_u_nks = []
    for o in remain_lambda_list:
        lc += 1
        u_k = org_u_nks[o].drop(columns=[str(l) for l in remove_lambda_idx])
        u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
        # add a column to df for using groupby
        u_k['lambda'] = f"lambda_{lc}"
        u_k['window'] = f"{lc}"
        # set lambda index for later groupby
        u_k = u_k.set_index(['lambda', 'window'])
        real_u_nks.append(u_k)
    test_start_lambda_idx = -1
    for i in range(estimate_start_lambda_idx + 1):
        if i in remain_lambda_list:
            test_start_lambda_idx += 1
    real_u_nks = pd.concat([u_nk for u_nk in real_u_nks])
    real_mbar_estimator = MBAR(method="L-BFGS-B").fit(real_u_nks)
    real_overlap_matrix = real_mbar_estimator.overlap_matrix

    # linear interpolation potential energy + boltzmann pick sampling, overlap estimation
    ratio = (insert_lambda_idx - estimate_start_lambda_idx) / (neighbour_lambda_idx - estimate_start_lambda_idx)
    remove_lambda_idx = list(range(estimate_start_lambda_idx + 1, estimate_start_lambda_idx + interval + 1))
    remain_lambda_list = sorted([l for l in range(len(org_u_nks)) if l not in remove_lambda_idx])
    samples_per_lambda = len(org_u_nks[0].iloc[:, 0])
    all_u_samples = dict()
    exp_f_insert = 0.0
    for o in remain_lambda_list:
        u = np.asarray(org_u_nks[o].iloc[:, estimate_start_lambda_idx]) * (1 - ratio) + \
            np.asarray(org_u_nks[o].iloc[:, neighbour_lambda_idx]) * ratio
        base_u = sum([np.exp(f_k[k] - np.asarray(org_u_nks[k].iloc[:, o])) for k in remain_lambda_list])
        exp_f_insert += sum(np.exp(-u) / base_u) / samples_per_lambda
        all_u_samples[o] = u
    f_insert = -np.log(exp_f_insert)
    print(f"boltzmann pick sampling estimate f_insert at {insert_lambda_idx}: {f_insert}")
    tag_list = []
    w_list = []
    for l_idx in all_u_samples:
        u = all_u_samples[l_idx]
        for u_idx in range(len(u)):
            base_u = sum([np.exp(f_k[k] - np.asarray(org_u_nks[l_idx].iloc[:, k][u_idx])) for k in remain_lambda_list])
            w_list.append(np.exp(f_insert-u[u_idx]) / (base_u * samples_per_lambda))
            tag_list.append((l_idx, u_idx))
        # print(f"boltzmann pick sampling {l_idx} cur_w: {sum(w_list)}")

    pick_tag_list = random.choices(tag_list, weights=w_list, k=samples_per_lambda)
    print("pick_tag_list: ", Counter([t for t, _ in pick_tag_list]))
    bp_u_nks = []
    lc = -1
    remain_lambda_list = sorted(remain_lambda_list + [insert_lambda_idx])
    for o in remain_lambda_list:
        lc += 1
        if o == insert_lambda_idx:
            u_k = pd.DataFrame()
            for o2 in remain_lambda_list:
                if o2 != insert_lambda_idx:
                    u_k[str(o2)] = [org_u_nks[t][str(o2)][i] for t, i in pick_tag_list]
                else:
                    u_k[str(o2)] = [org_u_nks[t][str(estimate_start_lambda_idx)][i] * (1 - ratio) +
                                    org_u_nks[t][str(neighbour_lambda_idx)][i] * ratio for t, i in pick_tag_list]
            u_k = u_k[sorted(u_k.columns, key=lambda x: float(x))]
        else:
            u_k = org_u_nks[o].drop(columns=[str(l) for l in remove_lambda_idx])
            u_k[str(insert_lambda_idx)] = all_u_samples[o]
            u_k = u_k[sorted(u_k.columns, key=lambda x: float(x))]
        u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
        # add a column to df for using groupby
        u_k['lambda'] = f"lambda_{lc}"
        u_k['window'] = f"{lc}"
        # set lambda index for later groupby
        u_k = u_k.set_index(['lambda', 'window'])
        bp_u_nks.append(u_k)
    bp_u_nks = pd.concat([u_nk for u_nk in bp_u_nks])
    bp_mbar_estimator = MBAR(method="L-BFGS-B").fit(bp_u_nks)
    bp_overlap_matrix = bp_mbar_estimator.overlap_matrix
    print(f"{estimate_start_lambda_idx}->{insert_lambda_idx}, "
          f"between [{estimate_start_lambda_idx}, {neighbour_lambda_idx}] \nreal overlap: "
          f"{real_overlap_matrix[estimate_start_lambda_idx, estimate_start_lambda_idx + 1]} \nestimate overlap: "
          f"{bp_overlap_matrix[estimate_start_lambda_idx, estimate_start_lambda_idx + 1]}\n")
    ax = plot_mbar_overlap_matrix(real_overlap_matrix)
    ax.figure.savefig(fig_path / f"real_overlap_matrix_{interval}.png")
    ax = plot_mbar_overlap_matrix(bp_overlap_matrix)
    ax.figure.savefig(fig_path / f"estimate_overlap_matrix_{interval}.png")
