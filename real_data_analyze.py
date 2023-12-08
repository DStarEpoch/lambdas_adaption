import numpy as np
from copy import copy
from pathlib import Path
import pandas as pd
from alchemlyb.estimators import MBAR as _MBAR
from util.real_data_handler import RealDataHandler
from alchemlyb.visualisation import plot_mbar_overlap_matrix
import matplotlib.pyplot as plt

fig_path = Path("./figures")
fig_path.mkdir(parents=True, exist_ok=True)

system_output_path = Path("./real_data/ejm_31")
sim_type = "abfe"
org_u_nks = RealDataHandler.get_files_from_directory(directory=system_output_path / sim_type,
                                                     energy_file_name="prod_npt.csv").u_nks
STATE_NUM = len(org_u_nks)
u_nks = pd.concat([u_nk for u_nk in org_u_nks])
mbar_estimator = _MBAR(method="L-BFGS-B").fit(u_nks)

f_k = [0.0]
for i in range(len(mbar_estimator.delta_f_) - 1):
    f_k.append(mbar_estimator.delta_f_.iloc[i, i+1] + f_k[i])
    print(f"{i} -> {i+1}: {mbar_estimator.delta_f_.iloc[i, i+1]}, f_k: {f_k[i+1]}")
ax = plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
ax.figure.savefig(fig_path / "overlap_matrix.png")
org_overlap_matrix = mbar_estimator.overlap_matrix


partial_overlap_matrix = list()
org_w_nk = mbar_estimator._mbar.W_nk
N_K = mbar_estimator._mbar.N_k
for k in range(STATE_NUM):
    tmp_w_nk = copy(org_w_nk)
    for i in range(tmp_w_nk.shape[0]):
        for j in range(tmp_w_nk.shape[1]):
            tmp_w_nk[i][j] *= org_w_nk[i][k]
    O_ijk = N_K * (org_w_nk.T @ tmp_w_nk)
    partial_overlap_matrix.append(O_ijk)
    print("build partial_overlap_matrix", k)

estimate_start_lambda_idx = 15
estimate_end_lambda_idx = 36
i = estimate_start_lambda_idx
for j in range(estimate_start_lambda_idx+1, estimate_end_lambda_idx):
    test_u_nks = []
    remove_lambda_list = list(range(estimate_start_lambda_idx + 1, j)) + [2, 5, 7, 40, 45, 50, 53, 57, 60]
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
    for i in range(estimate_start_lambda_idx + 1):
        if i in remain_lambda_list:
            test_start_lambda_idx += 1
    test_u_nks = pd.concat([u_nk for u_nk in test_u_nks])
    test_mbar_estimator = _MBAR(method="L-BFGS-B").fit(test_u_nks)
    test_overlap_matrix = test_mbar_estimator.overlap_matrix
    ax = plot_mbar_overlap_matrix(test_mbar_estimator.overlap_matrix)
    ax.figure.savefig(fig_path / f"test_real_overlap_matrix_{j}.png")

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
          f"\nestimate overlap: {org_overlap_matrix[estimate_start_lambda_idx, j] * C,}, "
          f"\nreal overlap {test_start_lambda_idx}->{test_start_lambda_idx + 1}: "
          f"{test_overlap_matrix[test_start_lambda_idx, test_start_lambda_idx + 1]}")
