# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from alchemlyb.estimators import MBAR
from util.real_data_handler import RealDataHandler
from alchemlyb.visualisation import plot_mbar_overlap_matrix
from util.calc_partial_overlap import calc_partial_overlap_matrix

fig_path = Path("./figures")
fig_path.mkdir(parents=True, exist_ok=True)

system_output_path = Path("./real_data/ejm_31")
sim_type = "abfe"
org_u_nks = RealDataHandler.get_files_from_directory(directory=system_output_path / sim_type,
                                                     energy_file_name="prod_npt.csv").u_nks
STATE_NUM = len(org_u_nks)
u_nks = pd.concat([u_nk for u_nk in org_u_nks])
mbar_estimator = MBAR(method="L-BFGS-B").fit(u_nks)

plot_data = {
    "x": [],
    "estimate": [],
    "real": [],
}
f_k = [0.0]
for i in range(len(mbar_estimator.delta_f_) - 1):
    f_k.append(mbar_estimator.delta_f_.iloc[i, i+1] + f_k[i])
    print(f"{i} -> {i+1}: {mbar_estimator.delta_f_.iloc[i, i+1]}, f_k: {f_k[i+1]}")
ax = plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
ax.figure.savefig(fig_path / "overlap_matrix.png")
org_overlap_matrix = mbar_estimator.overlap_matrix


partial_overlap_matrix = calc_partial_overlap_matrix(mbar_estimator)

# estimate_start_lambda_idx = 15
# estimate_end_lambda_idx = 36
# estimate_start_lambda_idx = 20
# estimate_end_lambda_idx = 41
estimate_start_lambda_idx = 25
estimate_end_lambda_idx = 46
i = estimate_start_lambda_idx
for j in range(estimate_start_lambda_idx+1, estimate_end_lambda_idx):
    test_u_nks = []
    remove_lambda_list = list(range(estimate_start_lambda_idx + 1, j)) + [2, 5, 7, 50, 53, 57, 60]
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
    test_mbar_estimator = MBAR(method="L-BFGS-B").fit(test_u_nks)
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
          f"\nestimate overlap: {org_overlap_matrix[estimate_start_lambda_idx, j] * C}, "
          f"\nreal overlap {test_start_lambda_idx}->{test_start_lambda_idx + 1}: "
          f"{test_overlap_matrix[test_start_lambda_idx, test_start_lambda_idx + 1]}")
    plot_data["x"].append(f"{estimate_start_lambda_idx}->{j}")
    plot_data["estimate"].append(org_overlap_matrix[estimate_start_lambda_idx, j] * C)
    plot_data["real"].append(test_overlap_matrix[test_start_lambda_idx, test_start_lambda_idx + 1])

plt.close("all")
h1 = plt.plot(range(len(plot_data["estimate"])), plot_data["estimate"], color="red", marker="o")
h2 = plt.plot(range(len(plot_data["real"])), plot_data["real"], color="blue", marker="^")
plt.legend(handles=[h1[0], h2[0]], labels=["estimate", "real"], loc="best")
plt.xticks(range(len(plot_data["x"])), plot_data["x"], rotation=45)
plt.show()
