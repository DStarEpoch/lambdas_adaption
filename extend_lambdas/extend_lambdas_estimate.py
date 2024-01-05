# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from typing import List


def FrobeniusNorm(m1: np.matrix, m2: np.matrix):
    return np.sqrt(np.trace(np.dot(m1, m2.T)))


def selectFromUNKS(u_nks: List[pd.DataFrame], select_lambdas_idx: List[int]) -> List[pd.DataFrame]:
    remove_lambdas_idx = [l for l in range(len(u_nks)) if l not in select_lambdas_idx]
    ret_u_nks = []
    lc = -1
    for o in select_lambdas_idx:
        lc += 1
        u_k = u_nks[o].drop(columns=[str(l) for l in remove_lambdas_idx])
        u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
        # add a column to df for using groupby
        u_k['lambda'] = f"lambda_{lc}"
        u_k['window'] = f"{lc}"
        # set lambda index for later groupby
        u_k = u_k.set_index(['lambda', 'window'])
        ret_u_nks.append(u_k)
    return ret_u_nks


if __name__ == "__main__":
    import argparse
    from copy import copy
    import matplotlib.pyplot as plt
    from alchemlyb.estimators import MBAR
    from util.real_data_handler import RealDataHandler
    from alchemlyb.visualisation import plot_mbar_overlap_matrix
    from extend_lambdas.lambda_multiplier import LambdaMultiplier
    from shortest_path_opt.distance_builder import DistanceBuilder
    from shortest_path_opt.shortest_path import ShortestPath

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="simulation data directory path which should"
                                                            "contain path of windows of prod_npt.csv")
    parser.add_argument("-t", "--temperature", type=float, default=310, help="(Kelvin)")
    args = parser.parse_args()

    handler = RealDataHandler.get_files_from_directory(directory=args.directory, temperature=args.temperature)
    lambda_multiplier = LambdaMultiplier(org_u_nks=handler.u_nks)

    # org_f_k = copy(lambda_multiplier.f_k)
    # org_distance_matrix = DistanceBuilder(u_nks=handler.u_nks).distance_matrix
    # org_exp_m = np.asmatrix(np.exp(-np.asarray(org_distance_matrix)))
    # print("\n org_Frobenius_norm: ", FrobeniusNorm(org_exp_m, org_exp_m))
    # f_k_list = [["org_f_k", org_f_k]]
    # matrix_distance_list = []
    # for gap in range(1, 9):
    #     lambda_multiplier = LambdaMultiplier(org_u_nks=handler.u_nks)
    #     remove_info = lambda_multiplier.drop(gap)
    #     lambda_multiplier.extend(insert_lambdas_info=remove_info, processes=3)
    #     new_f_k = copy(lambda_multiplier.f_k)
    #     f_k_list.append([f"est_f_k_{gap}", new_f_k])
    #     new_distance_matrix = DistanceBuilder(u_nks=lambda_multiplier.u_nks).distance_matrix
    #     d_exp_m = np.asmatrix(np.exp(-np.asarray(new_distance_matrix)) - np.exp(-np.asarray(org_distance_matrix)))
    #     matrix_distance = FrobeniusNorm(d_exp_m, d_exp_m)
    #     matrix_distance_list.append((gap, matrix_distance))
    #     print(f"interval: {gap}, matrix_distance: {matrix_distance}")
    #
    # color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
    # c = -1
    # for k, v in f_k_list:
    #     c += 1
    #     plt.plot(v, label=k, color=color_list[c % len(color_list)])
    # plt.gca().set(title='Free energy estimation', ylabel='F')
    # plt.legend()
    # plt.show()
    #
    # plt.close("all")
    # matrix_distance_list = np.asarray(matrix_distance_list)
    # plt.plot(matrix_distance_list[:, 0], matrix_distance_list[:, 1])
    # plt.gca().set(title='Matrix distance', ylabel='distance', xlabel="interval")
    # plt.show()

    mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in handler.u_nks]))
    plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
    plt.show()

    plt.close("all")
    lambda_multiplier.extend(times=1, processes=1)
    shortest_path = ShortestPath(u_nks=lambda_multiplier.u_nks)
    min_cost, select_seq = shortest_path.optimize(target_lambda_num=len(handler.u_nks))
    print("select_seq: ", select_seq)
    print("select_lambda_info:")
    for info in lambda_multiplier.getLambdasInfo(select_seq):
        print(str(info), end="\t")
    print()
    new_u_nks = selectFromUNKS(lambda_multiplier.u_nks, select_seq)
    mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in new_u_nks]))
    plot_mbar_overlap_matrix(mbar_estimator.overlap_matrix)
    plt.show()
