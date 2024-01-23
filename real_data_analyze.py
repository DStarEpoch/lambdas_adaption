# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from alchemlyb.estimators import MBAR
from util.real_data_handler import RealDataHandler
from util.opt_collections import buildDistanceMatrix
from util.opt_collections import adaptionInsertLambdas


def FrobeniusNorm(m1: np.matrix, m2: np.matrix):
    return np.sqrt(abs(np.trace(np.dot(m1, m2.T))))


def matrixFidelity(org_m: np.matrix, eval_m: np.matrix):
    org_norm = FrobeniusNorm(org_m, org_m)
    d_m = eval_m - org_m
    delta_norm = FrobeniusNorm(d_m, org_m)
    return (org_norm - delta_norm) / org_norm


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


def dropFromUNKS(org_f_k: List[float], interval: float = 1) \
        -> Tuple[List[float], List[Tuple[int, int, float]], List[int]]:
    """
    :param
    org_f_k: List[float], original free energy list
    interval: float, drop lambdas by interval

    :return:
    new_f_k: List[float], new free energy list for remaining lambdas
    remove_lambda_pos: List[Tuple[int, int, float]], list of dropped lambdas position, format (start, end, ratio)
    remain_lambda_idx: List[int],
    """
    remain_len = round((len(org_f_k) - 1) / (interval + 1)) + 1
    spacing = (len(org_f_k) - 1) * 1.0 / remain_len

    # decide which lambdas to remove
    remove_lambda_idx = list()
    remaining_lambdas = [0]
    for i in range(remain_len):
        target_position = round((i + 1) * spacing)
        for j in range(remaining_lambdas[-1] + 1, target_position):
            remove_lambda_idx.append(j)
        remaining_lambdas.append(target_position)

    remain_f_k = [org_f_k[idx] for idx in remaining_lambdas]

    org_lambda_idx_to_new_lambda_idx = {v: i for i, v in enumerate(remaining_lambdas)}
    remove_lambdas_pos = []
    for i in remove_lambda_idx:
        i_pre = i_next = remaining_lambdas[0]
        for j in remaining_lambdas:
            if j > i:
                i_next = j
                break
            i_pre = j
        ratio = (i - i_pre) / (i_next - i_pre)
        i_pre = org_lambda_idx_to_new_lambda_idx[i_pre]
        i_next = org_lambda_idx_to_new_lambda_idx[i_next]
        remove_lambdas_pos.append((i_pre, i_next, ratio))

    return remain_f_k, remove_lambdas_pos, remaining_lambdas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="simulation data directory path which should"
                                                            "contain path of windows of prod_npt.csv")
    parser.add_argument("-t", "--temperature", type=float, default=310, help="(Kelvin)")
    args = parser.parse_args()

    org_u_nks = RealDataHandler.get_files_from_directory(directory=args.directory, temperature=args.temperature).u_nks
    mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in org_u_nks]))
    org_distance_matrix = buildDistanceMatrix([u_nk.transpose().values.tolist() for u_nk in org_u_nks])
    org_exp_m = np.asmatrix(np.exp(-np.asarray(org_distance_matrix)))

    f_k = [0.0]
    for i in range(len(mbar_estimator.delta_f_) - 1):
        f_k.append(mbar_estimator.delta_f_.iloc[i, i+1] + f_k[i])
    del mbar_estimator

    fidelity_list = []
    for intvl in range(1, 9):
        new_f_k, insert_lambdas_pos, remain_lambda_idx = dropFromUNKS(f_k, intvl)
        new_u_nks = selectFromUNKS(org_u_nks, remain_lambda_idx)
        bp_u_nks, all_lambdas_info = adaptionInsertLambdas(new_u_nks, f_k=new_f_k,
                                                           insert_lambdas_pos=insert_lambdas_pos)

        predict_distance_matrix = buildDistanceMatrix(bp_u_nks)
        predict_distance_matrix = np.asmatrix(np.exp(-np.asarray(predict_distance_matrix)))

        fidelity = matrixFidelity(predict_distance_matrix, org_exp_m)
        fidelity_list.append((intvl, fidelity))

        bp_u_nks.clear()
        all_lambdas_info.clear()
        del bp_u_nks
        del all_lambdas_info
        print(f"interval: {intvl}, fidelity: {fidelity}")

    fidelity_list = np.asarray(fidelity_list)
    plt.plot(fidelity_list[:, 0], fidelity_list[:, 1])
    plt.gca().set(title='Matrix fidelity', ylabel='fidelity', xlabel="interval")
    plt.show()
