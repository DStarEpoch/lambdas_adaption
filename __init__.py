# -*- coding:utf-8 -*-
import gc
import numpy as np
import pandas as pd
from typing import List
from alchemlyb.estimators import MBAR
from dp_optimizer import optimize
from sample_generator import LambdaInfoContext, genSamplesForInsertLambda


def buildDistanceMatrix(u_nks_to_list: List[List[List[float]]]) -> List[List[float]]:
    def mean_delta_u(u_nk: List[List[float]], idx1: int, idx2: int):
        return np.mean(np.asarray(u_nk[idx2]) - np.asarray(u_nk[idx1]))
    dim = len(u_nks_to_list)
    distance_matrix: List[List[float]] = [[0.0 for __ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            else:
                # -ΔλΔ<u>
                distance_matrix[i][j] = distance_matrix[j][i] = \
                    -(mean_delta_u(u_nks_to_list[j], i, j) - mean_delta_u(u_nks_to_list[i], i, j))
    return distance_matrix


def adaptionAndOptLambdas(org_u_nks: List[pd.DataFrame], target_lambda_num: int,
                          retain_lambdas_idx: List[int] = None, extends: int = 1):
    if not retain_lambdas_idx:
        retain_lambdas_idx = [0, len(org_u_nks) - 1]
    else:
        retain_lambdas_idx = list(set(retain_lambdas_idx).union({0, len(org_u_nks) - 1}))
        retain_lambdas_idx.sort()

    mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in org_u_nks]))
    f_k = [0.0]
    for i in range(len(mbar_estimator.delta_f_) - 1):
        f_k.append(mbar_estimator.delta_f_.iloc[i, i + 1] + f_k[i])
    del mbar_estimator
    gc.collect()

    insert_lambdas_pos = []
    for i in range(len(org_u_nks) - 1):
        for t in range(extends):
            insert_lambdas_pos.append((i, i + 1, (t + 1) * 1.0 / (extends + 1)))

    org_u_nks_to_list: List[List[List[float]]] = [org_u_nk.transpose().values.tolist() for org_u_nk in org_u_nks]
    lambda_num = len(org_u_nks_to_list)
    samples_per_lambda = len(org_u_nks_to_list[0][0])
    bp_u_nks, all_lambdas_info = genSamplesForInsertLambda(lambda_num=lambda_num, samples_per_lambda=samples_per_lambda,
                                                           org_u_nks=org_u_nks_to_list, f_k=f_k,
                                                           insert_lambdas_pos=insert_lambdas_pos)
    bp_u_nks: List[List[List[float]]]
    all_lambdas_info: List[LambdaInfoContext]

    retain_lambdas_idx = list({idx for idx in range(len(all_lambdas_info)) if
                               (not all_lambdas_info[idx].is_insert and all_lambdas_info[idx].org_idx
                                in retain_lambdas_idx)})

    distance_matrix = buildDistanceMatrix(bp_u_nks)
    min_cost, select_seq = optimize(distance_matrix, retain_lambdas_idx, target_lambda_num)

    select_lambdas_pos = []
    for i in select_seq:
        info = all_lambdas_info[i]
        select_lambdas_pos.append((info.start_lambda_idx, info.end_lambda_idx, info.ratio))

    bp_u_nks.clear()
    all_lambdas_info.clear()
    del bp_u_nks
    del all_lambdas_info
    return min_cost, select_lambdas_pos
