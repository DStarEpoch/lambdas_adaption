# -*- coding:utf-8 -*-
import tqdm
import numpy as np
from copy import copy
from typing import List
from alchemlyb.estimators import MBAR


def calc_partial_overlap_matrix(mbar_estimator: MBAR) -> List[np.matrix]:
    partial_overlap_matrix = list()
    org_w_nk = mbar_estimator._mbar.W_nk
    N_K = mbar_estimator._mbar.N_k
    state_num = len(N_K)
    for k in tqdm.tqdm(range(state_num), total=state_num, desc="calculate partial overlap"):
        tmp_w_nk = copy(org_w_nk)
        for i in range(tmp_w_nk.shape[0]):
            for j in range(tmp_w_nk.shape[1]):
                tmp_w_nk[i][j] *= org_w_nk[i][k]
        O_ijk = N_K * (org_w_nk.T @ tmp_w_nk)
        partial_overlap_matrix.append(O_ijk)
    return partial_overlap_matrix
