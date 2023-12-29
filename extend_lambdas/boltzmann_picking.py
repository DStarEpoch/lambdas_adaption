# -*- coding:utf-8 -*-
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
from alchemlyb.estimators import MBAR


class LambdaInfoContext(object):

    f: float = 0.0
    is_insert: bool = False
    org_idx: int = 0

    def __init__(self, start_lambda_idx: int, end_lambda_idx: int,
                 ratio: float, is_insert: bool = False):
        # U_insert = U_start * (1 - ratio) + U_end * ratio
        self.start_lambda_idx = start_lambda_idx
        self.end_lambda_idx = end_lambda_idx
        self.ratio = ratio
        self.is_insert = is_insert


class BoltzmannPicking(object):

    f_k: List[float]

    def __init__(self, org_u_nks: List[pd.DataFrame]):
        self.org_u_nks = org_u_nks
        mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat(org_u_nks))
        self.f_k = [0.0]
        for i in range(len(mbar_estimator.delta_f_) - 1):
            self.f_k.append(mbar_estimator.delta_f_.iloc[i, i + 1] + self.f_k[i])

    def genSampleForInsertLambda(self, insert_lambdas_info: List[Tuple[int, int, float]]):
        samples_per_lambda = len(self.org_u_nks[0].iloc[:, 0])

        base_u = [sum([np.exp(self.f_k[k] - np.asarray(self.org_u_nks[k].iloc[:, i]))
                       for k in range(len(self.org_u_nks))])
                  for i in range(len(self.org_u_nks))]

        bp_u_nks = list()
        for insert_lambda in insert_lambdas:
            ratio = insert_lambda.ratio
            start_lambda_idx = insert_lambda.start_lambda_idx
            end_lambda_idx = insert_lambda.end_lambda_idx
            all_u_samples = dict()
            exp_f_insert = 0.0
            for o in range(len(self.org_u_nks)):
                u = np.asarray(self.org_u_nks[o].iloc[:, start_lambda_idx]) * (1 - ratio) + \
                    np.asarray(self.org_u_nks[o].iloc[:, end_lambda_idx]) * ratio
                exp_f_insert += sum(np.exp(-u) / base_u[o]) / samples_per_lambda
                all_u_samples[o] = u
            f_insert = -np.log(exp_f_insert)

            tag_list = []
            w_list = []
            for l_idx in all_u_samples:
                u = all_u_samples[l_idx]
                for u_idx in range(len(u)):
                    bias = sum([np.exp(self.f_k[k] - np.asarray(self.org_u_nks[l_idx].iloc[:, k][u_idx]))
                                for k in range(len(self.org_u_nks))])
                    w_list.append(np.exp(f_insert - u[u_idx]) / (bias * samples_per_lambda))
                    tag_list.append((l_idx, u_idx))

            pick_tag_list = random.choices(tag_list, weights=w_list, k=samples_per_lambda)

            bp_u_nk = pd.DataFrame()
            for i in range(len(self.org_u_nks)):
                if o2 != insert_lambda_idx:
                    u_k[str(o2)] = [org_u_nks[t][str(o2)][i] for t, i in pick_tag_list]
                else:
                    u_k[str(o2)] = [org_u_nks[t][str(estimate_start_lambda_idx)][i] * (1 - ratio) +
                                    org_u_nks[t][str(neighbour_lambda_idx)][i] * ratio for t, i in pick_tag_list]
            u_k = u_k[sorted(u_k.columns, key=lambda x: float(x))]

    def _arrangeLambdaIdx(self, insert_lambdas_info: List[Tuple[int, int, float]]) -> List[LambdaInfoContext]:
        ret = list()
        insert_lambdas_info = insert_lambdas_info + [(i, i, 0.0) for i in range(len(self.org_u_nks))]
        for info in insert_lambdas_info:

        return list()

