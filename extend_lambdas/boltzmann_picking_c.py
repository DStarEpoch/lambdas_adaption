# -*- coding:utf-8 -*-
import pandas as pd
from typing import List, Tuple
from alchemlyb.estimators import MBAR
from sample_generator import LambdaInfoContext, SampleGenerator


class BoltzmannPicking(object):
    f_k: List[float]

    def __init__(self, org_u_nks: List[pd.DataFrame], initial_f_k: List[float] = None):
        self.org_u_nks = org_u_nks

        if initial_f_k is None:
            mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat(org_u_nks))
            self.f_k = [0.0]
            for i in range(len(mbar_estimator.delta_f_) - 1):
                self.f_k.append(mbar_estimator.delta_f_.iloc[i, i + 1] + self.f_k[i])
        else:
            self.f_k = [initial_f_k[i] for i in range(len(org_u_nks))]

    def genSamplesWithInsertLambda(self, insert_lambdas_pos: List[Tuple[int, int, float]]) \
            -> Tuple[List[pd.DataFrame], List[LambdaInfoContext]]:
        list_u_nks: List[List[List[float]]] = [org_u_nk.transpose().values.tolist() for org_u_nk in self.org_u_nks]

        lambda_num = len(list_u_nks)
        samples_per_lambda = len(list_u_nks[0][0])
        generator = SampleGenerator(lambda_num, samples_per_lambda, list_u_nks, self.f_k)
        bp_u_nks, all_lambdas_info = generator.genSamplesForInsertLambda(insert_lambdas_pos=insert_lambdas_pos)

        bp_u_nks_to_dfs = list()
        for i in range(len(bp_u_nks)):
            u_nk = dict()
            for j in range(len(bp_u_nks[i])):
                u_nk[str(j)] = bp_u_nks[i][j]
            u_nk = pd.DataFrame.from_dict(u_nk)
            u_nk.columns = [f'{k}' for k in range(len(u_nk.columns))]
            u_nk['lambda'] = f"lambda_{i}"
            u_nk['window'] = f"{i}"
            # set lambda index for later groupby
            u_nk = u_nk.set_index(['lambda', 'window'])
            bp_u_nks_to_dfs.append(u_nk)

        del bp_u_nks
        return bp_u_nks_to_dfs, all_lambdas_info

