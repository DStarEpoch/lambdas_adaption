# -*- coding:utf-8 -*-
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
from alchemlyb.estimators import MBAR


class LambdaInfoContext(object):

    f_k: float = 0.0
    is_insert: bool = False
    org_idx: int = -1

    def __init__(self, start_lambda_idx: int, end_lambda_idx: int,
                 ratio: float, is_insert: bool = False,
                 org_idx: int = -1, f_k: float = 0.0):
        # U_insert = U_start * (1 - ratio) + U_end * ratio
        self.start_lambda_idx = start_lambda_idx
        self.end_lambda_idx = end_lambda_idx
        self.ratio = ratio
        self.is_insert = is_insert
        self.org_idx = org_idx
        self.f_k = f_k

    @property
    def rank(self):
        return self.start_lambda_idx * (1 - self.ratio) + self.end_lambda_idx * self.ratio


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

    @property
    def org_lambdas_info(self) -> List[LambdaInfoContext]:
        return [LambdaInfoContext(i, i, 0.0, False, i, self.f_k[i]) for i in range(len(self.org_u_nks))]

    def genSamplesWithInsertLambda(self, insert_lambdas_info: List[Tuple[int, int, float]]) \
            -> Tuple[List[pd.DataFrame], List[LambdaInfoContext]]:
        all_lambdas_info = self._arrangeLambdaInfo(insert_lambdas_info=insert_lambdas_info)

        samples_per_lambda = len(self.org_u_nks[0].iloc[:, 0])
        base_u = [sum([np.exp(self.f_k[k] - np.asarray(self.org_u_nks[k].iloc[:, i]))
                       for k in range(len(self.org_u_nks))])
                  for i in range(len(self.org_u_nks))]

        # pre-compute free energy of inserted lambdas and picking list
        pick_tag_list_dict = dict()
        all_u_samples_list = list()
        for lambda_info in all_lambdas_info:
            all_u_samples = dict()
            if not lambda_info.is_insert:
                all_u_samples_list.append(all_u_samples)
                continue
            ratio = lambda_info.ratio
            start_lambda_idx = lambda_info.start_lambda_idx
            end_lambda_idx = lambda_info.end_lambda_idx
            exp_f_insert = 0.0
            for o in range(len(self.org_u_nks)):
                u = np.asarray(self.org_u_nks[o].iloc[:, start_lambda_idx]) * (1 - ratio) + \
                    np.asarray(self.org_u_nks[o].iloc[:, end_lambda_idx]) * ratio
                exp_f_insert += sum(np.exp(-u) / base_u[o]) / samples_per_lambda
                all_u_samples[o] = u
            f_insert = -np.log(exp_f_insert)
            lambda_info.f_k = f_insert

            tag_list = []
            w_list = []
            for l_idx in all_u_samples:
                u = all_u_samples[l_idx]
                for u_idx in range(len(u)):
                    bias = sum([np.exp(self.f_k[k] - np.asarray(self.org_u_nks[l_idx].iloc[:, k][u_idx]))
                                for k in range(len(self.org_u_nks))])
                    w_list.append(np.exp(f_insert - u[u_idx]) / (bias * samples_per_lambda))
                    tag_list.append((l_idx, u_idx))

            all_u_samples_list.append(all_u_samples)
            pick_tag_list = random.choices(tag_list, weights=w_list, k=samples_per_lambda)
            pick_tag_list_dict[lambda_info.rank] = pick_tag_list

        # generate fake sampling for inserted lambda and reorder u_nks for all lambdas
        bp_u_nks = list()
        lc = -1
        for i in range(len(all_lambdas_info)):
            lc += 1
            u_nk = pd.DataFrame()
            cur_info = all_lambdas_info[i]
            cur_pick_tag_list = pick_tag_list_dict[cur_info.rank]

            for iter_idx in range(len(all_lambdas_info)):
                # l2->cur
                # l1->iter, 1->iter_start, 1'->iter_end
                # u_l2_at_l1 = u_l2_at_1 * (1-r1) + u_l2_at_1' * r1
                iter_info = all_lambdas_info[iter_idx]
                if not cur_info.is_insert:
                    cur_org_idx = cur_info.org_idx
                    if not iter_info.is_insert:
                        iter_org_idx = iter_info.org_idx
                        u_nk[str(iter_idx)] = self.org_u_nks[cur_org_idx][str(iter_org_idx)]
                    else:
                        iter_ratio = iter_info.ratio
                        iter_start_lambda_idx = iter_info.start_lambda_idx
                        iter_end_lambda_idx = iter_info.end_lambda_idx
                        u_nk[str(iter_idx)] = [self.org_u_nks[cur_org_idx][str(iter_start_lambda_idx)][u_idx]
                                               * (1 - iter_ratio) +
                                               self.org_u_nks[cur_org_idx][str(iter_end_lambda_idx)][u_idx]
                                               * iter_ratio
                                               for u_idx in range(samples_per_lambda)]
                else:
                    if not iter_info.is_insert:
                        iter_org_idx = iter_info.org_idx
                        u_nk[str(iter_idx)] = [self.org_u_nks[l_idx][str(iter_org_idx)][u_idx]
                                               for l_idx, u_idx in cur_pick_tag_list]
                    else:
                        iter_ratio = iter_info.ratio
                        iter_start_lambda_idx = iter_info.start_lambda_idx
                        iter_end_lambda_idx = iter_info.end_lambda_idx
                        u_nk[str(iter_idx)] = [self.org_u_nks[l_idx][str(iter_start_lambda_idx)][u_idx]
                                               * (1 - iter_ratio)
                                               + self.org_u_nks[l_idx][str(iter_end_lambda_idx)][u_idx]
                                               * iter_ratio
                                               for l_idx, u_idx in cur_pick_tag_list]

            u_nk.columns = [f'{i}' for i in range(len(u_nk.columns))]
            # add a column to df for using groupby
            u_nk['lambda'] = f"lambda_{lc}"
            u_nk['window'] = f"{lc}"
            # set lambda index for later groupby
            u_nk = u_nk.set_index(['lambda', 'window'])
            bp_u_nks.append(u_nk)

        return bp_u_nks, all_lambdas_info

    def _arrangeLambdaInfo(self, insert_lambdas_info: List[Tuple[int, int, float]]) -> List[LambdaInfoContext]:
        all_lambdas_info = [LambdaInfoContext(info[0], info[1], info[2], True)
                            for info in insert_lambdas_info] + self.org_lambdas_info
        all_lambdas_info.sort(key=lambda x: x.rank)
        return all_lambdas_info

