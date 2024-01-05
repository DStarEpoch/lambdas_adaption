# -*- coding:utf-8 -*-
import tqdm
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from alchemlyb.estimators import MBAR
from typing import List, Tuple, Dict, Union


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

    def __str__(self):
        return f"lambda: {self.start_lambda_idx}-{self.ratio}-{self.end_lambda_idx}"

    @property
    def rank(self):
        return self.start_lambda_idx * (1 - self.ratio) + self.end_lambda_idx * self.ratio


class CollectPickTagsContext(object):
    idx: int
    cur_lambda_info: LambdaInfoContext
    org_u_nks: List[pd.DataFrame]
    f_k: List[float]

    def __init__(self, idx: int, cur_lambda_info: LambdaInfoContext, org_u_nks: List[pd.DataFrame], f_k: List[float]):
        self.idx = idx
        self.cur_lambda_info = cur_lambda_info
        self.org_u_nks = org_u_nks
        self.f_k = f_k


def parallelCollectPickTags(context: CollectPickTagsContext) -> Dict[str, Union[float, List[int], object, int]]:
    lambda_info = context.cur_lambda_info
    org_u_nks = context.org_u_nks
    f_k = context.f_k
    context_idx = context.idx
    ret = {"idx": context_idx, "rank": lambda_info.rank, "lambda_info": lambda_info}

    samples_per_lambda = len(org_u_nks[0].iloc[:, 0])
    if not lambda_info.is_insert:
        ret["pick_tag_list"] = None
        return ret
    ratio = lambda_info.ratio
    start_lambda_idx = lambda_info.start_lambda_idx
    end_lambda_idx = lambda_info.end_lambda_idx
    sum_p = 0.0
    all_u_samples = dict()
    for o in range(len(org_u_nks)):
        p_over_bias_list = []
        for u_idx in range(samples_per_lambda):
            p_over_bias = np.exp(-(np.asarray(org_u_nks[o].iloc[u_idx, start_lambda_idx]) * (1 - ratio)
                                       + np.asarray(org_u_nks[o].iloc[u_idx, end_lambda_idx]) * ratio)) \
                              / np.sum([np.exp(f_k[k] - org_u_nks[o].iloc[u_idx, k])
                                        for k in range(len(org_u_nks))])
            p_over_bias_list.append(p_over_bias / samples_per_lambda)
        sum_p += sum(p_over_bias_list)
        all_u_samples[o] = p_over_bias_list
    f_insert = -np.log(sum_p)
    lambda_info.f_k = f_insert

    tag_list = []
    w_list = []
    for l_idx in all_u_samples:
        p_over_bias_list = all_u_samples[l_idx]
        for u_idx in range(len(p_over_bias_list)):
            w_list.append(p_over_bias_list[u_idx] / sum_p)
            tag_list.append((l_idx, u_idx))

    pick_tag_list = random.choices(tag_list, weights=w_list, k=samples_per_lambda)
    ret["pick_tag_list"] = pick_tag_list
    return ret


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

    def genSamplesWithInsertLambda(self, insert_lambdas_info: List[Tuple[int, int, float]], processes: int = 1) \
            -> Tuple[List[pd.DataFrame], List[LambdaInfoContext]]:
        all_lambdas_info = self._arrangeLambdaInfo(insert_lambdas_info=insert_lambdas_info)

        samples_per_lambda = len(self.org_u_nks[0].iloc[:, 0])

        # pre-compute free energy of inserted lambdas and picking list
        pick_tag_list_dict: Dict[float, List[int]] = dict()
        params = [CollectPickTagsContext(idx=idx, cur_lambda_info=info, org_u_nks=self.org_u_nks, f_k=self.f_k)
                  for idx, info in enumerate(all_lambdas_info)]
        if processes > 1:
            with Pool(processes) as pool:
                result = list(tqdm.tqdm(pool.imap(parallelCollectPickTags, params),
                                        total=len(params), desc="pre-compute free energy"))
                for r in result:
                    if r["pick_tag_list"] is None:
                        continue
                    pick_tag_list_dict[r["rank"]] = r["pick_tag_list"]
                    all_lambdas_info[r["idx"]] = r["lambda_info"]
        else:
            for param in tqdm.tqdm(params, total=len(params), desc="pre-compute free energy"):
                r = parallelCollectPickTags(param)
                if r["pick_tag_list"] is None:
                    continue
                pick_tag_list_dict[r["rank"]] = r["pick_tag_list"]
                all_lambdas_info[r["idx"]] = r["lambda_info"]

        # generate fake sampling for inserted lambda and reorder u_nks for all lambdas
        bp_u_nks = list()
        lc = -1
        for i in tqdm.tqdm(range(len(all_lambdas_info)), total=len(all_lambdas_info), desc="generate fake sampling"):
            lc += 1
            u_nk = dict()
            cur_info = all_lambdas_info[i]

            for iter_idx in range(len(all_lambdas_info)):
                # l2->cur
                # l1->iter, 1->iter_start, 1'->iter_end
                # u_l2_at_l1 = u_l2_at_1 * (1-r1) + u_l2_at_1' * r1
                iter_info = all_lambdas_info[iter_idx]
                if not cur_info.is_insert:
                    cur_org_idx = cur_info.org_idx
                    if not iter_info.is_insert:
                        iter_org_idx = iter_info.org_idx
                        u_nk[str(iter_idx)] = list(self.org_u_nks[cur_org_idx][str(iter_org_idx)])
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
                    cur_pick_tag_list = pick_tag_list_dict[cur_info.rank]
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

            u_nk = pd.DataFrame.from_dict(u_nk)
            u_nk.columns = [f'{i}' for i in range(len(u_nk.columns))]
            # add a column to df for using groupby
            u_nk['lambda'] = f"lambda_{lc}"
            u_nk['window'] = f"{lc}"
            # set lambda index for later groupby
            u_nk = u_nk.set_index(['lambda', 'window'])
            bp_u_nks.append(u_nk)

        self.f_k = [info.f_k for info in all_lambdas_info]
        return bp_u_nks, all_lambdas_info

    def _arrangeLambdaInfo(self, insert_lambdas_info: List[Tuple[int, int, float]]) -> List[LambdaInfoContext]:
        all_lambdas_info = [LambdaInfoContext(info[0], info[1], info[2], True)
                            for info in insert_lambdas_info] + self.org_lambdas_info
        all_lambdas_info.sort(key=lambda x: x.rank)
        return all_lambdas_info
