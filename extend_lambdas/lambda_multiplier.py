# -*- coding:utf-8 -*-
import random
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from alchemlyb.estimators import MBAR
from util.real_data_handler import RealDataHandler
from alchemlyb.visualisation import plot_mbar_overlap_matrix
from extend_lambdas.boltzmann_picking import BoltzmannPicking


class LambdaMultiplier(object):

    def __init__(self, org_u_nks: List[pd.DataFrame]):
        self._u_nks = org_u_nks
        self._boltzmann_picking = BoltzmannPicking(org_u_nks=org_u_nks)

    @property
    def u_nks(self):
        return self._u_nks

    @property
    def f_k(self):
        return self._boltzmann_picking.f_k

    def extend(self, times: int = 1, insert_lambdas_info: List[Tuple[int, int, float]] = None):
        """

        param times: each pair of adjacent lambdas evenly insert times new lambdas
        :return:
        """

        if insert_lambdas_info is None:
            insert_lambdas_info = []
            for i in range(len(self._u_nks) - 1):
                for t in range(times):
                    insert_lambdas_info.append((i, i+1, t*1.0 / (times + 1)))

        bp_u_nks, new_lambdas_info = self._boltzmann_picking.genSamplesWithInsertLambda(
            insert_lambdas_info=insert_lambdas_info
        )

        self._u_nks = bp_u_nks
        initial_f_k = [info.f_k for info in new_lambdas_info]
        self._boltzmann_picking = BoltzmannPicking(org_u_nks=self._u_nks, initial_f_k=initial_f_k)

    def drop(self, interval: float = 1) -> List[Tuple[int, int, float]]:
        """

        :param interval:
        :return:
        """
        org_f_k = self.f_k
        remain_len = round((len(self.u_nks) - 1) / (interval + 1)) + 1
        spacing = len(self.u_nks) * 1.0 / remain_len

        # decide which lambdas to remove
        remove_lambda_idx = list()
        remaining_lambdas = [0]
        for i in range(remain_len):
            target_position = round((i + 1) * spacing)
            for j in range(remaining_lambdas[-1], target_position):
                remove_lambda_idx.append(j)
            remaining_lambdas.append(target_position)

        remain_f_k = [org_f_k[i] for i in remaining_lambdas]
        remain_u_nks = []
        lc = -1
        for idx in remaining_lambdas:
            lc += 1
            u_k = self.u_nks[idx].drop(columns=[str(l) for l in remove_lambda_idx])
            u_k.columns = [f'{i}' for i in range(len(u_k.columns))]
            # add a column to df for using groupby
            u_k['lambda'] = f"lambda_{lc}"
            u_k['window'] = f"{lc}"
            # set lambda index for later groupby
            u_k = u_k.set_index(['lambda', 'window'])
            remain_u_nks.append(u_k)

        self._u_nks = remain_u_nks
        self._boltzmann_picking = BoltzmannPicking(org_u_nks=remain_u_nks, initial_f_k=remain_f_k)
        remove_lambdas_info = []
        for i in remove_lambda_idx:
            i_pre = i_next = remaining_lambdas[0]
            for j in remaining_lambdas:
                if j > i:
                    i_next = j
                    break
                i_pre = j
            remove_lambdas_info.append((i_pre, i_next, (i - i_pre) / (i_next - i_pre)))

        return remove_lambdas_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="simulation data directory path which should"
                                                            "contain path of windows of prod_npt.csv")
    parser.add_argument("-i", "--interval", type=float, default=2)
    parser.add_argument("-t", "--temperature", type=float, default=310)


