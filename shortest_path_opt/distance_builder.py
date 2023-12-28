# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from typing import List


class DistanceBuilder(object):

    def __init__(self, u_nks: List[pd.DataFrame]):
        self.u_nks = u_nks
        self._distance_matrix: List[List[float]] = self._build_distance_matrix()

    def _build_distance_matrix(self) -> List[List[float]]:
        def mean_delta_u(u_nk: pd.DataFrame, idx1: int, idx2: int):
            return np.mean(u_nk[str(idx2)] - u_nk[str(idx1)])
        dim = len(self.u_nks)
        distance_matrix: List[List[float]] = [[0.0 for __ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                else:
                    # -ΔλΔ<u>
                    distance_matrix[i][j] = distance_matrix[j][i] = \
                        -(mean_delta_u(self.u_nks[j], i, j) - mean_delta_u(self.u_nks[i], i, j))
        return distance_matrix

    @property
    def distance_matrix(self) -> List[List[float]]:
        return self._distance_matrix
