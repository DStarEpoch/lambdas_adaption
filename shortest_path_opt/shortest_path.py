# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from shortest_path_opt.distance_builder import DistanceBuilder


class DPInfo:
    _parent: DPInfo = None
    latest_insert_idx: int
    cost: float = float("inf")

    def __init__(self, insert_idx: int) -> None:
        self.latest_insert_idx = insert_idx

    @property
    def sequence(self):
        ptr = self
        ret_list = [ptr.latest_insert_idx]
        while ptr._parent is not None:
            ptr = ptr._parent
            ret_list.append(ptr.latest_insert_idx)
        ret_list.reverse()
        return ret_list

    def setParent(self, p: DPInfo):
        self._parent = p

    def __repr__(self) -> str:
        return f"<k: {self.latest_insert_idx}, cost: {round(self.cost, 3)}>"


class ShortestPath(object):

    def __init__(self, u_nks: List[pd.DataFrame]):
        self.distance_matrix: np.ndarray = DistanceBuilder(u_nks).distance_matrix

    def optimize(self, target_lambda_num: int, retain_lambda_idx: List[int] = None) -> Tuple[float, List[int]]:
        """
        find sequence between ends lambdas by the shortest distance with certain target_lambda_num
        :param target_lambda_num:
        :param retain_lambda_idx:   list of lambdas that forced to be retained
        :return:
        min_cost: minimum cost of the path
        optimal_sequence: optimal sequence of lambdas
        """
        if retain_lambda_idx is None:
            retain_lambda_idx = [0, len(self.distance_matrix) - 1]
        else:
            retain_lambda_idx = list(set(retain_lambda_idx).union([0, len(self.distance_matrix) - 1]))
            retain_lambda_idx.sort()

        if target_lambda_num <= len(retain_lambda_idx):
            cost = 0.0
            return cost, retain_lambda_idx

        n = len(self.distance_matrix)
        m = target_lambda_num - len(retain_lambda_idx)

        # initialize DPInfo
        # dp[i][j][k]: optimal path from lambda_0 to lambda_i with j selected points and try to insert lambda_k
        dp = [[[DPInfo(k) for k in range(i + 1)] for _ in range(m)] for i in range(n)]
        # Initialize no selected points case
        for i in range(n):
            for k in range(i + 1):
                dp[i][0][k].cost = self.distance_matrix[0][k] + self.distance_matrix[k][i]

        # DP
        for i in range(n):
            for j in range(1, m):
                for k in range(i + 1):
                    if k in retain_lambda_idx:
                        # skip retained lambdas
                        continue
                    # Try selecting index k
                    for k_prime in range(k):
                        if k_prime in retain_lambda_idx:
                            continue
                        new_cost = dp[k_prime][j - 1][k_prime].cost + \
                                   self.distance_matrix[k_prime][k] + self.distance_matrix[k][i]
                        if dp[i][j][k].cost > new_cost:
                            dp[i][j][k].cost = new_cost
                            dp[i][j][k].setParent(dp[k_prime][j - 1][k_prime])

        # find min cost
        min_cost = float('inf')
        solution_seq = None
        for k in range(n):
            if dp[n - 1][m - 1][k].cost < min_cost:
                min_cost = dp[n - 1][m - 1][k].cost
                solution_seq = dp[n - 1][m - 1][k].sequence
        solution_seq += retain_lambda_idx
        solution_seq.sort()

        return min_cost, solution_seq
