# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from functools import partial


def calcEstOverlap(partial_overlap_matrix_list: List[np.matrix], org_overlap_matrix: np.matrix,
                   target_lambda_num: int, retained_lambda_idx: List[int],
                   lambda1: int, lambda2: int):
    total_lambda_num = len(partial_overlap_matrix_list[0])
    average_interval = round((total_lambda_num - len(retained_lambda_idx)) /
                             (target_lambda_num - len(retained_lambda_idx) + 1))
    # guess which remaining lambdas are not used in the calculation
    remove_lambda_list = list(range(lambda1 + 1, lambda2)) + \
                         [l for l in range(0, lambda1, average_interval) if l not in retained_lambda_idx] + \
                         [l for l in range(lambda2+1, total_lambda_num, average_interval)
                          if l not in retained_lambda_idx]
    remain_lambda_list = [l for l in range(total_lambda_num) if l not in remove_lambda_list]
    # calculate C factor for overlap estimation
    C1 = sum([partial_overlap_matrix_list[k][lambda1][lambda2] for k in remain_lambda_list])
    C2 = sum([partial_overlap_matrix_list[k][lambda1][lambda2] for k in range(total_lambda_num)])
    C = C2 / C1
    estimate_overlap = org_overlap_matrix[lambda1, lambda2] * C
    C1 = sum([partial_overlap_matrix_list[k][lambda1][lambda1] for k in remain_lambda_list])
    C2 = sum([partial_overlap_matrix_list[k][lambda1][lambda1] for k in range(total_lambda_num)])
    C = C2 / C1
    diag_overlap = org_overlap_matrix[lambda1, lambda1] * C
    return estimate_overlap / diag_overlap


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


def cost_func(estimate_overlap_func,
              retained_lambda_idx: List[int], estimate_mean: float,
              start_idx: int, end_idx: int, insert_idx: List[int] = None):

    if insert_idx is None:
        insert_idx = []

    idx_seq = []
    for i in range(start_idx, end_idx + 1):
        if i in retained_lambda_idx or i in insert_idx:
            idx_seq.append(i)

    if len(idx_seq) < 2:
        return float('inf')

    overlap_area_list = []
    for i in range(len(idx_seq) - 1):
        if idx_seq[i] not in insert_idx and idx_seq[i+1] not in insert_idx:
            continue
        overlap_area_list.append(
            estimate_overlap_func(idx_seq[i], idx_seq[i+1])
        )

    if len(overlap_area_list) <= 0:
        return float('inf')

    ret = 0.0
    for a in overlap_area_list:
        ret += (a - estimate_mean) ** 2

    return ret


def findOptLambdasByDP(estimate_overlap_func,
                       overlap_matrix: np.matrix,
                       target_lambda_num: int, estimate_mean: float,
                       retained_lambda_idx: List[int] = None) -> Tuple[float, List[int]]:
    if retained_lambda_idx is None:
        retained_lambda_idx = [0, len(overlap_matrix) - 1]

    n = len(overlap_matrix)
    m = target_lambda_num - len(retained_lambda_idx)
    reduce_cost_func = partial(cost_func, estimate_overlap_func, retained_lambda_idx, estimate_mean)

    # Initialize the dynamic programming table dp with large values
    dp = [[[DPInfo(k) for k in range(i + 1)] for j in range(m)] for i in range(n)]

    # Initialize the base case: no selected points
    for i in range(n):
        for k in range(i + 1):
            if k in retained_lambda_idx:
                continue
            dp[i][0][k].cost = reduce_cost_func(0, i, [k, ])

    for i in range(n):
        for j in range(1, m):
            for k in range(i + 1):
                if k in retained_lambda_idx:
                    # skip retained lambdas
                    continue
                # Try selecting index k
                for k_prime in range(k):
                    if k_prime in retained_lambda_idx:
                        continue
                    new_cost = dp[k_prime][j - 1][k_prime].cost + reduce_cost_func(k_prime, i, [k_prime, k])
                    if dp[i][j][k].cost > new_cost:
                        dp[i][j][k].cost = new_cost
                        dp[i][j][k].setParent(dp[k_prime][j - 1][k_prime])

    min_cost = float('inf')
    solution_seq = None
    for k in range(n):
        if dp[n - 1][m - 1][k].cost < min_cost:
            min_cost = dp[n - 1][m - 1][k].cost
            solution_seq = dp[n - 1][m - 1][k].sequence

    solution_seq += retained_lambda_idx
    solution_seq.sort()
    return min_cost, solution_seq


class DPOptimizer(object):
    _partial_overlap_matrix_list: List[np.matrix]
    _org_overlap_matrix: np.matrix
    _target_lambda_num: int
    _retained_lambda_idx: List[int]

    def __init__(self, partial_overlap_matrix_list: List[np.matrix], org_overlap_matrix: np.matrix,
                 target_lambda_num: int, retained_lambda_idx: List[int] = None) -> None:
        self._partial_overlap_matrix_list = partial_overlap_matrix_list
        self._org_overlap_matrix = org_overlap_matrix
        self._target_lambda_num = target_lambda_num
        if retained_lambda_idx is None:
            retained_lambda_idx = [0, len(org_overlap_matrix) - 1]
        self._retained_lambda_idx = retained_lambda_idx

    def optimize(self, estimate_mean) -> Tuple[float, List[int]]:
        return findOptLambdasByDP(
            partial(calcEstOverlap, self._partial_overlap_matrix_list, self._org_overlap_matrix,
                    self._target_lambda_num, self._retained_lambda_idx),
            self._org_overlap_matrix, self._target_lambda_num, estimate_mean, self._retained_lambda_idx)

