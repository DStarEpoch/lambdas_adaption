# -*- coding:utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import List, Tuple
from shortest_path_opt.distance_builder import DistanceBuilder
from dp_optimizer import DPShortestPathOptimizer


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
        self.distance_matrix = DistanceBuilder(u_nks).distance_matrix

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

        optimizer = DPShortestPathOptimizer(self.distance_matrix, retain_lambda_idx)
        min_cost, solution_seq = optimizer.optimize(target_lambda_num)

        return min_cost, solution_seq
