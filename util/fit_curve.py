import bezier
import numpy as np
from typing import List
import matplotlib.pyplot as plt


class BezierCurve:
    def __init__(self, nodes_pos: np.ndarray):
        self.nodes = np.asfortranarray([nodes_pos[:, 0], nodes_pos[:, 1]])
        self.curve = bezier.Curve(self.nodes, degree=len(nodes_pos) - 1)

    def evaluate(self, s: float):
        return self.curve.evaluate(float(s))

    def evaluate_multi(self, s_vals: List[float]):
        s_vals = np.array(s_vals)
        return self.curve.evaluate_multi(s_vals)

    def plot(self, s_vals: List[float]):
        curve_nodes = self.evaluate_multi(s_vals)
        plt.plot(curve_nodes[0], curve_nodes[1], color="red", marker="o")
        plt.show()
