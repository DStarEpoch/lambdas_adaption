import bezier
import numpy as np
import matplotlib.pyplot as plt

nodes = np.asfortranarray([
    [0.0, 0.4, 0.625, 1.0],
    [0.0, -0.1, 0.8, 0.5],
])
curve = bezier.Curve(nodes, degree=3)
s_vals = np.linspace(0.0, 1.0, 10)
curve_nodes = curve.evaluate_multi(s_vals)

plt.plot(curve_nodes[0], curve_nodes[1], color="red", marker="o")
plt.show()
