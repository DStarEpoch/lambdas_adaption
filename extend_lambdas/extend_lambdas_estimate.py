# -*- coding:utf-8 -*-
import numpy as np
from extend_lambdas.lambda_multiplier import LambdaMultiplier


def FrobeniusNorm(m1: np.matrix, m2: np.matrix):
    return np.sqrt(np.trace(np.dot(m1, m2.T)))


if __name__ == "__main__":
    import argparse
    from copy import copy
    import matplotlib.pyplot as plt
    from util.real_data_handler import RealDataHandler
    from shortest_path_opt.distance_builder import DistanceBuilder

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="simulation data directory path which should"
                                                            "contain path of windows of prod_npt.csv")
    parser.add_argument("-t", "--temperature", type=float, default=310)
    args = parser.parse_args()

    handler = RealDataHandler.get_files_from_directory(directory=args.directory, temperature=args.temperature)
    lambda_multiplier = LambdaMultiplier(org_u_nks=handler.u_nks)
    org_f_k = copy(lambda_multiplier.f_k)
    org_distance_matrix = DistanceBuilder(u_nks=handler.u_nks).distance_matrix
    org_exp_m = np.asmatrix(np.exp(-np.asarray(org_distance_matrix)))
    print("\n org_Frobenius_norm: ", FrobeniusNorm(org_exp_m, org_exp_m))

    f_k_list = [["org_f_k", org_f_k]]
    matrix_distance_list = []
    for gap in range(1, 9):
        lambda_multiplier = LambdaMultiplier(org_u_nks=handler.u_nks)
        remove_info = lambda_multiplier.drop(gap)
        lambda_multiplier.extend(insert_lambdas_info=remove_info)
        new_f_k = copy(lambda_multiplier.f_k)
        f_k_list.append([f"est_f_k_{gap}", new_f_k])
        new_distance_matrix = DistanceBuilder(u_nks=lambda_multiplier.u_nks).distance_matrix
        d_exp_m = np.asmatrix(np.exp(-np.asarray(new_distance_matrix)) - np.exp(-np.asarray(org_distance_matrix)))
        matrix_distance = FrobeniusNorm(d_exp_m, d_exp_m)
        matrix_distance_list.append((gap, matrix_distance))
        print(f"interval: {gap}, matrix_distance: {matrix_distance}")

    color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
    c = -1
    for k, v in f_k_list:
        c += 1
        plt.plot(v, label=k, color=color_list[c % len(color_list)])
    plt.gca().set(title='Free energy estimation', ylabel='F')
    plt.legend()
    plt.show()

    plt.close("all")
    matrix_distance_list = np.asarray(matrix_distance_list)
    plt.plot(matrix_distance_list[:, 0], matrix_distance_list[:, 1])
    plt.gca().set(title='Matrix distance', ylabel='distance', xlabel="interval")
    plt.show()
