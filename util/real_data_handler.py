# -*- coding:utf-8 -*-
import pandas as pd
from pathlib import Path
from scipy.constants import R
from typing import List, Union
import matplotlib.pyplot as plt
from functools import cached_property


class RealDataHandler:

    def __init__(self, directory: str, energy_files: List[Path], temperature=310):
        self.directory = Path(directory).resolve()
        self.energy_files = energy_files
        self.temperature = temperature

    @cached_property
    def u_nks(self) -> List[pd.DataFrame]:
        return [self.csvParser(subfile) for subfile in self.energy_files]

    @cached_property
    def enthalpies(self):
        return [self.csvEnthalpy(subfile) for subfile in self.energy_files]

    @classmethod
    def get_files_from_directory(cls, directory: Union[Path, str],
                                 energy_file_name: str = "prod_npt.csv",
                                 temperature=310):
        energy_files = sorted(Path(directory).glob(f"*/{energy_file_name}"), key=lambda x: int(x.parent.name))
        return cls(directory, energy_files, temperature)

    def csvParser(self, csv: Path):
        KB = R / 1000
        BETA = 1 / (KB * self.temperature)  # mol / KJ

        df = pd.read_csv(csv, na_filter=True, memory_map=True, sep=r"\s+")

        # remove Time, U, pV column, only save the dU column
        u_k = df.loc[:, ~df.columns.isin(['Time', 'U', 'pV'])]

        # add pV and convert to reduced potential
        u_k = u_k.add(df.pV, axis='rows') * BETA

        # rename window list
        u_k.columns = [f'{i}' for i in range(len(u_k.columns))]

        # add a column to df for using groupby
        u_k['lambda'] = f"lambda_{csv.parent.name}"
        u_k['window'] = f"{int(csv.parent.name)}"  # for csv
        # set lambda index for later groupby
        return u_k.set_index(['lambda', 'window'])

    def csvEnthalpy(self, csv: Path) -> pd.DataFrame:
        KB = R / 1000
        BETA = 1 / (KB * self.temperature)
        df = pd.read_csv(csv, na_filter=True, memory_map=True, sep=r"\s+")
        H = (df.U + df.pV) * BETA
        return H


def plotRealDataEnergyDistribution(directory: str, temperature=310):
    import numpy as np
    from alchemlyb.estimators import MBAR
    handler = RealDataHandler.get_files_from_directory(directory=directory, temperature=temperature)
    enthalpies = handler.enthalpies
    color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
    for h_idx in range(len(enthalpies)):
        h = enthalpies[h_idx]
        plt.hist(h, bins=20, alpha=0.5, color=color_list[h_idx % len(color_list)], label=f"state_{h_idx}")
    plt.show()

    org_u_nks = handler.u_nks
    mbar_estimator = MBAR(method="L-BFGS-B").fit(pd.concat([u_nk for u_nk in org_u_nks]))
    f_k = [0.0]
    for i in range(len(mbar_estimator.delta_f_) - 1):
        f_k.append(mbar_estimator.delta_f_.iloc[i, i + 1] + f_k[i])
    plot_data = []
    target_idx = 18
    for state_idx in range(len(enthalpies)):
        for h_idx in range(len(enthalpies[state_idx])):
            h = enthalpies[state_idx][h_idx]
            base_u = sum([np.exp(f_k[k] - org_u_nks[k].iloc[h_idx, state_idx]) for k in range(len(f_k))])
            ratio = np.exp(f_k[target_idx]-org_u_nks[target_idx].iloc[h_idx, state_idx]) / base_u
            plot_data.append((h, ratio))
    plot_data = sorted(plot_data, key=lambda x: x[0])
    plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="simulation data directory path which should"
                                                            "contain path of windows of prod_npt.csv")
    parser.add_argument("-t", "--temperature", type=float, default=310)

    args = parser.parse_args()
    plotRealDataEnergyDistribution(args.directory, args.temperature)

