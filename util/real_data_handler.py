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


def real_data_test(directory: str, temperature=310, target_lambda_num=32):
    import time
    from shortest_path_opt.shortest_path import ShortestPath
    handler = RealDataHandler.get_files_from_directory(directory=directory, temperature=temperature)
    dp_shortest_path = ShortestPath(handler.u_nks)
    start_t = time.time()
    min_cost, optimal_sequence = dp_shortest_path.optimize(target_lambda_num=target_lambda_num,
                                                           retain_lambda_idx=list(range(8)))
    end_t = time.time()
    print(f"min_cost: {min_cost} time: {end_t-start_t}\noptimal_sequence: {optimal_sequence}")

    enthalpies = handler.enthalpies
    color_list = ["red", "green", "blue", "yellow", "grey", "purple", "orange", "pink", "cyan", "brown"]
    for h_idx in range(len(enthalpies)):
        h = enthalpies[h_idx]
        plt.hist(h, bins=20, alpha=0.5, color=color_list[h_idx % len(color_list)], label=f"state_{h_idx}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="simulation data directory path which should"
                                                            "contain path of windows of prod_npt.csv")
    parser.add_argument("-t", "--temperature", type=float, default=310)

    args = parser.parse_args()
    real_data_test(args.directory, args.temperature)

