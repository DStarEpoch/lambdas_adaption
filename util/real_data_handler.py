# -*- coding:utf-8 -*-
import pandas as pd
from typing import List, Union
from pathlib import Path
from scipy.constants import R
from functools import cached_property


class RealDataHandler:

    def __init__(self, directory: str, energy_files: List[Path], temperature=310):
        self.directory = Path(directory).resolve()
        self.energy_files = energy_files
        self.temperature = temperature

    @cached_property
    def u_nks(self) -> List[pd.DataFrame]:
        return [self.csvParser(subfile) for subfile in self.energy_files]

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
