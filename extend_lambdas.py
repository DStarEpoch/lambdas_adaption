# -*- coding:utf-8 -*-
import random
import numpy as np
from pathlib import Path
from util.real_data_handler import RealDataHandler

system_output_path = Path("./real_data/ejm_31")
sim_type = "abfe"
org_u_nks = RealDataHandler.get_files_from_directory(directory=system_output_path / sim_type,
                                                     energy_file_name="prod_npt.csv").u_nks

estimate_start_lambda_idx = 15
