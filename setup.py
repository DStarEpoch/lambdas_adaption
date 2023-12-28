# -*- coding:utf-8 -*-
import numpy
from setuptools import setup, Extension, find_packages

include_dirs = [numpy.get_include()]

ising_module = Extension('ising', sources=['ising/ising_model.c'])
dp_optimizer_module = Extension('dp_optimizer',
                                sources=[
                                    'shortest_path_opt/source/dp_info.c',
                                    'shortest_path_opt/source/dp_shortest_path_optimizer.c',
                                    'shortest_path_opt/source/dp_optimizer.c',
                                ],
                                include_dirs=include_dirs + ['shortest_path_opt/include/',
                                                             'shortest_path_opt/'])

setup(name='lambda_adaption',
      py_modules=["lambda_adaption"],
      packages=find_packages(),
      version='1.0',
      description='Package for lambda adaption algorithm and examples',
      ext_modules=[ising_module, dp_optimizer_module]
      )
