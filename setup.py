# -*- coding:utf-8 -*-
import numpy
from distutils.core import setup, Extension
include_dirs = [numpy.get_include()]

ising_module = Extension('ising', sources=['ising/ising_model.c'])
dp_shortest_path_optimizer_module = Extension('dp_shortest_path_optimizer',
                                              sources=['shortest_path_opt/source/dp_shortest_path_optimizer.c'],
                                              include_dirs=include_dirs + ['shortest_path_opt/include/'])
setup(name='lambda_adaption',
      version='1.0',
      description='Package for lambda adaption algorithm and examples',
      ext_modules=[ising_module, dp_shortest_path_optimizer_module]
      )
