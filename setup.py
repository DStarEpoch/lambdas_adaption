# -*- coding:utf-8 -*-
from distutils.core import setup, Extension

ext_module = Extension('ising', sources=['ising/ising_model.c'])
setup(name='lambda_adaption',
      version='1.0',
      description='Package for lambda adaption algorithm and examples',
      ext_modules=[ext_module]
      )
