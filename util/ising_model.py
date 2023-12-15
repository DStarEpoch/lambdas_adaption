# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np
from numpy.random import rand


class IsingModel(object):

    def __init__(self, N=100, beta=0.4):
        self.config = 2 * np.random.randint(2, size=(N, N)) - 1
        self.N = N
        self.beta = beta

    def mcmove(self):
        """Monte Carlo move using Metropolis algorithm """
        for i in range(self.N):
            for j in range(self.N):
                a = np.random.randint(0, self.N)
                b = np.random.randint(0, self.N)
                s = self.config[a, b]
                nb = self.config[(a + 1) % self.N, b] + self.config[a, (b + 1) % self.N] + \
                     self.config[(a - 1) % self.N, b] + self.config[a, (b - 1) % self.N]
                cost = 2 * s * nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * self.beta):
                    s *= -1
                self.config[a, b] = s

    @property
    def energy(self):
        """Energy of a given configuration"""
        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                S = self.config[i, j]
                nb = self.config[(i + 1) % self.N, j] + self.config[i, (j + 1) % self.N] + \
                     self.config[(i - 1) % self.N, j] + self.config[i, (j - 1) % self.N]
                energy += -nb * S
        return energy / 4.

    @property
    def dimless_energy(self):
        return self.energy * self.beta

    @property
    def magnetization(self):
        """Magnetization of a given configuration"""
        mag = np.sum(self.config)
        return mag

    def dimlessEnergyOnConfig(self, config):
        """Energy of a given configuration"""
        energy = 0
        for i in range(len(config)):
            for j in range(len(config[i])):
                S = config[i, j]
                nb = config[(i + 1) % len(config), j] + config[i, (j + 1) % len(config[i])] + \
                     config[(i - 1) % len(config), j] + config[i, (j - 1) % len(config[i])]
                energy += -nb * S
        energy *= self.beta
        return energy / 4.

    def swap(self, other: IsingModel):
        """Swap the configurations of two objects based on Metropolis algorithm"""
        before_swap_dimless_energy = self.dimless_energy + other.dimless_energy
        # try swap
        after_swap_dimless_energy = self.dimlessEnergyOnConfig(other.config) + other.dimlessEnergyOnConfig(self.config)
        delta_dimless_energy = after_swap_dimless_energy - before_swap_dimless_energy
        # Metropolis
        if delta_dimless_energy > 0:
            if rand() < np.exp(-delta_dimless_energy):
                self.config, other.config = other.config, self.config
                return True
        else:
            self.config, other.config = other.config, self.config
            return True
        return False
