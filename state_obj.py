# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np
import openmm as mm
from typing import List
from openmm.vec3 import Vec3
from openmm.unit import kelvin, joule, mole


class StateObj(object):
    R = 8.314 * joule / mole / kelvin
    force_template: str = '{force_constant}*(x-{x0})^2 + {p0} + 100*y^4 + 100*z^4'
    context: mm.Context
    brownian_gamma = 1.0
    integrator_step = 0.002

    def __init__(self, temperature, force_constant, x0, p0=0.0):
        system = mm.System()
        system.addParticle(1)  # added particle with a unit mass
        force = mm.CustomExternalForce(
            self.force_template.format(
                force_constant=force_constant,
                x0=x0,
                p0=p0
            )
        )  # defines the potential
        force.addParticle(0, [])
        system.addForce(force)

        self.x0 = x0
        self.p0 = p0
        self.force_constant = force_constant
        self.temperature = temperature * kelvin
        # brownian integrator with temperature, gamma, step size
        self.integrator = mm.BrownianIntegrator(temperature,
                                                self.brownian_gamma,
                                                self.integrator_step)
        self.context = mm.Context(system, self.integrator)
        self.context.setPositions([[x0, 0, 0]])
        self.context.setVelocitiesToTemperature(temperature)

        self._sample_x: List[List[List[Vec3]]] = list()

    @property
    def traj(self):
        ret = []
        for l in self._sample_x:
            ret += l
        return ret

    @property
    def potentials(self):
        self_cur_pos = self.context.getState(getPositions=True).getPositions()
        ret = []
        for l in self._sample_x:
            for x in l:
                self.context.setPositions(x)
                ret.append(self.context.getState(getEnergy=True).getPotentialEnergy() / (self.R * self.temperature))
        self.context.setPositions(self_cur_pos)
        return ret

    @property
    def properties(self) -> dict:
        ret = dict()
        ret["temperature"] = self.temperature / kelvin
        ret["force_constant"] = self.force_constant
        ret["x0"] = self.x0
        ret["velocities"] = self.context.getState(getVelocities=True).getVelocities()
        ret["positions"] = self.context.getState(getPositions=True).getPositions()
        ret["_sample_x"] = self._sample_x
        ret["brownian_gamma"] = self.brownian_gamma
        ret["integrator_step"] = self.integrator_step
        return ret

    @classmethod
    def createFromProperties(cls, properties):
        obj = cls(properties["temperature"], properties["force_constant"], properties["x0"])
        obj.context.setVelocities(properties["velocities"])
        obj.context.setPositions(properties["positions"])
        obj._sample_x = properties["_sample_x"]
        obj.brownian_gamma = properties["brownian_gamma"]
        obj.integrator_step = properties["integrator_step"]
        return obj

    def forward(self, n_steps, sample_steps) -> List[List[Vec3]]:
        ret_x = list()
        for i in range(1, n_steps + 1):
            self.integrator.step(1)
            if i % sample_steps == 0:
                ret_x.append(self.context.getState(getPositions=True).getPositions())
        self._sample_x.append(ret_x)
        return ret_x

    def tryExchange(self, other: StateObj):
        self_potential = self.context.getState(getEnergy=True).getPotentialEnergy()
        other_potential = other.context.getState(getEnergy=True).getPotentialEnergy()

        self_cur_v = self.context.getState(getVelocities=True).getVelocities()
        self_cur_pos = self.context.getState(getPositions=True).getPositions()
        other_cur_v = other.context.getState(getVelocities=True).getVelocities()
        other_cur_pos = other.context.getState(getPositions=True).getPositions()

        self.context.setPositions(other_cur_pos)
        self.context.setVelocities(other_cur_v)
        other.context.setPositions(self_cur_pos)
        other.context.setVelocities(self_cur_v)
        after_exchange_self_potential = self.context.getState(getEnergy=True).getPotentialEnergy()
        after_exchange_other_potential = other.context.getState(getEnergy=True).getPotentialEnergy()

        dE = after_exchange_self_potential + after_exchange_other_potential - (self_potential + other_potential)
        accept_rate = min(1, np.exp(-dE / (self.temperature * self.R)))
        if np.random.rand() < accept_rate:
            return True
        else:
            self.context.setPositions(self_cur_pos)
            self.context.setVelocities(self_cur_v)
            other.context.setPositions(other_cur_pos)
            other.context.setVelocities(other_cur_v)
            return False

    def calcRelativePotentialOnTraj(self, traj: List[List[Vec3]], trunk_step=10) -> List[float]:  # dimensionless
        potentials = list()
        system = mm.System()
        system.addParticle(1)  # added particle with a unit mass
        force = mm.CustomExternalForce(
            self.force_template.format(
                force_constant=self.force_constant,
                x0=self.x0,
                p0=self.p0
            )
        )  # defines the potential
        force.addParticle(0, [])
        system.addForce(force)
        integrator = mm.BrownianIntegrator(
            self.temperature / kelvin,
            self.brownian_gamma,
            self.integrator_step
        )
        state_context = mm.Context(system, integrator)
        for i in range(0, min(len(self.traj), len(traj)), trunk_step):
            target_pos = traj[i]
            if isinstance(target_pos, Vec3):
                target_pos = [target_pos, ]
            state_context.setPositions(self.traj[i])
            U1 = state_context.getState(getEnergy=True).getPotentialEnergy()
            try:
                state_context.setPositions(target_pos)
            except ValueError as e:
                print("target_pos: ", target_pos)
                raise e
            U2 = state_context.getState(getEnergy=True).getPotentialEnergy()
            dU = (U2 - U1) / (self.R * self.temperature)
            potentials.append(dU)
        return potentials
