import taichi as ti
import numpy 
import math


@ti.data_oriented
class DEMMaterial:
    def __init__(self, max_material_num):
        self.rho = ti.field(float, shape=(max_material_num,))
        self.ForceLocalDamping = ti.field(float, shape=(max_material_num,))
        self.TorqueLocalDamping = ti.field(float, shape=(max_material_num,))
        self.NormalViscousDamping = ti.field(float, shape=(max_material_num,))
        self.TangViscousDamping = ti.field(float, shape=(max_material_num,))
        self.modulus = ti.field(float, shape=(max_material_num,))
        self.possion = ti.field(float, shape=(max_material_num,))
        self.kn = ti.field(float, shape=(max_material_num,))
        self.ks = ti.field(float, shape=(max_material_num,))
        self.Mu = ti.field(float, shape=(max_material_num,))
        self.Rmu = ti.field(float, shape=(max_material_num,))

    @ti.kernel
    def ParticleLinearInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear contact Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.Mu[matID] = MatInfo[matID].Mu
        self.Rmu[matID] = MatInfo[matID].Rmu

    @ti.kernel
    def ParticleHertzInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Hertz contact Model')
        print('Shear modulus: = ', MatInfo[matID].Modulus)
        print('Possion ratio: = ', MatInfo[matID].possion)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Rolling friction coefficient = ', MatInfo[matID].Rmu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.Mu[matID] = MatInfo[matID].Mu
        self.Rmu[matID] = MatInfo[matID].Rmu

    @ti.kernel
    def ParticleLinearRollingInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear contact Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.Mu[matID] = MatInfo[matID].Mu
        self.Rmu[matID] = MatInfo[matID].Rmu

    @ti.kernel
    def ParticleBondInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear contact Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.Mu[matID] = MatInfo[matID].Mu
        self.Rmu[matID] = MatInfo[matID].Rmu

    
