import taichi as ti
import math
from Common.Function import *


@ti.data_oriented
class ContactModel:
    def __init__(self, max_material_num, dt):
        self.dt = dt
        self.max_material_num = max_material_num
        self.rho = ti.field(float, shape=(max_material_num,))
        self.ForceLocalDamping = ti.field(float, shape=(max_material_num,))
        self.TorqueLocalDamping = ti.field(float, shape=(max_material_num,))
        self.Mu = ti.field(float, shape=(max_material_num,))

    def UpdateDt(self, dt):
        self.dt = dt

    def DEMPMBarrierMethod(self):
        self.scalar = ti.field(float, shape=(self.max_material_num,))
        self.slipLim = ti.field(float, shape=(self.max_material_num,))

    @ti.func
    def BinarySearch(self, begining, ending, key, KEY):
        loc = -1
        while begining <= ending:
            mid_point = int((begining + ending) / 2)
            if KEY[mid_point] == key:
                loc = mid_point
                break
            elif KEY[mid_point] > key:
                ending = mid_point - 1
            elif KEY[mid_point] < key:
                begining = mid_point + 1
        return loc


@ti.data_oriented
class LinearContactModel(ContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)
        self.NormalViscousDamping = ti.field(float, shape=(max_material_num,))
        self.TangViscousDamping = ti.field(float, shape=(max_material_num,))
        self.kn = ti.field(float, shape=(max_material_num,))
        self.ks = ti.field(float, shape=(max_material_num,))

    def ResetMaterialProperty(self, keyword, modified_num):
        pass

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
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

    @ti.func
    def ComputeContactNormalForce(self, ContactPair, nc, matID1, matID2, m_eff, v_rel):
        kn = EffectiveValue(self.kn[matID1], self.kn[matID2])
        ndratio = ti.min(self.NormalViscousDamping[matID1], self.NormalViscousDamping[matID2])
        gapn = ContactPair.gapn[nc]
        norm = ContactPair.norm[nc]

        ContactPair.cnforce[nc] = kn * gapn * norm
        vn = v_rel.dot(norm) * norm
        ContactPair.cdnforce[nc] = -2 * ndratio * ti.sqrt(m_eff * kn) * vn

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, nc, matID1, matID2, m_eff, v_rel, end1, keyLoc):
        ks = EffectiveValue(self.ks[matID1], self.ks[matID2])
        miu = ti.min(self.Mu[matID1], self.Mu[matID2])
        nsratio = ti.min(self.TangViscousDamping[matID1], self.TangViscousDamping[matID2])
        norm = ContactPair.norm[nc]
        fn = ContactPair.cnforce[nc]

        vs = v_rel - v_rel.dot(norm) * norm
        trial_ft = -ks * vs * self.dt 
        
        if keyLoc != -1:
            TangForceOld = ContactPair.tangForceOld[end1, keyLoc]
            ft_ori = TangForceOld - TangForceOld.dot(norm) * norm
            ft_temp = TangForceOld.norm() * Normalize(ft_ori)
            trial_ft = trial_ft + ft_temp
        
        fric = miu * fn.norm()
        if trial_ft.norm() > fric:
            ContactPair.ctforce[nc] = fric * trial_ft.normalized()
        else:
            ContactPair.ctforce[nc] = trial_ft 

        ContactPair.cdsforce[nc] = -2 * nsratio * ti.sqrt(m_eff * ks) * vs


@ti.data_oriented
class HertzMindlinContactModel(ContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)
        self.modulus = ti.field(float, shape=(max_material_num,))
        self.possion = ti.field(float, shape=(max_material_num,))
        self.Restitution = ti.field(float, shape=(max_material_num,))

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Hertz contact Model')
        print('Shear modulus: = ', MatInfo[matID].Modulus)
        print('Possion ratio: = ', MatInfo[matID].possion)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Restitution = ', MatInfo[matID].Restitution)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.Restitution[matID] = MatInfo[matID].Restitution
        self.Mu[matID] = MatInfo[matID].Mu

    @ti.func
    def DampingRatio(self, matID1, matID2):
        restitution = ti.min(self.Restitution[matID1], self.Restitution[matID2])
        res = 0.
        if restitution == 0:
            res = 0.
        else:
            res = ti.log(restitution) / ti.sqrt(math.pi * math.pi + ti.log(restitution) * ti.log(restitution))
        return res

    @ti.func
    def ComputeContactNormalForce(self, ContactPair, nc, matID1, matID2, E1, E2, mu1, mu2, m_eff, rad_eff, v_rel):
        gapn = ContactPair.gapn[nc]
        norm = ContactPair.norm[nc]

        E = 1. / ((1 - mu1 * mu1) / E1 + (1 - mu2 * mu2) / E2)
        kn = 2 * E * ti.sqrt(gapn * rad_eff)
        res = self.DampingRatio(matID1, matID2)

        ContactPair.cnforce[nc] = 2./3. * kn * gapn * norm
        ContactPair.cdnforce[nc] = 1.8257 * res * v_rel.norm() * ti.sqrt(kn * m_eff) * sgn(v_rel.dot(norm)) * norm

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, nc, matID1, matID2, G1, G2, mu1, mu2, m_eff, rad_eff, v_rel, keyLoc):
        res = self.DampingRatio(matID1, matID2)
        miu = ti.min(self.Mu[matID1], self.Mu[matID2])
        gapn = ContactPair.gapn[nc]
        norm = ContactPair.norm[nc]
        fn = ContactPair.cnforce[nc]

        G = 1. / ((2 - mu1) / G1 + (2 - mu2) / G2)
        ks = 8 * G * ti.sqrt(gapn * rad_eff)
        vs = v_rel - v_rel.dot(norm) * norm
        trial_ft = -ks * vs * self.dt
        
        if keyLoc != -1:
            TangForceOld = ContactPair.TangForceMap.map[keyLoc]
            ft_ori = TangForceOld - TangForceOld.dot(norm) * norm
            ft_temp = TangForceOld.norm() * Normalize(ft_ori)
            trial_ft = trial_ft + ft_temp
        
        fric = miu * fn.norm()
        if trial_ft.norm() > fric:
            ContactPair.ctforce[nc] = fric * trial_ft.normalized()
        else:
            ContactPair.ctforce[nc] = trial_ft 
        ContactPair.cdsforce[nc] = 1.8257 * res * vs * ti.sqrt(ks * m_eff)


@ti.data_oriented
class LinearRollingResistanceContactModel(LinearContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)
        self.kr = ti.field(float, shape=(max_material_num,))
        self.kt = ti.field(float, shape=(max_material_num,))
        self.RMu = ti.field(float, shape=(max_material_num,))
        self.TMu = ti.field(float, shape=(max_material_num,))

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear Rolling Resistance Contact Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Contact rolling stiffness: = ', MatInfo[matID].Kr)
        print('Contact twisting stiffness: = ', MatInfo[matID].Kt)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Rolling Friction coefficient = ', MatInfo[matID].Rmu)
        print('Twisting Friction coefficient = ', MatInfo[matID].Tmu)
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
        self.kr[matID] = MatInfo[matID].Kr
        self.kt[matID] = MatInfo[matID].Kt
        self.Mu[matID] = MatInfo[matID].Mu
        self.RMu[matID] = MatInfo[matID].Rmu
        self.TMu[matID] = MatInfo[matID].Tmu
    
    # Luding (2008) Introduction to discrete element method
    @ti.func
    def ComputeRollingFriction(self, ContactPair, nc, matID1, matID2, w1, w2, rad_eff, keyLoc):
        norm = ContactPair.norm[nc]
        rmiu = ti.min(self.RMu[matID1], self.RMu[matID2])
        kr = EffectiveValue(self.kr[matID1], self.kr[matID2])
        fn = ContactPair.cnforce[nc]

        vr = -rad_eff * norm.cross(w1 - w2)
        trial_fr = -kr * vr * self.dt 
        
        if keyLoc != -1:
            TangRollingOld = ContactPair.TangRollingMap.map[keyLoc]
            fr_pre = TangRollingOld - TangRollingOld.dot(norm) * norm
            fr_temp = TangRollingOld.norm() * Normalize(fr_pre)
            trial_fr = trial_fr + fr_temp
        
        fricRoll = rmiu * fn.norm()
        if trial_fr.norm() > fricRoll:
            ContactPair.Fr[nc] = fricRoll * trial_fr.normalized()
        else:
            ContactPair.Fr[nc] = trial_fr
            

    # J. S. Marshall (2009) Discrete-element modeling of particulate aerosol flows /JCP/
    @ti.func
    def ComputeTorsionFriction(self, ContactPair, nc, matID1, matID2, w1, w2, rad_eff, keyLoc):
        norm = ContactPair.norm[nc]
        tmiu = ti.min(self.TMu[matID1], self.TMu[matID2])
        kt = EffectiveValue(self.kt[matID1], self.kt[matID2])
        fn = ContactPair.cnforce[nc]

        vt = rad_eff * (w1 - w2).dot(norm) * norm
        trial_ft = -kt * vt * self.dt
        
        if keyLoc != -1:
            TangTwistingOld = ContactPair.TangTwistingMap.map[keyLoc]
            ft_pre = TangTwistingOld.dot(TangTwistingOld) * norm
            ft_temp = TangTwistingOld.norm() * Normalize(ft_pre)
            trial_ft = trial_ft + ft_temp

        fricTwist = tmiu * fn.norm()
        if trial_ft.norm() > fricTwist:
            ContactPair.Ft[nc] = fricTwist * trial_ft.normalized()
        else:
            ContactPair.Ft[nc] = trial_ft


@ti.data_oriented
class LinearBondContactModel(LinearContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear Bond Model')
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

    
@ti.data_oriented
class LinearParallelBondContactModel(LinearContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear Parallel Bond Model')
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
