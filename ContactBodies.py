import taichi as ti
from Function import *


@ti.data_oriented
class DEMContact:
    def __init__(self, max_contact_num, cmtype):
        self.CMType, self.max_contact_num = cmtype, max_contact_num
        self.endID1 = ti.field(int, shape=(max_contact_num,))
        self.endID2 = ti.field(int, shape=(max_contact_num,))
        self.isw2p = ti.field(int, shape=(max_contact_num,))
        self.cpos = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.m_eff = ti.field(float, shape=(max_contact_num,))
        self.rad_eff = ti.field(float, shape=(max_contact_num,))
        self.gapn = ti.field(float, shape=(max_contact_num,))
        self.cnforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.ctforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.cdnforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.cdtforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.v_rel = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.norm = ti.Vector.field(3, float, shape=(max_contact_num,))

        self.kn = ti.field(float, shape=(max_contact_num,))
        self.kt = ti.field(float, shape=(max_contact_num,))
        self.Miu = ti.field(float, shape=(max_contact_num,))
        self.Rmu = ti.field(float, shape=(max_contact_num,))
        self.vdn = ti.field(float, shape=(max_contact_num,))
        self.vdt = ti.field(float, shape=(max_contact_num,))

        self.contactNum0, self.contactNum = ti.field(int, shape=()), ti.field(int, shape=())
        self.RelTangForce0 = ti.Struct.field({                                     # List of relative displacement at previous timestep
            "key": int,                                                            # Hash Index
            "ft": ti.types.vector(3, float),                                       # Trial tangential force
        }, shape=(max_contact_num,))
        self.RelTangForce = ti.Struct.field({                                      # List of relative displacement 
            "key": int,                                                            # Hash Index
            "ft": ti.types.vector(3, float),                                       # Trial tangential force
        }, shape=(max_contact_num,))

    @ti.kernel
    def ResetContactList(self):
        for nc in range(self.contactNum[None]):
            self.endID1[nc] = 0
            self.endID2[nc] = 0
            self.m_eff[nc] = 0.
            self.gapn[nc] = 0.
            self.cnforce[nc] = ti.Vector([0., 0., 0.])
            self.ctforce[nc] = ti.Vector([0., 0., 0.])
            self.cdnforce[nc] = ti.Vector([0., 0., 0.])
            self.cdtforce[nc] = ti.Vector([0., 0., 0.])
            self.v_rel[nc] = ti.Vector([0., 0., 0.])
            self.norm[nc] = ti.Vector([0., 0., 0.])
            self.cpos[nc] = ti.Vector([0., 0., 0.])

            self.kn[nc] = 0.
            self.kt[nc] = 0.
            self.Miu[nc] = 0.
            self.Rmu[nc] = 0.
            self.vdn[nc] = 0.
            self.vdt[nc] = 0.
        
    @ti.kernel
    def ResetFtIntegration(self):
        for nc in self.RelTangForce:
            self.RelTangForce0[nc].key = self.RelTangForce[nc].key
            self.RelTangForce0[nc].ft = self.RelTangForce[nc].ft

        for nc in self.RelTangForce:
            self.RelTangForce[nc].key = -1
            self.RelTangForce[nc].ft = ti.Vector([0., 0., 0.])
        self.contactNum0[None] = self.contactNum[None]

    @ti.func
    def LinearModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        kn1, kn2, kt1, kt2 = matList.kn[matID1], matList.kn[matID2], matList.kt[matID2], matList.kt[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]

        self.kn[nc] = EffectiveValue(kn1, kn2)
        self.kt[nc] = EffectiveValue(kt1, kt2)
        self.Miu[nc] = ti.min(Miu1, Miu2)
        self.vdn[nc] = ti.min(vdn1, vdn2)
        self.vdt[nc] = ti.min(vdt1, vdt2)

    @ti.func
    def HertzModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        modulus1, modulus2, possion1, possion2 = matList.modulus[matID1], matList.modulus[matID2], matList.possion[matID2], matList.possion[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]
   
        modulus_eff = 0.5 * (modulus1 + modulus2)
        possion_eff = 0.5 * (possion1 + possion2)
    
        self.kn[nc] = (2 * modulus_eff * ti.sqrt(2 * self.rad_eff[nc])) / (3 * (1 - possion_eff))
        self.kt[nc] = (2 * modulus_eff ** 2 * 3 * (1 - possion_eff))
        self.Miu[nc] = ti.min(Miu1, Miu2)
        self.vdn[nc] = ti.min(vdn1, vdn2)
        self.vdt[nc] = ti.min(vdt1, vdt2)

    @ti.func
    def LinearRollingModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        kn1, kn2, kt1, kt2 = matList.kn[matID1], matList.kn[matID2], matList.kt[matID2], matList.kt[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]

        self.kn[nc] = EffectiveValue(kn1, kn2)
        self.kt[nc] = EffectiveValue(kt1, kt2)
        self.Miu[nc] = ti.min(Miu1, Miu2)
        self.vdn[nc] = ti.min(vdn1, vdn2)
        self.vdt[nc] = ti.min(vdt1, vdt2)

    @ti.func
    def LinearBondModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        kn1, kn2, kt1, kt2 = matList.kn[matID1], matList.kn[matID2], matList.kt[matID2], matList.kt[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]
  
        self.kn[nc] = EffectiveValue(kn1, kn2)
        self.kt[nc] = EffectiveValue(kt1, kt2)
        self.Miu[nc] = ti.min(Miu1, Miu2)
        self.vdn[nc] = ti.min(vdn1, vdn2)
        self.vdt[nc] = ti.min(vdt1, vdt2)

    @ti.kernel
    def ContactSetup(self, dem: ti.template()):
        partList, wallList, matList, neighborList = dem.lp, dem.lw, dem.lm, dem.ln
        self.contactNum[None] = neighborList.contact_pair_num[None]
        for nc in range(neighborList.contact_P2W_num[None]):
            end1, end2 = int(neighborList.contactPair[nc, 0]), int(neighborList.contactPair[nc, 1])
            matID1, matID2 = wallList.materialID[end1], partList.materialID[end2]
            pos1, pos2 = neighborList.contactPos[nc, 0], neighborList.contactPos[nc, 1]
            vel1, vel2, w1, w2 = wallList.v[end1], partList.v[end2], wallList.w[end1], partList.rotate[end2] @ partList.w[end2]
            m2, rad2 = partList.m[end2], partList.rad[end2]

            self.endID1[nc], self.endID2[nc] = end1, end2
            self.isw2p[nc] = 1
            self.m_eff[nc] = EffectiveValue(1e12, m2)
            self.rad_eff[nc] = 0.5 * EffectiveValue(1e12, rad2)

            if self.CMType == 0:
                self.LinearModelParas(nc, end1, end2, matID1, matID2, matList)
            elif self.CMType == 1:
                self.HertzModelParas(nc, end1, end2, matID1, matID2, matList)
            
            self.gapn[nc] = rad2 - (pos2 - pos1).norm()
            self.norm[nc] = (pos1 - pos2).normalized()
            self.cpos[nc] = pos1
            self.v_rel[nc] = vel1 + w1.cross(self.cpos[nc] - pos1) - (vel2 + w2.cross(self.cpos[nc] - pos2))

        for nc in range(neighborList.contact_P2W_num[None], neighborList.contact_pair_num[None]): 
            end1, end2 = int(neighborList.contactPair[nc, 0]), int(neighborList.contactPair[nc, 1])
            matID1, matID2 = partList.materialID[end1], partList.materialID[end2]
            pos1, pos2 = neighborList.contactPos[nc, 0], neighborList.contactPos[nc, 1]
            vel1, vel2, w1, w2 = partList.v[end1], partList.v[end2], partList.rotate[end1] @ partList.w[end1], partList.rotate[end2] @ partList.w[end2]
            m1, m2, rad1, rad2 = partList.m[end1], partList.m[end2], partList.rad[end1], partList.rad[end2]

            self.endID1[nc], self.endID2[nc] = end1, end2
            self.m_eff[nc] = EffectiveValue(m1, m2)
            self.rad_eff[nc] = 0.5 * EffectiveValue(rad1, rad2)

            if self.CMType == 0:
                self.LinearModelParas(nc, end1, end2, matID1, matID2, matList)
            elif self.CMType == 1:
                self.HertzModelParas(nc, end1, end2, matID1, matID2, matList)

            self.gapn[nc] = rad1 + rad2 - (pos2 - pos1).norm()
            self.norm[nc] = (pos1 - pos2).normalized()
            self.cpos[nc] = pos1 + (rad1 - 0.5 * self.gapn[nc]) * self.norm[nc]
            self.v_rel[nc] = vel1 + w1.cross(self.cpos[nc] - pos1) - (vel2 + w2.cross(self.cpos[nc] - pos2))
    
    @ti.func
    def PairingFunc(self, i, j):
        return ((i + j) * (i + j + 1) / 2. + j)

    # ================= Stefan Luding (2008) Introduction to Discrete Element Method =============== #
    @ti.func
    def Friction(self, nc, dt, partList):
        vt = self.v_rel[nc] - self.v_rel[nc].dot(self.norm[nc]) * self.norm[nc]  
        dxc = vt * dt
        trial_ft0 = -self.kt[nc] * dxc
        self.cdtforce[nc] = -2 * self.vdt[nc] * ti.sqrt(self.m_eff[nc] * self.kt[nc]) * vt
        key  = int(self.PairingFunc(self.isw2p[nc] * partList.particleNum[None] + self.endID1[nc], self.endID2[nc]))
        for i in range(self.contactNum0[None]):
            if self.RelTangForce0[i].key == key:
                trial_ft0 += self.RelTangForce0[i].ft
        
        trial_ft = trial_ft0 - trial_ft0.dot(self.norm[nc]) * self.norm[nc]
        self.RelTangForce[nc].key = key
        self.RelTangForce[nc].ft = trial_ft
        self.ctforce[nc] = trial_ft 

        fric = self.Miu[nc] * self.cnforce[nc].norm()
        if self.ctforce[nc].norm() > fric:
            self.ctforce[nc] = fric * trial_ft.normalized()
            self.RelTangForce[nc].ft = fric * trial_ft.normalized()

    @ti.func
    def RollingFriction(self, nc, dt, partList, wallList):
        w1, w2 = 0., 0.
        end1, end2 = self.endID1[nc], self.endID2[nc]
        if self.isw2p[nc] == 1:
            w1, w2 = wallList.w[end1], partList.w[end2]
        elif self.isw2p[nc] == 0:
            w1, w2 = partList.w[end1], partList.w[end2]
        vij = self.rad_eff[nc] * (self.norm[nc].cross(w2) - self.norm[nc].cross(w1))
        vt = vij - vij.dot(self.norm[nc]) * self.norm[nc]

    @ti.func
    def TorsionFriction(self, nc, dt, partList, wallList):
        pass

    @ti.func
    def NormalForce(self, nc):
        self.cnforce[nc] = self.kn[nc] * self.gapn[nc] * self.norm[nc]
        self.cdnforce[nc] = 2 * self.vdn[nc] * ti.sqrt(self.m_eff[nc] * self.kn[nc]) * self.v_rel[nc].dot(self.norm[nc]) * self.norm[nc]
        
    @ti.kernel
    def LinearModel(self, dem: ti.template()):
        partList, wallList = dem.lp, dem.lw
        for nc in range(self.contactNum[None]):
            if self.m_eff[nc] > 0:
                end1, end2 = self.endID1[nc], self.endID2[nc]
                self.NormalForce(nc)
                self.Friction(nc, dem.Dt[None], partList)
                totalf = self.cnforce[nc] + self.ctforce[nc] + self.cdnforce[nc] + self.cdtforce[nc]
                if self.isw2p[nc] == 1:
                    center = (wallList.p1[end1] + wallList.p2[end1] + wallList.p3[end1] + wallList.p4[end1]) / 4.
                    ti.atomic_add(wallList.Fc[end1], totalf)
                    ti.atomic_add(wallList.Tc[end1], totalf.cross(center - self.cpos[nc])) ## Transform from world frame to object frame!!!!!!!!!!!
                elif self.isw2p[nc] == 0:
                    ti.atomic_add(partList.Fc[end1], totalf)
                    ti.atomic_add(partList.Tc[end1], totalf.cross(partList.x[end1] - self.cpos[nc]))
                ti.atomic_sub(partList.Fc[end2], totalf)
                ti.atomic_sub(partList.Tc[end2], totalf.cross(partList.x[end2] - self.cpos[nc]))

    @ti.kernel
    def HertzModel(self):
        pass

    @ti.kernel
    def LinearRollingModel(self):
        pass

    @ti.kernel
    def LinearBondModel(self):
        pass


