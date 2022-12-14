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
        self.Tr = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.Tt = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.v_rel = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.vr_rel = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.vt_rel = ti.field(float, shape=(max_contact_num,))
        self.norm = ti.Vector.field(3, float, shape=(max_contact_num,))

        self.kn = ti.field(float, shape=(max_contact_num,))
        self.ks = ti.field(float, shape=(max_contact_num,))
        self.Miu = ti.field(float, shape=(max_contact_num,))
        self.Rmiu = ti.field(float, shape=(max_contact_num,))
        self.Tmiu = ti.field(float, shape=(max_contact_num,))

        self.contactNum0, self.contactNum = ti.field(int, shape=()), ti.field(int, shape=())
        self.RelTranslate = ti.Struct.field({                                      # List of relative displacement 
            "key": int,                                                            # Hash Index
            "ft": ti.types.vector(3, float)                                        # Trial tangential force
        }, shape=(max_contact_num,))
        self.RelRolling = ti.Struct.field({                                         # List of relative displacement 
            "key": int,                                                            # Hash Index
            "frt": ti.types.vector(3, float),                                       # Trial tangential force
        }, shape=(max_contact_num,))
        self.RelTwist = ti.Struct.field({                                         # List of relative displacement 
            "key": int,                                                            # Hash Index
            "ftt": ti.types.vector(3, float),                                       # Trial tangential force
        }, shape=(max_contact_num,))

    @ti.func
    def PairingFunc(self, i, j):
        return int((i + j) * (i + j + 1) / 2. + j)

    @ti.kernel
    def ResetContactList(self):
        for nc in range(self.contactNum[None]):
            self.endID1[nc] = 0
            self.endID2[nc] = 0
            self.isw2p[nc] = 0
            self.m_eff[nc] = 0.
            self.gapn[nc] = 0.
            self.cnforce[nc] = ti.Matrix.zero(float, 3)
            self.ctforce[nc] = ti.Matrix.zero(float, 3)
            self.Tr[nc] = ti.Matrix.zero(float, 3)
            self.Tt[nc] = ti.Matrix.zero(float, 3)
            self.v_rel[nc] = ti.Matrix.zero(float, 3)
            self.vr_rel[nc] = ti.Matrix.zero(float, 3)
            self.vt_rel[nc] = 0.
            self.norm[nc] = ti.Matrix.zero(float, 3)
            self.cpos[nc] = ti.Matrix.zero(float, 3)

            self.kn[nc] = 0.
            self.ks[nc] = 0.
            self.Miu[nc] = 0.
            self.Rmiu[nc] = 0.
            self.Tmiu[nc] = 0.
        
    @ti.kernel
    def ResetFtIntegration(self):
        for nc in range(self.contactNum0[None], self.contactNum[None]):
            self.RelTranslate[nc].key = -1
            self.RelTranslate[nc].ft = ti.Matrix.zero(float, 3)

        for nc in range(self.contactNum[None]):
            self.RelTranslate[nc].key = self.PairingFunc(self.endID1[nc], self.endID2[nc])
            self.RelTranslate[nc].ft = self.ctforce[nc]
        self.contactNum0[None] = self.contactNum[None]

    @ti.func
    def LinearModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        kn1, kn2, kt1, kt2 = matList.kn[matID1], matList.kn[matID2], matList.ks[matID2], matList.ks[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]

        self.kn[nc] = EffectiveValue(kn1, kn2)
        self.ks[nc] = EffectiveValue(kt1, kt2)
        self.Miu[nc] = ti.min(Miu1, Miu2)

    @ti.func
    def HertzModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        modulus1, modulus2, possion1, possion2 = matList.modulus[matID1], matList.modulus[matID2], matList.possion[matID2], matList.possion[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]
   
        modulus_eff = 0.5 * (modulus1 + modulus2)
        possion_eff = 0.5 * (possion1 + possion2)
    
        self.kn[nc] = (2 * modulus_eff * ti.sqrt(2 * self.rad_eff[nc])) / (3 * (1 - possion_eff))
        self.ks[nc] = (2 * modulus_eff ** 2 * 3 * (1 - possion_eff))
        self.Miu[nc] = ti.min(Miu1, Miu2)

    @ti.func
    def LinearRollingModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2, Rmu1, Rmu2 = matList.Mu[matID1], matList.Mu[matID2], matList.Rmiu[matID1], matList.Rmiu[matID2]
        kn1, kn2, kt1, kt2 = matList.kn[matID1], matList.kn[matID2], matList.ks[matID2], matList.ks[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]

        self.kn[nc] = EffectiveValue(kn1, kn2)
        self.ks[nc] = EffectiveValue(kt1, kt2)
        self.Miu[nc] = ti.min(Miu1, Miu2)
        self.Rmiu[nc] = ti.min(Rmu1, Rmu2)

    @ti.func
    def LinearBondModelParas(self, nc, end1, end2, matID1, matID2, matList):
        Miu1, Miu2 = matList.Mu[matID1], matList.Mu[matID2]
        kn1, kn2, kt1, kt2 = matList.kn[matID1], matList.kn[matID2], matList.ks[matID2], matList.ks[matID2]
        vdn1, vdn2, vdt1, vdt2 = matList.NormalViscousDamping[matID1], matList.TangViscousDamping[matID2], matList.NormalViscousDamping[matID2], matList.TangViscousDamping[matID2]
  
        self.kn[nc] = EffectiveValue(kn1, kn2)
        self.ks[nc] = EffectiveValue(kt1, kt2)
        self.Miu[nc] = ti.min(Miu1, Miu2)

    @ti.kernel
    def ContactSetup(self, dem: ti.template()):
        partList, wallList, matList, neighborList = dem.lp, dem.lw, dem.lm, dem.ln
        self.contactNum[None] = neighborList.contact_pair_num[None]
        for nc in range(neighborList.contact_P2W_num[None]):
            end1, end2 = int(neighborList.contactPair[nc, 0]), int(neighborList.contactPair[nc, 1])
            matID1, matID2 = wallList.materialID[end1], partList.materialID[end2]
            pos1, pos2 = neighborList.contactPos[nc, 0], neighborList.contactPos[nc, 1]
            vel1, vel2, w1, w2 = wallList.v[end1], partList.v[end2], wallList.w[end1], partList.w[end2]
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
            self.vr_rel[nc] = -self.rad_eff[nc] * (w1 - w2).cross(self.norm[nc])
            self.vt_rel[nc] = (w1 - w2).dot(self.norm[nc])

        for nc in range(neighborList.contact_P2W_num[None], neighborList.contact_pair_num[None]): 

            end1, end2 = int(neighborList.contactPair[nc, 0]), int(neighborList.contactPair[nc, 1])
            matID1, matID2 = partList.materialID[end1], partList.materialID[end2]
            pos1, pos2 = neighborList.contactPos[nc, 0], neighborList.contactPos[nc, 1]
            vel1, vel2, w1, w2 = partList.v[end1], partList.v[end2], partList.w[end1], partList.w[end2]
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
            vt = self.v_rel[nc] - self.v_rel[nc].dot(self.norm[nc]) * self.norm[nc]  
            self.vr_rel[nc] = -self.rad_eff[nc] * (w1 - w2).cross(self.norm[nc]) - 0.5 * ((rad2 - rad1) + (rad2 + rad1)) * vt
            self.vt_rel[nc] = (w1 - w2).dot(self.norm[nc])

    @ti.func
    def Friction(self, nc, dt, partList):
        vt = self.v_rel[nc] - self.v_rel[nc].dot(self.norm[nc]) * self.norm[nc]  
        trial_ft = -self.ks[nc] * vt * dt 
        key  = self.PairingFunc(self.isw2p[nc] * partList.particleNum[None] + self.endID1[nc], self.endID2[nc])
        for i in range(self.contactNum0[None]):
            if self.RelTranslate[i].key == key:
                ft_pre = self.RelTranslate[i].ft - self.RelTranslate[i].ft.dot(self.norm[nc]) * self.norm[nc]
                ft_temp = self.RelTranslate[i].ft.norm() * Normalize(ft_pre)
                trial_ft = trial_ft + ft_temp

        fric = self.Miu[nc] * self.cnforce[nc].norm()
        if trial_ft.norm() > fric:
            self.ctforce[nc] = fric * trial_ft.normalized()
        else:
            self.ctforce[nc] = trial_ft

    @ti.func
    def RollingFriction(self, nc, dt, partList, wallList):
        vrt = self.vr_rel[nc] - self.vr_rel[nc].dot(self.norm[nc]) * self.norm[nc]  
        trial_frt = -self.kr[nc] * vrt * dt 
        key  = self.PairingFunc(self.isw2p[nc] * partList.particleNum[None] + self.endID1[nc], self.endID2[nc])
        for i in range(self.contactNum0[None]):
            if self.RelRolling[i].key == key:
                frt_pre = self.RelRolling[i].frt - self.RelRolling[i].frt.dot(self.norm[nc]) * self.norm[nc]
                frt_temp = self.RelRolling[i].frt.norm() * Normalize(frt_pre)
                trial_frt = trial_frt + frt_temp

        fricRoll = self.Rmiu[nc] * self.cnforce[nc].norm()
        if trial_frt.norm() > fricRoll:
            self.Tr[nc] = fricRoll * trial_frt.normalized()
        else:
            self.Tr[nc] = trial_frt

    @ti.func
    def TorsionFriction(self, nc, dt, partList, wallList):
        vtt = self.vt_rel[nc] - self.vt_rel[nc].dot(self.norm[nc]) * self.norm[nc]  
        trial_ftt = -self.kt[nc] * vtt * dt 
        key  = self.PairingFunc(self.isw2p[nc] * partList.particleNum[None] + self.endID1[nc], self.endID2[nc])
        for i in range(self.contactNum0[None]):
            if self.RelTwist[i].key == key:
                ftt_pre = self.RelTwist[i].ftt - self.RelTwist[i].ftt.dot(self.norm[nc]) * self.norm[nc]
                ftt_temp = self.RelTwist[i].ftt.norm() * Normalize(ftt_pre)
                trial_ftt = trial_ftt + ft_ttemp

        fricTwist = self.Tmiu[nc] * self.cnforce[nc].norm()
        if trial_ftt.norm() > fricTwist:
            self.Tt[nc] = fricTwist * trial_ftt.normalized()
        else:
            self.Tt[nc] = trial_ftt

    @ti.func
    def NormalForce(self, nc):
        self.cnforce[nc] = self.kn[nc] * self.gapn[nc] * self.norm[nc]

    @ti.kernel
    def LinearModel(self, dem: ti.template()):
        partList, wallList = dem.lp, dem.lw
        for nc in range(self.contactNum[None]):
            if self.m_eff[nc] > 0:
                end1, end2 = self.endID1[nc], self.endID2[nc]
                self.NormalForce(nc)
                self.Friction(nc, dem.Dt[None], partList)
                Ftotal = self.cnforce[nc] + self.ctforce[nc]
                Ttotal = self.Tr[nc] + self.Tt[nc]
                #print(nc, self.endID1[nc], self.endID2[nc], self.norm[nc], self.cnforce[nc], self.ctforce[nc])
                if self.isw2p[nc] == 1:
                    center = (wallList.p1[end1] + wallList.p2[end1] + wallList.p3[end1] + wallList.p4[end1]) / 4.
                    wallList.Fc[end1] += Ftotal
                    wallList.Tc[end1] += Ftotal.cross(center - self.cpos[nc]) + Ttotal
                elif self.isw2p[nc] == 0:
                    partList.Fc[end1] += Ftotal
                    partList.Tc[end1] += Ftotal.cross(partList.x[end1] - self.cpos[nc]) + Ttotal
                partList.Fc[end2] -= Ftotal
                partList.Tc[end2] -= Ftotal.cross(partList.x[end2] - self.cpos[nc]) + Ttotal

    @ti.kernel
    def HertzModel(self):
        pass

    @ti.kernel
    def LinearRollingModel(self):
        pass

    @ti.kernel
    def LinearBondModel(self):
        pass


