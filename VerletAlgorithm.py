import taichi as ti
from Function import *
import Quaternion as Quaternion
import numpy 


# ================= PFC3D 5.00 Itasca (2015) =============== #
def NeighborSearching(dem):
    neighList, contList = dem.ln, dem.lc
    neighList.InitNBSList()
    neighList.InsertParticle(dem.lp)
    neighList.FindNeighborP2W(dem)
    neighList.FindNeighborP2P(dem)


def ContactCal(dem):
    neighList, contList = dem.ln, dem.lc       
    contList.ContactSetup(dem)
    neighList.ResetNeighborList()
    contList.LinearModel(dem)
    contList.ResetContactList()
    contList.ResetFtIntegration()


@ti.kernel
def UpdateQuaternion(dem: ti.template()):
    partList = dem.lp
    for np in range(partList.particleNum[None]):
        pass


@ti.kernel
def UpdateAngularVelocity(dem: ti.template()):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        beta = matList.TorqueLocalDamping[matID]
        w, wi, aw0 = partList.w[np], partList.wi[np], partList.aw[np]

        generalTorque = (partList.Tex[np] + partList.Td[np] + partList.Tc[np]) * Zero2OneVector(partList.fixedW[np])

        '''J = Quaternion.QuaternionRotate(partList.q[np], partList.inv_I[np])
        L = partList.inv_I[np].inverse() @ w
        partList.aw[np] = J @ (generalTorque - beta * generalTorque - w.cross(L))
        w_bar = w + 0.5 * dem.Dt[None] * (aw0 + partList.aw[np])
        partList.q[np] = Quaternion.QuaternionTraction(w_bar * dem.Dt[None], partList.q[np])
        J = Quaternion.QuaternionRotate(partList.q[np], partList.inv_I[np])
        w = J @ L'''

        aw = partList.inv_I[np] @ (generalTorque - beta * generalTorque.norm() * Normalize(partList.w[np])) 
        w += 0.5 * dem.Dt[None] * (partList.aw[np] + aw)

        partList.w[np] = w + partList.w[np] * partList.fixedW[np]
        partList.aw[np] = aw


@ti.kernel
def UpdateVelocity(dem: ti.template()):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        alpha = matList.ForceLocalDamping[matID]
        v, vi, av0 = partList.v[np], partList.v[np], partList.av[np]

        generalForce = (partList.Fd[np] + partList.Fc[np] + partList.Fex[np] + dem.Gravity * partList.m[np]) * Zero2OneVector(partList.fixedV[np])
        av = (generalForce - alpha * generalForce.norm() * Normalize(partList.v[np])) / partList.m[np]

        vi += 0.5 * dem.Dt[None] * av0
        v += 0.5 * dem.Dt[None] * (av0 + av)

        partList.vi[np] = vi + partList.v[np] * partList.fixedV[np]
        partList.v[np] = v + partList.v[np] * partList.fixedV[np]
        partList.av[np] = av


@ti.kernel
def UpdatePosition(dem: ti.template()):
    partList = dem.lp
    for np in range(partList.particleNum[None]):
        partList.x[np] += partList.vi[np] * dem.Dt[None]
            
        # Todo: Update quaternion
        partList.Fc[np] = ti.Matrix.zero(float, 3)
        partList.Tc[np] = ti.Matrix.zero(float, 3)
        partList.Fd[np] = ti.Matrix.zero(float, 3)
        partList.Td[np] = ti.Matrix.zero(float, 3)
        # Todo: Periodic condition
