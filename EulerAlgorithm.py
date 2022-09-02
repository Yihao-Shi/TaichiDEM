import taichi as ti
from Function import *
import Quaternion as Quaternion
import numpy 


# ================= M. P. Allen, D. J. Tildesley (1989) Computer simualtion of liquids =============== #
@ti.kernel
def AccerlationInit(dem: ti.template()):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        alpha, beta = matList.ForceLocalDamping[matID], matList.TorqueLocalDamping[matID]
        generalForce = (partList.Fd[np] + partList.Fc[np] + partList.Fex[np] + dem.Gravity * partList.m[np]) * Zero2OneVector(partList.fixedV[np])
        generalTorque = (partList.Tex[np] + partList.Td[np] + partList.Tc[np]) * Zero2OneVector(partList.fixedW[np])

        partList.av[np] = (generalForce - alpha * generalForce.norm() * Normalize(partList.v[np])) / partList.m[np]
        partList.aw[np] = partList.inv_i[np] @ (generalTorque - beta * generalTorque.norm() * Normalize(partList.w[np])) 


@ti.kernel
def UpdatePosition(dem: ti.template()):
    partList = dem.lp
    for np in range(partList.particleNum[None]):
        partList.x[np] += partList.v[np] * dem.Dt[None] + 0.5 * partList.av[np] * dem.Dt[None] ** 2
            
        partList.Fc[np] = ti.Matrix.zero(float, 3)
        partList.Tc[np] = ti.Matrix.zero(float, 3)
        partList.Fd[np] = ti.Matrix.zero(float, 3)
        partList.Td[np] = ti.Matrix.zero(float, 3)
        # Todo: Periodic condition


def NeighborSearching(dem):
    neighList, contList = dem.ln, dem.lc
    neighList.InitNBSList()
    neighList.SumParticles(dem.lp)
    neighList.FindNeighborP2W(dem)
    neighList.BoardNeighborList(dem.lp)
    neighList.BoardSearch(dem.lp)


def ContactCal(dem):
    neighList, contList = dem.ln, dem.lc       
    contList.ContactSetup(dem)
    neighList.ResetNeighborList()
    contList.LinearModel(dem)
    contList.ResetFtIntegration()
    contList.ResetContactList()
    

@ti.kernel
def UpdateAngularVelocity(dem: ti.template(), t: float):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        beta = matList.TorqueLocalDamping[matID]
        
        generalTorque = (partList.Tex[np] + partList.Td[np] + partList.Tc[np]) * Zero2OneVector(partList.fixedW[np])
        generalTorque = generalTorque - beta * generalTorque.norm() * Normalize(partList.w[np])
        Lt = partList.rotate[np] @ (partList.Am[np] + 0.5 * dem.Dt[None] * generalTorque)
        Lmid = partList.rotate[np] @ (partList.Am[np] + dem.Dt[None] * generalTorque)
        wt = partList.inv_I[np]  @ Lt
        wmid = partList.inv_I[np]  @ Lmid
        qt = partList.q[np] + 0.5 * dem.Dt[None] * Quaternion.SetDQ(partList.q[np], wt)
        partList.q[np] += dem.Dt[None] * Quaternion.SetDQ(qt, wmid)
        partList.w[np] = partList.rotate[np].inverse() @ wmid
        partList.rotate[np] = Quaternion.SetToRotate(partList.q[np])
        
        if t == 0:
            partList.Am[np] += 0.5 * dem.Dt[None] * generalTorque 
        else:
            partList.Am[np] += dem.Dt[None] * generalTorque 


@ti.kernel
def UpdateVelocity(dem: ti.template()):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        alpha = matList.ForceLocalDamping[matID]
        v, av0 = partList.v[np], partList.av[np]

        generalForce = (partList.Fd[np] + partList.Fc[np] + partList.Fex[np] + dem.Gravity * partList.m[np]) * Zero2OneVector(partList.fixedV[np])
        partList.av[np] = (generalForce - alpha * generalForce.norm() * Normalize(partList.v[np])) / partList.m[np]
        partList.v[np] += 0.5 * dem.Dt[None] * (av0 + partList.av[np])
