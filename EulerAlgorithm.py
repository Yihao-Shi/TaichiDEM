import taichi as ti
from Function import *
import Quaternion as Quaternion
import numpy 


# ================= David Baraff (1997) An Introduction to Physically Based Modeling: Rigid Body Simulation =============== #
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


@ti.kernel
def UpdateQuaternion(dem: ti.template()):
    partList = dem.lp
    for np in range(partList.particleNum[None]):
        w_2, aw_2 = 0.5 * partList.w[np], 0.5 * partList.aw[np]
        Qwb = Quaternion.SetFromValue(w_2[0], w_2[1], w_2[2], 0)
        Qawb = Quaternion.SetFromValue(aw_2[0], aw_2[1], aw_2[2], 0)
        Qb = partList.q[np]
        Aq = Quaternion.Multiply(Qb, Qwb)
        Q = Qb + Quaternion.Sacle(Aq, dem.Dt[None]) + 0.5 * dem.Dt[None] * dem.Dt[None] * (Quaternion.Multiply(Aq, Qwb) + Quaternion.Multiply(Qb, Qawb))
        Q = Quaternion.Normalized(Q)
        partList.q[np] = Quaternion.Multiply(partList.q0[np], Q)
        partList.rotate[np] = Quaternion.SetToRotate(partList.q[np])
        partList.irotate[np] = Quaternion.SetToRotate(Quaternion.inverse(partList.q[np]))


def NeighborSearching(dem):
    neighList, contList = dem.ln, dem.lc
    neighList.InitNBSList()
    neighList.InitWallList()
    neighList.InsertParticle(dem.lp)
    neighList.InsertWall(dem.lw)
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
def UpdateAngularVelocity(dem: ti.template()):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        beta = matList.TorqueLocalDamping[matID]
        w, awb = partList.w[np], partList.aw[np]

        generalTorque = (partList.Tex[np] + partList.Td[np] + partList.Tc[np]) * Zero2OneVector(partList.fixedW[np])
        
        J = partList.inv_I[np]
        aw0 = J @ (generalTorque - beta * generalTorque.norm() * Normalize(partList.w[np])) 
        w0 = w + 0.5 * dem.Dt[None] * (awb + aw0)
        aw1 = J @ (-w0.cross(J.inverse() @ w0))
        w1 = w0 + 0.5 * dem.Dt[None] * aw1
        aw2 = J @ (-w1.cross(J.inverse() @ w1))
        w2 = w1 + 0.5 * dem.Dt[None] * aw2

        partList.w[np] = w2 + partList.w[np] * partList.fixedW[np]
        partList.aw[np] = aw2


@ti.kernel
def UpdateVelocity(dem: ti.template()):
    partList, matList = dem.lp, dem.lm
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        alpha = matList.ForceLocalDamping[matID]
        v, av0 = partList.v[np], partList.av[np]

        generalForce = (partList.Fd[np] + partList.Fc[np] + partList.Fex[np] + dem.Gravity * partList.m[np]) * Zero2OneVector(partList.fixedV[np])
        av = (generalForce - alpha * generalForce.norm() * Normalize(partList.v[np])) / partList.m[np]
        v += 0.5 * dem.Dt[None] * (av0 + av)

        partList.v[np] = v + partList.v[np] * partList.fixedV[np]
        partList.av[np] = av
