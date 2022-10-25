import taichi as ti
import DEMLib3D.Quaternion as Quaternion
import numpy 
import math


@ti.data_oriented
class DEMParticle:
    def __init__(self, max_particle_num):
        self.particleNum = ti.field(int, shape=[])
        self.max_particle_num = max_particle_num
        self.shapeType = ti.field(int, max_particle_num)  
        self.ID = ti.field(int, max_particle_num)      
        self.group = ti.field(int, max_particle_num)
        self.materialID = ti.field(int, max_particle_num)
        self.cellID = ti.field(int, max_particle_num)

        self.m = ti.field(float, max_particle_num)
        self.rad = ti.field(float, max_particle_num)

        self.x = ti.Vector.field(3, float, max_particle_num)
        self.v = ti.Vector.field(3, float, max_particle_num)
        self.av = ti.Vector.field(3, float, max_particle_num)
        self.Fex = ti.Vector.field(3, float, max_particle_num)
        self.Fc = ti.Vector.field(3, float, max_particle_num)
        self.Fd = ti.Vector.field(3, float, max_particle_num)
        self.fixedV = ti.Vector.field(3, int, max_particle_num)
        self.q = ti.Vector.field(4, float, max_particle_num)
        self.rotate = ti.Matrix.field(3, 3, float, max_particle_num)
        self.Am = ti.Vector.field(3, float, max_particle_num)
        self.w = ti.Vector.field(3, float, max_particle_num)
        self.inv_I = ti.Matrix.field(3, 3, float, max_particle_num)
        self.Td = ti.Vector.field(3, float, max_particle_num)
        self.Tc = ti.Vector.field(3, float, max_particle_num)
        self.Tex = ti.Vector.field(3, float, max_particle_num)
        self.aw = ti.Vector.field(3, float, max_particle_num)
        self.fixedW = ti.Vector.field(3, int, max_particle_num)
    
    # =========================================== Particle Initialization ====================================== #
    @ti.func
    def OrientationInit(self, np, norm):
        self.q[np] = Quaternion.SetFromTwoVec(ti.Vector([0., 0., 1.]), norm)
        self.rotate[np] = Quaternion.SetToRotate(self.q[np])

    @ti.func
    def SphereParas(self, np, rad, pos, rho):
        self.rad[np] = rad
        self.m[np] = (4./3.) * rho * math.pi * rad ** 3
        self.x[np] = pos
        self.inv_I[np] = ((2./5.) * self.m[np] * self.rad[np] ** 2 * ti.Matrix.identity(float, 3)).inverse()

    @ti.func
    def DEMParticleInit(self, np, nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW, grav, dt):
        self.ID[np] = np
        self.group[np] = nb
        self.shapeType[np] = shapeType
        self.materialID[np] = MatID
        self.Fex[np] = fex
        self.Tex[np] = tex
        self.v[np] = v0
        self.w[np] = w0
        self.av[np] = fex / self.m[np] + grav
        self.aw[np] = self.inv_I[np] @ tex 
        self.fixedV[np] = fixedV
        self.fixedW[np] = fixedW

    @ti.kernel
    def FindMaxRadius(self) -> float:
        rmax = 0.
        for np in range(self.particleNum[None]):
            if self.rad[np] > rmax:
                rmax = self.rad[np]
        return rmax

    # =========================================== Particle Reset ====================================== #
    @ti.func
    def ParticleCalm(self, np):
        self.v[np] = ti.Matrix.zero(float, 3)
        self.w[np] = ti.Matrix.zero(float, 3)
    
    @ti.kernel
    def ComputeUnbalanceForce(self):
        pass

    
    # =========================================== Particle Generation ====================================== #
    @ti.kernel
    def CreateSphere(self, nb: int, BodyInfo: ti.template(), matList: ti.template(), grav: ti.types.vector(3, float), dt: float):
        assert self.particleNum[None] + 1 <= self.max_particle_num
        if BodyInfo[nb].rlo != BodyInfo[nb].rhi: print("Error occured in the input radius")
        print("Generate Type: Create Sphere")
        print("Material ID = ", BodyInfo[nb].Mat)
        print("Disk radius = ", BodyInfo[nb].rlo)
        print("Center of disk = ", BodyInfo[nb].pos0)
        print("The external force = ", BodyInfo[nb].fex)
        print("The external torque = ", BodyInfo[nb].tex)
        print("Initial Velocity = ", BodyInfo[nb].v0)
        print("Initial Orientation = ", BodyInfo[nb].orientation)
        print("Initial Angular Velocity = ", BodyInfo[nb].w0)
        print("Fixed Velocity = ", BodyInfo[nb].fixedV)
        print("Fixed Angular Velocity = ", BodyInfo[nb].fixedW)
        print("Particle Number = 1", '\n')

        shapeType = BodyInfo[nb].shapeType
        MatID = BodyInfo[nb].Mat
        rad = BodyInfo[nb].rlo
        pos = BodyInfo[nb].pos0
        fex = BodyInfo[nb].fex
        tex = BodyInfo[nb].tex
        v0 = BodyInfo[nb].v0
        norm = BodyInfo[nb].orientation
        w0 = BodyInfo[nb].w0
        fixedV = BodyInfo[nb].fixedV
        fixedW = BodyInfo[nb].fixedW
        
        self.SphereParas(self.particleNum[None], rad, pos, matList.rho[MatID])
        self.DEMParticleInit(self.particleNum[None], nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW, grav, dt)
        self.OrientationInit(self.particleNum[None], norm)
        self.particleNum[None] += 1

    @ti.kernel
    def GenerateSphere(self, nb: int, BodyInfo: ti.template(), matList: ti.template(), grav: ti.types.vector(3, float), dt: float):
        assert self.particleNum[None] + BodyInfo[nb].pnum <= self.max_particle_num
        print("Generate Type: Generate Disk")
        print("Material ID = ", BodyInfo[nb].Mat)
        print("Initial of box = ", BodyInfo[nb].pos0)
        print("Length of box = ", BodyInfo[nb].len)
        print("The maximum radius = ", BodyInfo[nb].rhi)
        print("The minimum radius = ", BodyInfo[nb].rlo)
        print("The external force = ", BodyInfo[nb].fex)
        print("The external torque = ", BodyInfo[nb].tex)
        print("Initial Velocity = ", BodyInfo[nb].v0)
        print("Initial Orientation = ", BodyInfo[nb].orientation)
        print("Initial Angular Velocity = ", BodyInfo[nb].w0)
        print("Fixed Velocity = ", BodyInfo[nb].fixedV)
        print("Fixed Angular Velocity = ", BodyInfo[nb].fixedW)
        print("Particle Number = ", BodyInfo[nb].pnum, '\n')
        shapeType = BodyInfo[nb].shapeType
        MatID = BodyInfo[nb].Mat
        fex = BodyInfo[nb].fex
        tex = BodyInfo[nb].tex
        v0 = BodyInfo[nb].v0
        norm = BodyInfo[nb].orientation
        w0 = BodyInfo[nb].w0
        fixedV = BodyInfo[nb].fixedV
        fixedW = BodyInfo[nb].fixedW

        head, tail = 0, 1
        desired_samples = BodyInfo[nb].pnum
        pos = BodyInfo[nb].pos0 + 0.5 * BodyInfo[nb].len
        radius = 0.5 * (BodyInfo[nb].rlo + BodyInfo[nb].rhi)
        self.SphereParas(self.particleNum[None], radius, pos, matList.rho[MatID])
        self.DEMParticleInit(self.particleNum[None], nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW, grav, dt)
        self.OrientationInit(self.particleNum[None], norm)
        while head < tail and head < desired_samples:
            source_x = self.x[self.particleNum[None] + head]
            source_rad = self.rad[self.particleNum[None] + head]
            head += 1
            for _ in range(100):
                radius = BodyInfo[nb].rlo + ti.random() * (BodyInfo[nb].rhi - BodyInfo[nb].rlo)
                randvector = ti.Vector([ti.random(), ti.random(), ti.random()])
                offset = randvector.normalized() * ((1 + ti.random()) * radius + source_rad)
                new_x = source_x + offset

                if BodyInfo[nb].pos0[0] + radius <= new_x[0] < BodyInfo[nb].pos0[0] + BodyInfo[nb].len[0] - radius \
                    and BodyInfo[nb].pos0[1] + radius <= new_x[1] < BodyInfo[nb].pos0[1] + BodyInfo[nb].len[1] - radius \
                    and BodyInfo[nb].pos0[2] + radius <= new_x[2] < BodyInfo[nb].pos0[2] + BodyInfo[nb].len[2] - radius:
                    collision = False
                    for p in range(self.particleNum[None], self.particleNum[None] + tail):
                        if (new_x - self.x[p]).norm() < self.rad[p] + radius:
                            collision = True
                    if not collision and tail < desired_samples:
                        self.SphereParas(self.particleNum[None] + tail, radius, new_x, matList.rho[MatID])
                        self.DEMParticleInit(self.particleNum[None] + tail, nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW, grav, dt)
                        self.OrientationInit(self.particleNum[None] + tail, norm)
                        tail += 1
        self.particleNum[None] += tail

    @ti.kernel
    def FillBallInBox(self, nb: int, BodyInfo: ti.template(), matList: ti.template(), grav: ti.types.vector(3, float), dt: float):
        pnum = ti.floor(BodyInfo[nb].len[0] / BodyInfo[nb].rad / 2) * ti.floor(BodyInfo[nb].len[1] / BodyInfo[nb].rad / 2)
        assert self.particleNum[None] + pnum <= self.max_particle_num
        print("Generate Type: Fill in Box")
        print("Material ID = ", BodyInfo[nb].Mat)
        print("The maximum radius = ", BodyInfo[nb].rhi)
        print("The minimum radius = ", BodyInfo[nb].rlo)
        print("Initial of box = ", BodyInfo[nb].pos0)
        print("Length of box = ", BodyInfo[nb].len)
        print("The external force = ", BodyInfo[nb].fex)
        print("The external torque = ", BodyInfo[nb].tex)
        print("Initial Velocity = ", BodyInfo[nb].v0)
        print("Initial Orientation = ", BodyInfo[nb].orientation)
        print("Initial Angular Velocity = ", BodyInfo[nb].w0)
        print("Fixed Velocity = ", BodyInfo[nb].fixedV)
        print("Fixed Angular Velocity = ", BodyInfo[nb].fixedW)
        print("Particle Number = ", pnum, '\n')
        shapeType = BodyInfo[nb].shapeType
        MatID = BodyInfo[nb].Mat
        fex = BodyInfo[nb].fex
        tex = BodyInfo[nb].tex
        v0 = BodyInfo[nb].v0
        norm = BodyInfo[nb].orientation
        w0 = BodyInfo[nb].w0
        fixedV = BodyInfo[nb].fixedV
        fixedW = BodyInfo[nb].fixedW

        tail = 0
        radius = BodyInfo[nb].rlo
        pos = BodyInfo[nb].pos0 + ti.Vector([radius, radius, radius])
        while pos[2] + radius <= BodyInfo[nb].pos0[2] + BodyInfo[nb].len[2]:
            pos[1] = BodyInfo[nb].pos0[1] + radius
            while pos[1] + radius <= BodyInfo[nb].pos0[1] + BodyInfo[nb].len[1]:
                pos[0] = BodyInfo[nb].pos0[0] + radius
                while pos[0] + radius <= BodyInfo[nb].pos0[0] + BodyInfo[nb].len[0]:
                    self.SphereParas(self.particleNum[None] + tail, radius, pos, matList.rho[MatID])
                    self.DEMParticleInit(self.particleNum[None] + tail, nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW, grav, dt)
                    self.OrientationInit(self.particleNum[None] + tail, norm)
                    tail += 1
                    pos[0] += 2 * radius
                pos[1] += 2 * radius
            pos[2] += 2 * radius
        self.particleNum[None] += tail

    @ti.kernel
    def CreateCuboid(self):
        pass

    @ti.kernel
    def CreateTetrahedron(self):
        pass
    
    # =============================================== Solve ======================================================= #
    @ti.func
    def CalViscousDampingEnergy(self):
        pass

    @ti.func
    def CalLocalDampingEnergy(self):
        pass

    @ti.func
    def CalContactStrainEnergy(self):
        pass  

    @ti.func
    def CalKineticEnergy(self):
        pass  

    @ti.func
    def CalExternalEnergy(self):
        pass

    @ti.func
    def CalSlippingEnergy(self):
        pass

    @ti.func
    def CalAngularMoment(self, np):
        return self.m[np] * self.x[np].cross(self.v[np])
