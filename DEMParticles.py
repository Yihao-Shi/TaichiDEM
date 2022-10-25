import taichi as ti
import DEMLib3D.Quaternion as Quaternion
import numpy 
import math


@ti.data_oriented
class DEMParticle:
    def __init__(self, domain, max_particle_num, contModel, Gravity):
        self.particleNum = ti.field(int, shape=[])
        self.domain = domain
        self.max_particle_num = max_particle_num
        
        def Check():
            if self.max_particle_num == 0:
                print("Parameter /max_particle_num/ must be larger than 0")
                assert 0
        Check()

        self.shapeType = ti.field(int, max_particle_num)  
        self.ID = ti.field(int, max_particle_num)      
        self.group = ti.field(int, max_particle_num)
        self.materialID = ti.field(int, max_particle_num)
        self.cellID = ti.field(int, max_particle_num)

        self.m = ti.field(float, max_particle_num)
        self.rad = ti.field(float, max_particle_num)

        self.x = ti.Vector.field(3, float, max_particle_num)
        self.disp = ti.Vector.field(3, float, max_particle_num)
        self.v = ti.Vector.field(3, float, max_particle_num)
        self.av = ti.Vector.field(3, float, max_particle_num)
        self.Fex = ti.Vector.field(3, float, max_particle_num)
        self.Fc = ti.Vector.field(3, float, max_particle_num)
        self.fixedV = ti.Vector.field(3, int, max_particle_num)
        
        self.theta = ti.Vector.field(3, float, max_particle_num)
        self.w = ti.Vector.field(3, float, max_particle_num)
        self.aw = ti.Vector.field(3, float, max_particle_num)
        self.inv_I = ti.field(float, max_particle_num)
        self.Tc = ti.Vector.field(3, float, max_particle_num)
        self.Tex = ti.Vector.field(3, float, max_particle_num)
        self.fixedW = ti.Vector.field(3, int, max_particle_num)

        self.contModel = contModel
        self.gravity = Gravity
    
    # =========================================== Particle Initialization ====================================== #
    @ti.func
    def SphereParas(self, np, rad, pos, rho):
        self.rad[np] = rad
        self.m[np] = (4./3.) * rho * math.pi * rad ** 3
        self.x[np] = pos
        self.inv_I[np] = 1. / ((2./5.) * self.m[np] * self.rad[np] ** 2)

    @ti.func
    def DEMParticleInit(self, np, nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW):
        self.ID[np] = np
        self.group[np] = nb
        self.shapeType[np] = shapeType
        self.materialID[np] = MatID
        self.Fex[np] = fex
        self.Tex[np] = tex
        self.v[np] = v0
        self.w[np] = w0
        self.fixedV[np] = fixedV
        self.fixedW[np] = fixedW

    @ti.kernel
    def FindMaxRadius(self) -> float:
        rmax = 0.
        for np in range(self.particleNum[None]):
            if self.rad[np] > rmax:
                rmax = self.rad[np]
        return rmax

    @ti.func
    def CheckInDomain(self, np):
        return all(ti.Matrix.zero(float, 3) <= self.x[np] <= self.domain)

    # =========================================== Particle Reset ====================================== #
    def UpdateParticleVel(self, particle_id, v):
        self.v[particle_id] = v

    @ti.func
    def ParticleCalm(self, np):
        self.v[np] = ti.Matrix.zero(float, 3)
        self.w[np] = ti.Matrix.zero(float, 3)
    
    @ti.kernel
    def ComputeUnbalanceForce(self):
        pass

    @ti.kernel
    def ComputeMaximumParticleDisp(self) -> float:
        max_particle_disp = 0.
        for np in range(self.particleNum[None]):
            disp = self.disp[np].norm()
            ti.atomic_max(max_particle_disp, disp)
        return max_particle_disp

    @ti.kernel
    def ResetParicleDisp(self):
        for np in range(self.particleNum[None]):
            self.disp[np] = ti.Matrix.zero(float, 3)

    @ti.kernel
    def DeleteParticles(self):
        remaining_particles = 0
        ti.loop_config(serialize=True)
        for np in range(self.particleNum[None]):
            if self.CheckInDomain(np):
                self.shapeType[remaining_particles] = self.shapeType[np]
                self.ID[remaining_particles] = self.ID[np]
                self.group[remaining_particles] = self.group[np]
                self.materialID[remaining_particles] = self.materialID[np]
                self.cellID[remaining_particles] = self.cellID[np]

                self.m[remaining_particles] = self.m[np]
                self.rad[remaining_particles] = self.rad[np]

                self.x[remaining_particles] = self.x[np]
                self.disp[remaining_particles] = self.disp[np]
                self.v[remaining_particles] = self.v[np]
                self.av[remaining_particles] = self.av[np]
                self.Fex[remaining_particles] = self.Fex[np]
                self.Fc[remaining_particles] = self.Fc[np]
                self.fixedV[remaining_particles] = self.fixedV[np]
                
                self.theta[remaining_particles] = self.theta[np]
                self.w[remaining_particles] = self.w[np]
                self.aw[remaining_particles] = self.aw[np]
                self.inv_I[remaining_particles] = self.inv_I[np]
                self.Tc[remaining_particles] = self.Tc[np]
                self.Tex[remaining_particles] = self.Tex[np]
                self.fixedW[remaining_particles] = self.fixedW[np]
                remaining_particles += 1

        self.particleNum[None] = remaining_particles

    
    # =========================================== Particle Generation ====================================== #
    @ti.kernel
    def CreateSphere(self, nb: int, BodyInfo: ti.template()):
        if nb >= 0:
            assert BodyInfo[nb].rlo == BodyInfo[nb].rhi
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
            
            self.SphereParas(self.particleNum[None], rad, pos, self.contModel.rho[MatID])
            self.DEMParticleInit(self.particleNum[None], nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW)
            assert self.particleNum[None] + 1 <= self.max_particle_num
            self.particleNum[None] += 1

    @ti.kernel
    def GenerateSphere(self, nb: int, BodyInfo: ti.template()):
        if nb >= 0:
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
            assert self.particleNum[None] <= self.max_particle_num
            self.SphereParas(self.particleNum[None], radius, pos, self.contModel.rho[MatID])
            self.DEMParticleInit(self.particleNum[None], nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW)
            while head < tail <= desired_samples:
                source_x = self.x[self.particleNum[None] + head]
                source_rad = self.rad[self.particleNum[None] + head]
                head += 1
                for _ in range(100):
                    radius = BodyInfo[nb].rlo + ti.random() * (BodyInfo[nb].rhi - BodyInfo[nb].rlo)
                    u, v = ti.random(), ti.random()
                    theta, phi = 2 * math.pi * u, ti.acos(2 * v - 1)
                    randvector = ti.Vector([ti.sin(theta) * ti.sin(phi), ti.cos(theta) * ti.sin(phi), ti.cos(phi)])
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
                            assert self.particleNum[None] + tail <= self.max_particle_num
                            self.SphereParas(self.particleNum[None] + tail, radius, new_x, self.contModel.rho[MatID])
                            self.DEMParticleInit(self.particleNum[None] + tail, nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW)
                            tail += 1
            print("Particle Number: ", tail, '\n')
            self.particleNum[None] += tail

    @ti.kernel
    def DistributeSphere(self, nb: int, BodyInfo: ti.template()):
        if nb >= 0:
            print("Generate Type: Distribute Disk")
            print("Material ID = ", BodyInfo[nb].Mat)
            print("Initial of box = ", BodyInfo[nb].pos0)
            print("Length of box = ", BodyInfo[nb].len)
            print("The maximum radius = ", BodyInfo[nb].rhi)
            print("The minimum radius = ", BodyInfo[nb].rlo)
            print("Void Ratio = ", BodyInfo[nb].VoidRatio)
            print("The external force = ", BodyInfo[nb].fex)
            print("The external torque = ", BodyInfo[nb].tex)
            print("Initial Velocity = ", BodyInfo[nb].v0)
            print("Initial Orientation = ", BodyInfo[nb].orientation)
            print("Initial Angular Velocity = ", BodyInfo[nb].w0)
            print("Fixed Velocity = ", BodyInfo[nb].fixedV)
            print("Fixed Angular Velocity = ", BodyInfo[nb].fixedW)
            shapeType = BodyInfo[nb].shapeType
            MatID = BodyInfo[nb].Mat
            fex = BodyInfo[nb].fex
            tex = BodyInfo[nb].tex
            v0 = BodyInfo[nb].v0
            norm = BodyInfo[nb].orientation
            w0 = BodyInfo[nb].w0
            fixedV = BodyInfo[nb].fixedV
            fixedW = BodyInfo[nb].fixedW

            '''domain = BodyInfo[nc].len[0] * BodyInfo[nc].len[1] * BodyInfo[nc].len[2]
            desired_samples = 

            for np in range(BodyInfo[nb].pnum):
                
                self.SphereParas(self.particleNum[None], radius, pos, self.contModel.rho[MatID])
                self.DEMParticleInit(self.particleNum[None], nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW)
                self.OrientationInit(self.particleNum[None], norm)
            
            print("Particle Number: ", tail, '\n')
            self.particleNum[None] += tail'''

    @ti.kernel
    def FillBallInBox(self, nb: int, BodyInfo: ti.template()):
        if nb >= 0:
            assert BodyInfo[nb].rlo == BodyInfo[nb].rhi
            pnum = ti.floor(BodyInfo[nb].len[0] / BodyInfo[nb].rlo / 2) * ti.floor(BodyInfo[nb].len[1] / BodyInfo[nb].rlo / 2) * ti.floor(BodyInfo[nb].len[2] / BodyInfo[nb].rlo / 2)
            assert (self.particleNum[None] + pnum <= self.max_particle_num)
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
                        self.SphereParas(self.particleNum[None] + tail, radius, pos, self.contModel.rho[MatID])
                        self.DEMParticleInit(self.particleNum[None] + tail, nb, shapeType, MatID, fex, tex, v0, w0, fixedV, fixedW)
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
