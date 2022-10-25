import taichi as ti
import DEMLib3D.Quaternion as Quaternion
import DEMLib3D.DEM_particles as DEMParticle
import numpy 
import math


@ti.data_oriented
class DEMClumpTemplate(DEMParticle.DEMParticle):
    def __init__(self, max_template_num, max_pebble_num):
        self.max_template_num = max_template_num
        self.max_pebble_num = max_pebble_num

        self.templateNum = ti.field(int, shape=[])
        self.pebbleNum = ti.field(int, max_template_num)
        
        self.ID = ti.field(int, max_template_num)      
        self.center = ti.Vector.field(3, float, (max_template_num, max_pebble_num))
        self.rad = ti.field(float, (max_template_num, max_pebble_num))

        self.q = ti.Vector.field(4, float, max_particle_num)
        self.rotate = ti.Matrix.field(3, 3, float, max_particle_num)
        self.Am = ti.Vector.field(3, float, max_particle_num)

    
    # =========================================== Particle Initialization ====================================== #
    @ti.func
    def ClumpVolume(self, ncl, grid_size):
        for npee in range(self.pebbleNum[ncl]):
            posc = self.center[ncl, npee]
            rad = self.rad[ncl, npee]
            
 
    @ti.func
    def OrientationInit(self, np, norm):
        self.q[np] = Quaternion.SetFromTwoVec(ti.Vector([0., 0., 1.]), norm)
        self.rotate[np] = Quaternion.SetToRotate(self.q[np])


    @ti.func
    def ClumpParas(self, np, rad, pos, rho):
        self.rad[np] = rad
        self.m[np] = (4./3.) * rho * math.pi * rad ** 3
        self.x[np] = pos
        self.Am[np] = self.rotate[np].inverse() @ self.CalClumpIntertia() @ self.w[np]
        self.inv_I[np] = self.CalClumpIntertia().inverse()

    @ti.func
    def CalClumpIntertia(self):
        pass
