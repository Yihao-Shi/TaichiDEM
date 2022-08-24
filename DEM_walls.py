import taichi as ti
import numpy 
import math

# ==================================== Wall ============================================= #
#       SET LINES: LINE1 p1->p2; LINE2 p2->p3; LINE3 p3->p4; LINE4 p4->p1
                                 

@ti.data_oriented
class DEMWall:
    def __init__(self, max_wall_num):
        self.wallNum = ti.field(int, shape=[])
        self.ID = ti.field(int, max_wall_num)      
        self.materialID = ti.field(int, max_wall_num)
        self.isactive = ti.field(int, max_wall_num)
        self.p1 = ti.Vector.field(3, float, max_wall_num)
        self.p2 = ti.Vector.field(3, float, max_wall_num)
        self.p3 = ti.Vector.field(3, float, max_wall_num)
        self.p4 = ti.Vector.field(3, float, max_wall_num)
        self.norm = ti.Vector.field(3, float, max_wall_num)

        self.v = ti.Vector.field(3, float, max_wall_num)
        self.v0 = ti.Vector.field(3, float, max_wall_num)
        self.av = ti.Vector.field(3, float, max_wall_num)
        self.av0 = ti.Vector.field(3, float, max_wall_num)
        self.Fex = ti.Vector.field(3, float, max_wall_num)
        self.Fc = ti.Vector.field(3, float, max_wall_num)
        self.Fd = ti.Vector.field(3, float, max_wall_num)
        self.fixedV = ti.Vector.field(3, int, max_wall_num)
        self.w = ti.Vector.field(3, float, max_wall_num)
        self.w0 = ti.Vector.field(3, float, max_wall_num)
        self.I = ti.Matrix.field(3, 3, float, max_wall_num)
        self.Td = ti.Vector.field(3, float, max_wall_num)
        self.Tc = ti.Vector.field(3, float, max_wall_num)
        self.Tex = ti.Vector.field(3, float, max_wall_num)
        self.aw = ti.Vector.field(3, float, max_wall_num)
        self.aw0 = ti.Vector.field(3, float, max_wall_num)
        self.fixedW = ti.Vector.field(3, int, max_wall_num)
    
    @ti.func
    def GetWallNorm(self, p1, p2, p3):
        crossvec = (p2 - p1).cross(p3 - p1)
        self.norm[nw] = crossvec.normalized()

    @ti.func
    def DEMWallInit(self, nw, MatID, p1, p2, p3, p4, norm, fex, v0, w0, fixedV, fixedW):
        self.ID[nw] = nw
        self.materialID[nw] = MatID
        self.isactive[nw] = 1
        self.p1[nw] = p1
        self.p2[nw] = p2
        self.p3[nw] = p3
        self.p4[nw] = p4
        self.norm[nw] = norm
        self.Fex[nw] = fex
        self.v[nw] = v0
        self.v0[nw] = v0
        self.w[nw] = w0
        self.w0[nw] = w0
        self.fixedV[nw] = fixedV
        self.fixedW[nw] = fixedW

    @ti.kernel
    def CreatePlane(self, nb: int, WallInfo: ti.template()):
        print("Generate Type: Create Wall")
        print("Material ID = ", WallInfo[nb].Mat)
        print("The vertex the wall = ", WallInfo[nb].point1)
        print("The vertex the wall = ", WallInfo[nb].point2)
        print("The vertex the wall = ", WallInfo[nb].point3)
        print("The vertex the wall = ", WallInfo[nb].point4)
        print("The normal direction of the wall = ", WallInfo[nb].norm)
        print("The external force = ", WallInfo[nb].fex)
        print("Initial Velocity = ", WallInfo[nb].v0)
        print("Initial Angular Velocity = ", WallInfo[nb].w0)
        print("Fixed Velocity = ", WallInfo[nb].fixedV)
        print("Fixed Angular Velocity = ", WallInfo[nb].fixedW)
        print("Wall Number = ", self.wallNum[None] + 1, '\n')
        MatID = WallInfo[nb].Mat
        p1 = WallInfo[nb].point1
        p2 = WallInfo[nb].point2
        p3 = WallInfo[nb].point3
        p4 = WallInfo[nb].point4
        norm = WallInfo[nb].norm.normalized()
        fex = WallInfo[nb].fex
        v0 = WallInfo[nb].v0
        w0 = WallInfo[nb].w0
        fixedV = WallInfo[nb].fixedV
        fixedW = WallInfo[nb].fixedW
        self.DEMWallInit(self.wallNum[None], MatID, p1, p2, p3, p4, norm, fex, v0, w0, fixedV, fixedW)
        self.wallNum[None] += 1

    @ti.kernel
    def CreateWall(self, nb: int, WallInfo: ti.template()):
        pass

    @ti.kernel
    def CreateCylinder(self, nb: int, WallInfo: ti.template()):
        pass

    @ti.kernel
    def InputFacet(self, nb: int, WallInfo: ti.template()):
        pass
