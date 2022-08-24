import taichi as ti


@ti.data_oriented
class NeighborSearchInCell:
    def __init__(self, domain, gridsize, max_contact_num, max_particle_num, max_wall_num, max_particle_in_cell):
        self.gridR = gridsize
        self.cnum = ti.Vector([int(domain[0] / self.gridR), int(domain[1] / self.gridR), int(domain[2] / self.gridR)])              # Grid Number
        self.cellSum = self.cnum[0] * self.cnum[1] * self.cnum[2]
        self.max_particle_num = max_particle_num

        self.id = ti.field(int, self.cellSum)                                                                        # ID of grids
        self.x = ti.Vector.field(3, float, self.cellSum)                                                             # Position
        self.ParticleInCellNum = ti.field(int, self.cellSum)                                                  
        self.ParticleNeighbor = ti.field(int, (self.cellSum, max_particle_in_cell))                                          
        self.contactPair = ti.field(int, (int(max_contact_num), 2))
        self.contactPos = ti.Vector.field(3, float, (int(max_contact_num), 2))
        self.contact_pair_num, self.contact_P2W_num = ti.field(int, shape=()), ti.field(int, shape=())

        self.target_cell = ti.Vector.field(3, int, 13)
        self.target_cell[0], self.target_cell[1], self.target_cell[2] = ti.Vector([0, 0, -1]), ti.Vector([-1, 0, -1]), ti.Vector([-1, 0, 0])
        self.target_cell[3], self.target_cell[4], self.target_cell[5] = ti.Vector([-1, 0, 1]), ti.Vector([0, -1, -1]), ti.Vector([0, -1, 0])
        self.target_cell[6], self.target_cell[7], self.target_cell[8] = ti.Vector([0, -1, 1]), ti.Vector([-1, -1, -1]), ti.Vector([-1, -1, 0])
        self.target_cell[9], self.target_cell[10], self.target_cell[11], self.target_cell[12]= ti.Vector([-1, -1, 1]), ti.Vector([1, -1, -1]), ti.Vector([1, -1, 0]), ti.Vector([1, -1, -1])

        # ==================================================== Wall ============================================================ #
        self.wallCenter = ti.Vector.field(3, float, max_wall_num)
        self.len = ti.field(float, max_wall_num)
        self.WallInCellNum = ti.field(int, self.cellSum)
        self.WallNeighbor = ti.field(int, (self.cellSum, max_wall_num))

    # ========================================================= #
    #                                                           #
    #                  Get Cell ID & Index                      #
    #                                                           #
    # ========================================================= # 
    @ti.func
    def GetCellIndex(self, nc):
        ig = (nc % (self.cnum[0] * self.cnum[1])) % self.cnum[0]
        jg = (nc % (self.cnum[0] * self.cnum[1])) // self.cnum[0]
        kg = nc // (self.cnum[0] * self.cnum[1])
        return ig, jg, kg

    @ti.func
    def GetCellID(self, i, j, k):
        return int(i + j * self.cnum[0] + k * self.cnum[0] * self.cnum[1])
    
    # ======================================== DEM Cell Initialization ======================================== #
    @ti.kernel
    def CellInit(self):
        for nc in self.id:
            ig, jg, kg = self.GetCellIndex(nc)
            pos = (ti.Vector([ig, jg, kg]) + 0.5) * self.gridR
            self.id[nc] = nc
            self.x[nc] = pos

    @ti.kernel
    def ResetNeighborList(self):
        self.contact_P2W_num[None] = 0
        self.contact_pair_num[None] = 0

    @ti.kernel
    def InitNBSList(self):
        for i in range(self.cellSum):
            for j in range(self.ParticleInCellNum[i]):
                self.ParticleNeighbor[i, j] = -1
        for i in self.ParticleInCellNum:
            self.ParticleInCellNum[i] = 0

    @ti.func
    def GetCellID(self, index):
        return int(index[0] + index[1] * self.cnum[0] + index[2] * self.cnum[0] * self.cnum[1])

    @ti.kernel
    def InsertParticle(self, partList: ti.template()):
        for np in range(partList.particleNum[None]):
            cellID = self.GetCellID(partList.x[np] // self.gridR)
            partList.cellID[np] = cellID
            temp = ti.atomic_add(self.ParticleInCellNum[cellID], 1)
            self.ParticleNeighbor[cellID, temp] = np

    @ti.func
    def UpdateProximityP2P(self, cellID, partList):
        for first in range(self.ParticleInCellNum[cellID]):
            master = self.ParticleNeighbor[cellID, first]
            if self.ParticleInCellNum[cellID] > 1:
                for npc in range(first + 1, self.ParticleInCellNum[cellID]):
                    slave = self.ParticleNeighbor[cellID, npc]
                    pos1, rad1 = partList.x[master], partList.rad[master]
                    pos2, rad2 = partList.x[slave], partList.rad[slave]
                    self.FineSearchP2P(master, slave, pos1, pos2, rad1, rad2)
                
            for cell in ti.static(range(self.target_cell.shape[0])):
                currCellID = cellID + self.GetCellID(self.target_cell[cell])
                if 0 <= currCellID <= self.cellSum:
                    for npnc in range(self.ParticleInCellNum[currCellID]):
                        neighSlave = self.ParticleNeighbor[currCellID, npnc]
                        pos1, rad1 = partList.x[master], partList.rad[master]
                        pos2, rad2 = partList.x[neighSlave], partList.rad[neighSlave]
                        self.FineSearchP2P(master, neighSlave, pos1, pos2, rad1, rad2)

    @ti.func
    def FineSearchP2P(self, end1, end2, pos1, pos2, rad1, rad2):
        if (pos1 - pos2).norm() - rad1 - rad2 < 0.:
            temp = ti.atomic_add(self.contact_pair_num[None], 1)
            self.contactPair[temp, 0] = end1
            self.contactPair[temp, 1] = end2
            self.contactPos[temp, 0] = pos1
            self.contactPos[temp, 1] = pos2

    @ti.kernel
    def FindNeighborP2P(self, dem: ti.template()):
        partList, matList, contList = dem.lp, dem.lm, dem.lc
        for cellID in range(self.cellSum):
            if self.ParticleInCellNum[cellID] > 0:
                self.UpdateProximityP2P(cellID, partList)
    
    # ============================================ Wall ================================================= #
    @ti.kernel
    def InitWallList(self):
        for i in range(self.cellSum):
            for j in range(self.WallInCellNum[i]):
                self.WallNeighbor[i, j] = -1
        for i in self.WallInCellNum:
            self.WallInCellNum[i] = 0

    @ti.func
    def CalDiagonal(self, nw, p1, p2, p3, p4):
        leng = (p1 - p2).norm()
        x = (p1 + p2) / 2.
        if (p2 - p3).norm() > leng:
            leng = (p2 - p3).norm()
            x = (p2 + p3) / 2.
        if (p3 - p4).norm() > leng:
            leng = (p3 - p4).norm()
            x = (p3 + p4) / 2.
        if (p1 - p4).norm() > leng:
            leng = (p1 - p4).norm()
            x = (p1 + p4) / 2.
        self.wallCenter[nw], self.len[nw] =  x, leng

    @ti.kernel
    def SetupWallElementBox(self, wallList: ti.template()):
        for nw in range(wallList.wallNum[None]):
            self.CalDiagonal(nw, wallList.p1[nw], wallList.p2[nw], wallList.p3[nw], wallList.p4[nw])

    @ti.func
    def IsInPlane(self, p1, p2, p3, p4, p):
        inplane = 1
        u = (p1 - p).cross(p2 - p)
        v = (p2 - p).cross(p3 - p)
        w = (p3 - p).cross(p4 - p)
        x = (p4 - p).cross(p1 - p)
        if u.dot(v) < 0. or u.dot(w) < 0. or u.dot(x) < 0.:
            inplane = 0
        return inplane

    @ti.func
    def FindWallCenter(self, wallList, wid):
        return (wallList.p1[wid] + wallList.p2[wid] + wallList.p3[wid] + wallList.p4[wid]) / 4.

    @ti.func
    def PointToFacetDis(self, xp, xw, norm):
        return (xp - xw).dot(norm)

    @ti.func
    def PointProjection(self, xp, dist, norm):
        return xp - dist * norm

    @ti.kernel
    def InsertWall(self, wallList: ti.template()):
        for cellID in range(self.cellSum):
            for nw in range(wallList.wallNum[None]):
                P = self.FindWallCenter(wallList, nw)
                if self.PointToFacetDis(self.x[cellID], P, wallList.norm[nw]) < 0.87 * self.gridR:
                    temp = ti.atomic_add(self.WallInCellNum[cellID], 1)
                    self.WallNeighbor[cellID, temp] = nw
        
    @ti.func
    def FineSearchP2W(self, end1, end2, wallList, partList):
        P = self.FindWallCenter(wallList, end1)
        dist = self.PointToFacetDis(partList.x[end2], P, wallList.norm[end1])
        if 0 < dist < partList.rad[end2]:
            xc = self.PointProjection(partList.x[end2], dist, wallList.norm[end1])
            if self.IsInPlane(wallList.p1[end1], wallList.p2[end1], wallList.p3[end1], wallList.p4[end1], xc):
                temp = ti.atomic_add(self.contact_P2W_num[None], 1)
                self.contactPair[temp, 0] = end1
                self.contactPair[temp, 1] = end2
                self.contactPos[temp, 0] = xc
                self.contactPos[temp, 1] = partList.x[end2]
        elif dist < 0.: print("!! Particle is loacted in inactive side\n")

    @ti.func
    def UpdateProximityP2W(self, cellID, partList, wallList):
        for first in range(self.WallInCellNum[cellID]):
            end1 = self.WallNeighbor[cellID, first]
            for npc in range(self.ParticleInCellNum[cellID]):
                end2 = self.ParticleNeighbor[cellID, npc]
                self.FineSearchP2W(end1, end2, wallList, partList)

    @ti.kernel
    def FindNeighborP2W(self, dem: ti.template()):
        partList, wallList, matList, contList = dem.lp, dem.lw, dem.lm, dem.lc
        for cellID in range(self.cellSum):
            if self.WallInCellNum[cellID] > 0 and self.ParticleInCellNum[cellID] > 0:
                self.UpdateProximityP2W(cellID, partList, wallList)
        self.contact_pair_num[None] = self.contact_P2W_num[None]
