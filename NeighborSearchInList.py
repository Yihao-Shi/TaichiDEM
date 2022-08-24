import taichi as ti
import math


@ti.data_oriented
class NeighborSearchInList:
    def __init__(self, domain, gridsize, max_contact_num, max_particle_num, max_wall_num):
        self.gridR = gridsize
        self.cnum = ti.Vector([math.ceil(domain[0] / self.gridR), math.ceil(domain[1] / self.gridR), math.ceil(domain[2] / self.gridR)])              # Grid Number
        self.cellSum = self.cnum[0] * self.cnum[1] * self.cnum[2]
        self.max_particle_num = max_particle_num

        self.id = ti.field(int, self.cellSum)                                                                        # ID of grids
        self.x = ti.Vector.field(2, float, self.cellSum)                                                             # Position
        self.HeadparticleInCell = ti.field(int, self.cellSum)                                                        # Header List
        self.NextparticleInCell = ti.field(int, int(max_particle_num + 2))                                           # Next List
        self.contactPair = ti.field(int, (int(max_contact_num), 2))
        self.contactPos = ti.Vector.field(3, float, (int(max_contact_num), 2))
        self.contact_pair_num, self.contact_P2W_num = ti.field(int, shape=()), ti.field(int, shape=())

        self.neighbor_cell = ti.Vector.field(3, int, 13)
        self.neighbor_cell[0], self.neighbor_cell[1], self.neighbor_cell[2] = ti.Vector([0, 0, -1]), ti.Vector([-1, 0, -1]), ti.Vector([-1, 0, 0])
        self.neighbor_cell[3], self.neighbor_cell[4], self.neighbor_cell[5] = ti.Vector([-1, 0, 1]), ti.Vector([0, -1, -1]), ti.Vector([0, -1, 0])
        self.neighbor_cell[6], self.neighbor_cell[7], self.neighbor_cell[8] = ti.Vector([0, -1, 1]), ti.Vector([-1, -1, -1]), ti.Vector([-1, -1, 0])
        self.neighbor_cell[9], self.neighbor_cell[10], self.neighbor_cell[11], self.neighbor_cell[12]= ti.Vector([-1, -1, 1]), ti.Vector([1, -1, -1]), ti.Vector([1, -1, 0]), ti.Vector([1, -1, -1])

        # ================================================= Wall Parameters ========================================================= #
        self.wallCenter = ti.Vector.field(3, float, max_wall_num)
        self.len = ti.field(float, max_wall_num)


    @ti.kernel
    def ResetNeighborList(self):
        self.contact_P2W_num[None] = 0
        self.contact_pair_num[None] = 0

    @ti.kernel
    def InitNBSList(self):
        for i in self.HeadparticleInCell:
            self.HeadparticleInCell[i] = -1
        for i in self.NextparticleInCell:
            self.NextparticleInCell[i] = -1

    @ti.func
    def GetCellID(self, index):
        return int(index[0] + index[1] * self.cnum[0] + index[2] * self.cnum[0] * self.cnum[1])

    @ti.kernel
    def InsertParticle(self, partList: ti.template()):
        for np in range(partList.particleNum[None]):
            cellID = self.GetCellID(partList.x[np] // self.gridR)
            partList.cellID[np] = cellID
            self.NextparticleInCell[np] = self.HeadparticleInCell[cellID]
            self.HeadparticleInCell[cellID] = np
    
    @ti.func
    def FirstParticleDEM(self, ID, cellID):
        self.NextparticleInCell[ID] = self.HeadparticleInCell[cellID]

    @ti.func
    def UpdateProximityP2P(self, cellID, partList):
        firstID = self.max_particle_num 
        self.FirstParticleDEM(firstID, cellID)
        while self.NextparticleInCell[firstID] != -1:
            master = self.NextparticleInCell[firstID]
            nextID = master
            while self.NextparticleInCell[nextID] != -1:
                slave = self.NextparticleInCell[nextID]
                pos1, rad1 = partList.x[master], partList.rad[master]
                pos2, rad2 = partList.x[slave], partList.rad[slave]
                self.FineSearchP2P(master, slave, pos1, pos2, rad1, rad2)
                nextID = slave
            
            for cell in range(self.neighbor_cell.shape[0]):
                if self.GetCellID(self.neighbor_cell[cell]) != 0:
                    currCellID = cellID + self.GetCellID(self.neighbor_cell[cell])
                    if 0 <= currCellID <= self.cellSum:
                        neighNextID = self.max_particle_num + 1
                        self.FirstParticleDEM(neighNextID, currCellID)
                        while self.NextparticleInCell[neighNextID] != -1:
                            neighSlave = self.NextparticleInCell[neighNextID]
                            pos1, rad1 = partList.x[master], partList.rad[master]
                            pos2, rad2 = partList.x[neighSlave], partList.rad[neighSlave]
                            self.FineSearchP2P(master, neighSlave, pos1, pos2, rad1, rad2)
                            neighNextID = neighSlave

            firstID = master

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
            if self.HeadparticleInCell[cellID] != -1:
                self.UpdateProximityP2P(cellID, partList)
    
    # ============================================ Wall ================================================= #
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
    def FineSearchP2W(self, end1, end2, wallList, partList):
        P = (wallList.p1[end1] + wallList.p2[end1] + wallList.p3[end1] + wallList.p4[end1]) / 4.
        dist = (partList.x[end2] - P).dot(wallList.norm[end1])
        if 0 < dist < partList.rad[end2]:
            xc = partList.x[end2] - dist * wallList.norm[end1]
            if self.IsInPlane(wallList.p1[end1], wallList.p2[end1], wallList.p3[end1], wallList.p4[end1], xc):
                temp = ti.atomic_add(self.contact_P2W_num[None], 1)
                self.contactPair[temp, 0] = end1
                self.contactPair[temp, 1] = end2
                self.contactPos[temp, 0] = xc
                self.contactPos[temp, 1] = partList.x[end2]
        elif dist < 0.: print("!! Particle is loacted in inactive side\n")

    @ti.kernel
    def FindNeighborP2W(self, dem: ti.template()):
        partList, wallList, matList, contList = dem.lp, dem.lw, dem.lm, dem.lc
        for np in range(partList.particleNum[None]):
            for nw in range(wallList.wallNum[None]):
                if wallList.isactive[nw] == 1:
                    self.FineSearchP2W(nw, np, wallList, partList)
        self.contact_pair_num[None] = self.contact_P2W_num[None]
