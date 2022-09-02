import taichi as ti
import math


@ti.data_oriented
class NeighborSearchInCell:
    def __init__(self, domain, gridsize, max_contact_num, max_particle_num, max_wall_num, max_particle_in_cell):
        self.gridsize = gridsize
        self.cnum = ti.Vector([math.ceil(domain[0] / self.gridsize), int(domain[1] / self.gridsize), int(domain[2] / self.gridsize)])              # Grid Number
        self.cellSum = self.cnum[0] * self.cnum[1] * self.cnum[2]
        if self.cnum[2] == 0:
            self.cellSum = self.cnum[0] * self.cnum[1]
            if self.cnum[1] == 0:
                self.cellSum = self.cnum[0]

        self.id = ti.field(int, self.cellSum)                                                                        # ID of grids
        self.x = ti.Vector.field(3, float, self.cellSum)  # Position

        self.list_head = ti.field(int, self.cellSum)
        self.list_cur = ti.field(int, self.cellSum)
        self.list_tail = ti.field(int, self.cellSum)

        self.grain_count = ti.field(int, self.cellSum)
        self.column_sum = ti.field(int, (self.cnum[1] * self.cnum[2]))
        self.prefix_sum = ti.field(int, self.cellSum) 
        self.particle_id = ti.field(int, max_particle_num)

        self.contactPair = ti.field(int, (int(max_contact_num), 2))
        self.contactPos = ti.Vector.field(3, float, (int(max_contact_num), 2))
        self.contact_pair_num, self.contact_P2W_num = ti.field(int, shape=()), ti.field(int, shape=())

        self.target_cell = ti.Vector.field(3, int, 13)
        self.target_cell[0], self.target_cell[1], self.target_cell[2] = ti.Vector([0, 0, -1]), ti.Vector([-1, 0, -1]), ti.Vector([-1, 0, 0])
        self.target_cell[3], self.target_cell[4], self.target_cell[5] = ti.Vector([-1, 0, 1]), ti.Vector([0, -1, -1]), ti.Vector([0, -1, 0])
        self.target_cell[6], self.target_cell[7], self.target_cell[8] = ti.Vector([0, -1, 1]), ti.Vector([-1, -1, -1]), ti.Vector([-1, -1, 0])
        self.target_cell[9], self.target_cell[10], self.target_cell[11], self.target_cell[12]= ti.Vector([-1, -1, 1]), ti.Vector([1, -1, -1]), ti.Vector([1, -1, 0]), ti.Vector([1, -1, 1])

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

    @ti.func
    def GetCellIDyz(self, j, k):
        return int(j + k * self.cnum[1])
    
    # ======================================== DEM Cell Initialization ======================================== #
    @ti.kernel
    def CellInit(self):
        for nc in self.id:
            ig, jg, kg = self.GetCellIndex(nc)
            pos = (ti.Vector([ig, jg, kg]) + 0.5) * self.gridsize
            self.id[nc] = nc
            self.x[nc] = pos

    @ti.kernel
    def ResetNeighborList(self):
        self.contact_P2W_num[None] = 0
        self.contact_pair_num[None] = 0

    @ti.kernel
    def InitNBSList(self):
        pass
        '''for i in range(self.cellSum):
            for j in range(self.ParticleInCellNum[i]):
                self.ParticleNeighbor[i, j] = -1
        for i in self.ParticleInCellNum:
            self.ParticleInCellNum[i] = 0'''

    @ti.kernel
    def SumParticles(self, partList: ti.template()):
        for cellID in range(self.cellSum):
            self.grain_count[cellID] = 0

        for np in range(partList.particleNum[None]):
            cellID = self.GetCellID(partList.x[np][0] // self.gridsize, partList.x[np][1] // self.gridsize, partList.x[np][2] // self.gridsize)
            self.grain_count[cellID] += 1
            partList.cellID[np] = cellID

        for celly in range(self.cnum[1]):
            for cellz in range(self.cnum[2]):
                ParticleInRow = 0
                for cellx in range(self.cnum[0]):
                    cellID = self.GetCellID(cellx, celly, cellz)
                    ParticleInRow += self.grain_count[cellID]
                cellIDyz = self.GetCellIDyz(celly, cellz)
                self.column_sum[cellIDyz] = ParticleInRow

    @ti.kernel
    def BoardNeighborList(self, partList: ti.template()):
        ti.loop_config(serialize=True)
        self.prefix_sum[0] = 0
        for cellz in range(self.cnum[2]):
            for celly in range(self.cnum[1]):
                cellID = self.GetCellID(0, celly, cellz)
                cellIDyz = self.GetCellIDyz(celly, cellz)
                if cellID > 0 and cellIDyz > 0:
                    self.prefix_sum[cellID] = self.prefix_sum[cellID - self.cnum[0]] + self.column_sum[cellIDyz - 1]
                   

        for cellx in range(self.cnum[0]):
            for celly in range(self.cnum[1]):
                for cellz in range(self.cnum[2]): 
                    cellID = self.GetCellID(cellx, celly, cellz)

                    if cellx == 0:
                        self.prefix_sum[cellID] += self.grain_count[cellID]
                    else:
                        self.prefix_sum[cellID] = self.prefix_sum[cellID - 1] + self.grain_count[cellID]

                    self.list_head[cellID] = self.prefix_sum[cellID] - self.grain_count[cellID]
                    self.list_cur[cellID] = self.list_head[cellID]
                    self.list_tail[cellID] = self.prefix_sum[cellID]

        for np in range(partList.particleNum[None]):
            cellID = partList.cellID[np]
            grain_location = ti.atomic_add(self.list_cur[cellID], 1)
            self.particle_id[grain_location] = np


    @ti.func
    def FineSearchP2P(self, end1, end2, pos1, pos2, rad1, rad2):
        if (pos1 - pos2).norm() - rad1 - rad2 < 0.:
            temp = ti.atomic_add(self.contact_pair_num[None], 1)
            self.contactPair[temp, 0] = end1
            self.contactPair[temp, 1] = end2
            self.contactPos[temp, 0] = pos1
            self.contactPos[temp, 1] = pos2

    @ti.kernel
    def BoardSearch(self, partList: ti.template()):
        for master in range(partList.particleNum[None]):
            grid_idx = ti.floor(partList.x[master] / self.gridsize, int)

            x_begin = max(grid_idx[0] - 1, 0)
            x_end = min(grid_idx[0] + 2, self.cnum[0])
            y_begin = max(grid_idx[1] - 1, 0)
            y_end = min(grid_idx[1] + 2, self.cnum[1])
            z_begin = max(grid_idx[2] - 1, 0)
            z_end = min(grid_idx[2] + 2, self.cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end): 
                    for neigh_k in range(z_begin, z_end): 
                        cellID = self.GetCellID(neigh_i, neigh_j, neigh_k)
                        
                        for p_idx in range(self.list_head[cellID], self.list_tail[cellID]):
                            slave = self.particle_id[p_idx]
                            if master < slave: 
                                pos1, rad1 = partList.x[master], partList.rad[master]
                                pos2, rad2 = partList.x[slave], partList.rad[slave]
                                self.FineSearchP2P(master, slave, pos1, pos2, rad1, rad2)

    @ti.kernel
    def FindNeighborP2P(self, dem: ti.template()):
        partList, matList, contList = dem.lp, dem.lm, dem.lc
        for cellID in range(self.cellSum):
            if self.ParticleInCellNum[cellID] > 0:
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
                        if 0 <= currCellID <= self.cellSum and currCellID != cellID:
                            for npnc in range(self.ParticleInCellNum[currCellID]):
                                neighSlave = self.ParticleNeighbor[currCellID, npnc]
                                pos1, rad1 = partList.x[master], partList.rad[master]
                                pos2, rad2 = partList.x[neighSlave], partList.rad[neighSlave]
                                self.FineSearchP2P(master, neighSlave, pos1, pos2, rad1, rad2)
    
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
                dist = self.PointToFacetDis(self.x[cellID], P, wallList.norm[nw])
                xc = self.PointProjection(self.x[cellID], dist, wallList.norm[nw])
                if dist < 0.707 * self.gridsize and self.IsInPlane(wallList.p1[nw], wallList.p2[nw], wallList.p3[nw], wallList.p4[nw], xc):
                    temp = ti.atomic_add(self.WallInCellNum[cellID], 1)
                    self.WallNeighbor[cellID, temp] = nw
        
    @ti.func
    def FineSearchP2W(self, end1, end2, wallList, partList):
        P = self.FindWallCenter(wallList, end1)
        dist = self.PointToFacetDis(partList.x[end2], P, wallList.norm[end1])
        if wallList.isactive[end1] == 1:
            if 0 < dist < partList.rad[end2]:
                xc = self.PointProjection(partList.x[end2], dist, wallList.norm[end1])
                if self.IsInPlane(wallList.p1[end1], wallList.p2[end1], wallList.p3[end1], wallList.p4[end1], xc):
                    temp = ti.atomic_add(self.contact_P2W_num[None], 1)
                    self.contactPair[temp, 0] = end1
                    self.contactPair[temp, 1] = end2
                    self.contactPos[temp, 0] = xc
                    self.contactPos[temp, 1] = partList.x[end2]
            #elif dist < 0.: print("!! Particle is loacted in inactive side\n")
        elif wallList.isactive[end1] == 2:
            xc = self.PointProjection(partList.x[end2], dist, wallList.norm[end1])
            if self.IsInPlane(wallList.p1[end1], wallList.p2[end1], wallList.p3[end1], wallList.p4[end1], xc):
                temp = ti.atomic_add(self.contact_P2W_num[None], 1)
                self.contactPair[temp, 0] = end1
                self.contactPair[temp, 1] = end2
                self.contactPos[temp, 0] = xc
                self.contactPos[temp, 1] = partList.x[end2]

    @ti.func
    def UpdateProximityP2W(self, cellID, partList, wallList):
        for first in range(self.ParticleInCellNum[cellID]):
            end2 = self.ParticleNeighbor[cellID, first]
            for npw in range(self.WallInCellNum[cellID]):
                end1 = self.WallNeighbor[cellID, npw]
                if wallList.isactive[end1] >= 1:
                    self.FineSearchP2W(end1, end2, wallList, partList)

            for cell in ti.static(range(self.target_cell.shape[0])):
                currCellID = cellID + self.GetCellID(self.target_cell[cell])
                if 0 <= currCellID <= self.cellSum:
                    for npnw in range(self.WallInCellNum[currCellID]):
                        neighend1 = self.ParticleNeighbor[currCellID, npnw]
                        if wallList.isactive[neighend1] >= 1:
                            self.FineSearchP2W(neighend1, end2, wallList, partList)

    @ti.kernel
    def FindNeighborP2W(self, dem: ti.template()):
        partList, wallList, matList, contList = dem.lp, dem.lw, dem.lm, dem.lc
        for np in range(partList.particleNum[None]):
            for nw in range(wallList.wallNum[None]):
                if wallList.isactive[nw] == 1:
                    self.FineSearchP2W(nw, np, wallList, partList)
        self.contact_pair_num[None] = self.contact_P2W_num[None]
        '''partList, wallList, matList, contList = dem.lp, dem.lw, dem.lm, dem.lc
        for cellID in range(self.cellSum):
            if self.WallInCellNum[cellID] > 0 and self.ParticleInCellNum[cellID] > 0:
                self.UpdateProximityP2W(cellID, partList, wallList)
        self.contact_pair_num[None] = self.contact_P2W_num[None]'''
