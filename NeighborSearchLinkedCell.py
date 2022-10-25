import taichi as ti
import DEMLib3D.Aabb as Aabb


@ti.data_oriented
class NeighborSearchLinkedCell:
    def __init__(self, domain, gridsize, rad_max, verletDistance, max_wall_in_cell, max_potential_particle_pairs, max_potential_wall_pairs, partList, wallList, contPair):
        self.gridsize = gridsize
        self.rad_max = rad_max
        self.verletDistance = verletDistance
        self.cnum = ti.ceil(ti.Vector([domain[0] / self.gridsize, domain[1] / self.gridsize, domain[2] / self.gridsize]))              # Grid Number
        if not all(domain // self.gridsize == 0): print("Warning: The computational domain is suggested to be an integer multiple of the grid size\n")
        self.cellSum = self.cnum[0] * self.cnum[1] * self.cnum[2]
        if self.cnum[2] == 0:
            self.cellSum = self.cnum[0] * self.cnum[1]
            if self.cnum[1] == 0:
                self.cellSum = self.cnum[0]
        self.verletDistance = verletDistance

        self.id = ti.field(int, self.cellSum)                                                                        # ID of grids
        self.x = ti.Vector.field(3, float, self.cellSum)                                                             # Position
        self.partList = partList
        self.wallList = wallList
        self.contPair = contPair

        # ==================================================== Particles ============================================================ #
        self.ListHead = ti.field(int, self.cellSum)
        self.ListCur = ti.field(int, self.cellSum)
        self.ListTail = ti.field(int, self.cellSum)

        self.grain_count = ti.field(int, self.cellSum)
        self.column_sum = ti.field(int, (self.cnum[1] * self.cnum[2]))
        self.prefix_sum = ti.field(int, (self.cnum[1] * self.cnum[2])) 
        self.ParticleID = ti.field(int, self.partList.max_particle_num)

        self.potentialListP2P = ti.Struct.field({                                 # List of potential particle list
            "end1": int,                                                        
            "end2": int,                                                                                      
        }, shape=(max_potential_particle_pairs,))
        self.p2p = ti.field(int, ())

        # ==================================================== Wall ============================================================ #
        self.potentialListP2W = ti.Struct.field({                                 # List of potential particle list
            "end1": int,                                                        
            "end2": int,                                                                                                                              
        }, shape=(max_potential_wall_pairs,))
        self.p2w = ti.field(int, ())

        # ==================================================== Neighbor Cell ============================================================ #
        self.target_cell = ti.Vector.field(3, int, 13)
        self.target_cell[0], self.target_cell[1], self.target_cell[2] = ti.Vector([0, 0, -1]), ti.Vector([-1, 0, -1]), ti.Vector([-1, 0, 0])
        self.target_cell[3], self.target_cell[4], self.target_cell[5] = ti.Vector([-1, 0, 1]), ti.Vector([0, -1, -1]), ti.Vector([0, -1, 0])
        self.target_cell[6], self.target_cell[7], self.target_cell[8] = ti.Vector([0, -1, 1]), ti.Vector([-1, -1, -1]), ti.Vector([-1, -1, 0])
        self.target_cell[9], self.target_cell[10], self.target_cell[11], self.target_cell[12]= ti.Vector([-1, -1, 1]), ti.Vector([1, -1, -1]), ti.Vector([1, -1, 0]), ti.Vector([1, -1, 1])

        
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
    def SumParticles(self):
        for cellID in range(self.cellSum):
            self.grain_count[cellID] = 0

        for np in range(self.partList.particleNum[None]):
            cellID = self.GetCellID(self.partList.x[np][0] // self.gridsize, self.partList.x[np][1] // self.gridsize, self.partList.x[np][2] // self.gridsize)
            self.grain_count[cellID] += 1
            self.partList.cellID[np] = cellID

        for cellz in range(self.cnum[2]):
            for celly in range(self.cnum[1]):
                ParticleInRow = 0
                for cellx in range(self.cnum[0]):
                    cellID = self.GetCellID(cellx, celly, cellz)
                    ParticleInRow += self.grain_count[cellID]
                cellIDyz = self.GetCellIDyz(celly, cellz)
                self.column_sum[cellIDyz] = ParticleInRow

    @ti.kernel
    def BoardNeighborList(self):
        _prefix_sum = 0
        for cellz in range(self.cnum[2]):
            for celly in range(self.cnum[1]):
                cellIDyz = self.GetCellIDyz(celly, cellz)
                self.prefix_sum[cellIDyz] = ti.atomic_add(_prefix_sum, self.column_sum[cellIDyz])
                   
        for cellx in range(self.cnum[0]):
            for celly in range(self.cnum[1]):
                for cellz in range(self.cnum[2]): 
                    cellIDyz = self.GetCellIDyz(celly, cellz)
                    cellID = self.GetCellID(cellx, celly, cellz)
                    _prefix_sum_ = ti.atomic_add(self.prefix_sum[cellIDyz], self.grain_count[cellID])
                    
                    self.ListHead[cellID] = _prefix_sum_
                    self.ListCur[cellID] = self.ListHead[cellID]
                    self.ListTail[cellID] = _prefix_sum_ + self.grain_count[cellID]

        for np in range(self.partList.particleNum[None]):
            cellID = self.partList.cellID[np]
            grain_location = ti.atomic_add(self.ListCur[cellID], 1)
            self.ParticleID[grain_location] = np

    @ti.func
    def VerletTable(self, master, neigh_i, neigh_j, neigh_k, state):
        if 0 <= neigh_i <= self.cnum[0] and \
           0 <= neigh_j <= self.cnum[1] and \
           0 <= neigh_k <= self.cnum[2]:

           cellID = self.GetCellID(neigh_i, neigh_j, neigh_k)
           for p_idx in range(self.ListHead[cellID], self.ListTail[cellID]):
                slave = self.ParticleID[p_idx]
                pos1 = self.partList.x[master]
                pos2 = self.partList.x[slave]  
                rad1 = self.partList.rad[master]  
                rad2 = self.partList.rad[slave]
                if master >= slave and state == 1: continue
                if (pos2 - pos1).norm() <= rad1 + rad2 + self.verletDistance: 
                    count_pairs = ti.atomic_add(self.p2p[None], 1)
                    self.potentialListP2P[count_pairs].end1 = ti.min(master, slave)
                    self.potentialListP2P[count_pairs].end2 = ti.max(master, slave)

    @ti.kernel
    def BoardSearchP2P(self):
        self.p2p[None] = 0
        for master in range(self.partList.particleNum[None]):
            grid_idx = ti.floor(self.partList.x[master] / self.gridsize, int)

            '''self.VerletTable(master, grid_idx[0], grid_idx[1], grid_idx[2], state=1)
            self.VerletTable(master, grid_idx[0], grid_idx[1], grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0], grid_idx[1] + 1, grid_idx[2], state=0)
            self.VerletTable(master, grid_idx[0], grid_idx[1] + 1, grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1], grid_idx[2], state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1], grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1] + 1, grid_idx[2], state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1] + 1, grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0] - 1, grid_idx[1], grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0] - 1, grid_idx[1] - 1, grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0], grid_idx[1] - 1, grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1] - 1, grid_idx[2] + 1, state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1] - 1, grid_idx[2], state=0)
            self.VerletTable(master, grid_idx[0] + 1, grid_idx[1] - 1, grid_idx[2] - 1, state=0)'''
            
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
                        for p_idx in range(self.ListHead[cellID], self.ListTail[cellID]):
                            slave = self.ParticleID[p_idx]
                            if master < slave: 
                                count_pairs = ti.atomic_add(self.p2p[None], 1)
                                self.potentialListP2P[count_pairs].end1 = master
                                self.potentialListP2P[count_pairs].end2 = slave

    @ti.kernel
    def FineSearchP2P(self): 
        for p2p in range(self.p2p[None]):
            end1 = self.potentialListP2P[p2p].end1
            end2 = self.potentialListP2P[p2p].end2        
            pos1 = self.partList.x[end1]
            pos2 = self.partList.x[end2]  
            rad1 = self.partList.rad[end1]  
            rad2 = self.partList.rad[end2]   
            if (pos2 - pos1).norm() < rad1 + rad2:     
                self.contPair.ContactPP(end1, end2, pos1, pos2, rad1, rad2, TYPE=0)

    # ============================================ Wall ================================================= #
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
    def FindWallCenter(self, wid):
        return (self.wallList.p1[wid] + self.wallList.p2[wid] + self.wallList.p3[wid] + self.wallList.p4[wid]) / 4.

    @ti.func
    def PointToFacetDis(self, xp, xw, norm):
        return (xp - xw).dot(norm)

    @ti.func
    def PointProjection(self, xp, dist, norm):
        return xp - dist * norm


    @ti.kernel
    def FineSearchP2W(self):
        for p2w in range(self.p2w[None]):
            end1 = self.potentialListP2W[p2w].end1
            end2 = self.potentialListP2W[p2w].end2
            P = self.FindWallCenter(end1)
            dist = self.PointToFacetDis(self.partList.x[end2], P, self.wallList.norm[end1])
            contPoint = self.PointProjection(self.partList.x[end2], dist, self.wallList.norm[end1])
            
            overlap = 0.
            if self.wallList.isactive[end1] == 2:
                overlap = self.partList.rad[end2] - ti.abs(dist)
            elif self.wallList.isactive[end1] == 1 and dist > 0.:
                overlap = self.partList.rad[end2] - dist

            if overlap > 0. and self.IsInPlane(self.wallList.p1[end1], self.wallList.p2[end1], self.wallList.p3[end1], self.wallList.p4[end1], contPoint):
                self.contPair.ContactPW(end1, end2, contPoint, self.partList.x[end2], 0, self.partList.rad[end2], TYPE=1)


@ti.data_oriented
class NeighborSearchLinkedCell1(NeighborSearchLinkedCell):
    def __init__(self, domain, gridsize, rad_max, verletDistance, max_wall_in_cell, max_potential_particle_pairs, max_potential_wall_pairs, partList, wallList, contPair):
        super().__init__(domain, gridsize, rad_max, verletDistance, max_wall_in_cell, max_potential_particle_pairs, max_potential_wall_pairs, partList, wallList, contPair)
        self.WallInCellNum = ti.field(int, self.cellSum)
        self.WallNeighbor = ti.field(int, (self.cellSum, max_wall_in_cell))


    @ti.kernel
    def InitWallList(self):
        for i in self.WallInCellNum:
            self.WallInCellNum[i] = 0

        for cellID in range(self.cellSum):
            for nw in range(self.wallList.wallNum[None]):
                P = self.FindWallCenter(nw)
                dist = self.PointToFacetDis(self.x[cellID], P, self.wallList.norm[nw])
                xc = self.PointProjection(self.x[cellID], dist, self.wallList.norm[nw])
                if dist < 0.7071 * self.gridsize + self.rad_max:
                    temp = ti.atomic_add(self.WallInCellNum[cellID], 1)
                    self.WallNeighbor[cellID, temp] = nw

    @ti.kernel
    def BoardSearchP2W(self):
        self.p2w[None] = 0
        for cellID in range(self.cellSum):
            for nw in range(self.WallInCellNum[cellID]):
                for p_idx in range(self.ListHead[cellID], self.ListTail[cellID]):
                    count_pairs = ti.atomic_add(self.p2w[None], 1)
                    self.potentialListP2W[count_pairs].end1 = self.WallNeighbor[cellID, nw]
                    self.potentialListP2W[count_pairs].end2 = self.ParticleID[p_idx]


@ti.data_oriented
class NeighborSearchLinkedCell2(NeighborSearchLinkedCell):
    def __init__(self, domain, gridsize, rad_max, verletDistance, max_wall_in_cell, max_potential_particle_pairs, max_potential_wall_pairs, partList, wallList, contPair):
        super().__init__(domain, gridsize, rad_max, verletDistance, max_wall_in_cell, max_potential_particle_pairs, max_potential_wall_pairs, partList, wallList, contPair)
        self.xBound = ti.field(float, shape=(max_wall_in_cell, 2))
        self.yBound = ti.field(float, shape=(max_wall_in_cell, 2))
        self.zBound = ti.field(float, shape=(max_wall_in_cell, 2))


    @ti.kernel
    def InitWallList(self):
        for nw in range(self.wallList.wallNum[None]):
            xmin = ti.min(self.wallList.p1[nw][0], self.wallList.p2[nw][0], self.wallList.p3[nw][0], self.wallList.p4[nw][0])
            ymin = ti.min(self.wallList.p1[nw][1], self.wallList.p2[nw][1], self.wallList.p3[nw][1], self.wallList.p4[nw][1])
            zmin = ti.min(self.wallList.p1[nw][2], self.wallList.p2[nw][2], self.wallList.p3[nw][2], self.wallList.p4[nw][2])

            xmax = ti.max(self.wallList.p1[nw][0], self.wallList.p2[nw][0], self.wallList.p3[nw][0], self.wallList.p4[nw][0])
            ymax = ti.max(self.wallList.p1[nw][1], self.wallList.p2[nw][1], self.wallList.p3[nw][1], self.wallList.p4[nw][1])
            zmax = ti.max(self.wallList.p1[nw][2], self.wallList.p2[nw][2], self.wallList.p3[nw][2], self.wallList.p4[nw][2])

            self.xBound[nw, 0], self.xBound[nw, 1] = xmin, xmax
            self.yBound[nw, 0], self.yBound[nw, 1] = ymin, ymax
            self.zBound[nw, 0], self.zBound[nw, 1] = zmin, zmax


    @ti.kernel
    def BoardSearchP2W(self):
        half_length = 0.5 * self.gridsize
        self.p2w[None] = 0
        for cellID in range(self.cellSum):
            for nw in range(self.wallList.wallNum[None]):
                center = self.x[cellID]
                if Aabb.AxisDetection(self.xBound[nw, 0] - self.rad_max, self.xBound[nw, 1] + self.rad_max, center[0] - half_length, center[0] + half_length) and \
                   Aabb.AxisDetection(self.yBound[nw, 0] - self.rad_max, self.yBound[nw, 1] + self.rad_max, center[1] - half_length, center[1] + half_length) and \
                   Aabb.AxisDetection(self.zBound[nw, 0] - self.rad_max, self.zBound[nw, 1] + self.rad_max, center[2] - half_length, center[2] + half_length) and \
                   self.wallList.isactive[nw] != 0:
                    for p_idx in range(self.ListHead[cellID], self.ListTail[cellID]):
                        count_pairs = ti.atomic_add(self.p2w[None], 1)
                        self.potentialListP2W[count_pairs].end1 = nw
                        self.potentialListP2W[count_pairs].end2 = self.ParticleID[p_idx]
