import taichi as ti


@ti.data_oriented
class NeighborSearch:
    def __init__(self, max_particle_num, max_contact_num):
        self.n, self.q = GreatestPowderOfTwo(2 * max_particle_num)
        self.Bound = ti.Vector.field(3, float, self.n)
        self.contactPair = ti.Vector.field(3, int, int(max_contact_num))
        self.contact_pair_num = ti.field(int, shape=())
        self.step = ti.field(int, 2)
    
    @ti.kernel
    def ResetNeighborList(self):
        for i in self.Bound:
            self.Bound[i] = ti.Vector([0., 0., 0.])
        for i in self.contactPair:
            self.contactPair[i] = ti.Vector([0, 0, -1])
        self.contact_pair_num[None] = 0

    @ti.func
    def CompAndSwap(self, field, index1, index2, updown):
        if xor(field[index1][1] > field[index2][1], updown):
            temp = field[index1]
            field[index1] = field[index2]
            field[index2] = temp

    @ti.kernel
    def BitonicSort(self):
        for i in range(self.n // 2):
            halfstep = self.step[1] // 2
            index1 = i + (i // (halfstep)) * halfstep
            index2 = index1 + halfstep
            updown = (i * 2 // self.step[0]) & 1  
            self.CompAndSwap(self.Bound, index1, index2, updown)

    def SortInXAxis(self):
        for i in range(1, self.q + 1):
            self.step[0] = 2 ** i
            for j in range(i):
                self.step[1] = 2 ** (i - j)
                self.BitonicSort()

    @ti.kernel
    def AddParticles(self, partList: ti.template()):
        for np in range(partList.particleNum[None]):
            ID = partList.ID[np]
            x = partList.x[np][0]
            rad = partList.rad[np]
            self.Bound[2 * np] = ti.Vector([ID, x - rad, 0])
            self.Bound[2 * np + 1] = ti.Vector([ID, x + rad, 1])
        for np in range(2*partList.particleNum[None], self.n):
            self.Bound[np] = ti.Vector([-1, -1, -1])

    @ti.func
    def FineSearchBall(self, end1, end2, pos1, pos2, rad1, rad2):
        if (pos1 - pos2).norm() - rad1 - rad2 < 0.:
            temp = ti.atomic_add(self.contact_pair_num[None], 1)
            self.contactPair[temp] = [end1, end2, 0]

    @ti.func
    def LowBound(self, x, rad):
        xlo = x - rad
        return xlo

    @ti.func
    def UpBound(self, x, rad):
        xhi = x + rad
        return xhi

    @ti.func
    def SubAxisDetection(self, end1, end2, partList):
        f1 = self.LowBound(partList.x[end2][1], partList.rad[end2]) <= self.LowBound(partList.x[end1][1], partList.rad[end1]) <= self.UpBound(partList.x[end2][1], partList.rad[end2])
        f2 = self.LowBound(partList.x[end2][1], partList.rad[end2]) <= self.UpBound(partList.x[end1][1], partList.rad[end1]) <= self.UpBound(partList.x[end2][1], partList.rad[end2])
        f3 = self.LowBound(partList.x[end2][2], partList.rad[end2]) <= self.LowBound(partList.x[end1][2], partList.rad[end1]) <= self.UpBound(partList.x[end2][2], partList.rad[end2])
        f4 = self.LowBound(partList.x[end2][2], partList.rad[end2]) <= self.UpBound(partList.x[end1][2], partList.rad[end1]) <= self.UpBound(partList.x[end2][2], partList.rad[end2])
        return ((f1 or f2) and (f3 or f4))

    @ti.func
    def UpdateProximity(self, end1, end2, partList):
        pos1, pos2 = partList.x[end1], partList.x[end2]
        rad1, rad2 = partList.rad[end1], partList.rad[end2]
        if self.SubAxisDetection(end1, end2, partList):
            self.FineSearchBall(end1, end2, pos1, pos2, rad1, rad2)

    @ti.kernel
    def BoardPhaseCollision(self, partList: ti.template(), end: ti.template()):
        for i in range(self.n - 1):
            if end[i][0] >= 0 and end[i][2] == 0: 
                for j in range(i + 1, self.n):
                    if end[i][0] == end[j][0]: break
                    if end[j][2] == 0: 
                        if end[i][0] < end[j][0]:
                            self.UpdateProximity(int(end[i][0]), int(end[j][0]), partList)
                        elif end[i][0] > end[j][0]:
                            self.UpdateProximity(int(end[j][0]), int(end[i][0]), partList)

    def SweepAndPruneRun(self, dem):
        wallList, partList, contList, matList = dem.lw, dem.lp, dem.lc, dem.lm
        self.AddParticles(partList)
        self.SortInXAxis()
        #ti._kernels.parallel_sort(key, values)
        self.BoardPhaseCollision(partList, self.Bound)
        self.ParticleWallDetection(wallList, partList)
        contList.ContactSetup(self.contactPair, self.contact_pair_num[None], partList, wallList, matList)

    @ti.kernel
    def ParticleWallDetection(self, wallList: ti.template(), partList: ti.template()):
        for np in range(partList.particleNum[None]):
            pos1, rad1 = partList.x[np], partList.rad[np]
            for nw in range(wallList.wallNum[None]):
                pos2 = wallList.x[nw]
                norm = wallList.norm[nw]
                xw = pos1 - (pos1 - pos2).dot(norm) * norm
                if (pos1 - xw).norm() - rad1 < 0.:
                    temp = ti.atomic_add(self.contact_pair_num[None], 1)
                    self.contactPair[temp] = [end1, end2, 1]
