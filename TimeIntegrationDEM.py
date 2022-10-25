import taichi as ti
import time
import DEMLib3D.Graphic as graphic
import DEMLib3D.Spying as spy


@ti.data_oriented
class TimeIntegrationMPM:
    def __init__(self, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        self.dem = dem
        self.TIME = TIME
        self.saveTime = saveTime
        self.CFL = CFL
        self.vtkPath = vtkPath
        self.ascPath = ascPath
        self.adaptive = adaptive

    def TurnOnSolver(self, t, step, printNum):
        self.t = t
        self.step = step
        self.printNum = printNum

    def FinalizeSolver(self):
        self.t = 0.
        self.step = 0
        self.printNum = 0

    def UpdateSolver(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        self.TIME = TIME
        self.saveTime = saveTime
        self.CFL = CFL
        self.vtkPath = vtkPath
        self.ascPath = ascPath
        self.adaptive = adaptive


    def Output(self, start):
        print('# Step = ', self.step, '     ', 'Simulation time = ', self.t, '\n')
        graphic.WriteFileVTK_DEM(self.dem.partList, self.printNum, self.vtkPath)
        spy.MonitorDEM(self.t, self.dem.partList, self.printNum, self.ascPath)


#  --------------------------------------- Euler Algorithm ----------------------------------------- #
@ti.data_oriented
class SolverEuler(TimeIntegrationMPM):
    def __init__(self, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
                self.dem.neighborList.InitWallList()
                self.dem.neighborList.SumParticles()
                self.dem.neighborList.BoardNeighborList()
                self.dem.neighborList.BoardSearchP2W()
                self.dem.neighborList.BoardSearchP2P()
                self.dem.neighborList.FineSearchP2W()
                self.dem.neighborList.FineSearchP2P()
                self.dem.Engine.Integration()
            else:
                max_disp = self.dem.partList.ComputeMaximumParticleDisp()
                if max_disp > 0.5 * self.dem.verlet_distance:
                    self.dem.neighborList.InitWallList()
                    self.dem.neighborList.SumParticles()
                    self.dem.neighborList.BoardNeighborList()
                    self.dem.neighborList.BoardSearchP2W()
                    self.dem.neighborList.BoardSearchP2P()
                    self.dem.partList.ResetParicleDisp()
                self.dem.neighborList.FineSearchP2W()
                self.dem.neighborList.FineSearchP2P()
                self.dem.Engine.Integration()

            if self.saveTime - self.t % self.saveTime < self.dem.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.dem.Engine.Reset()
            self.dem.contPair.Reset()

            self.dem.AddParticlesInRun(self.t)

            self.t += self.dem.Dt[None]
            self.step += 1

        print('Physical time = ', time.time() - start)


#  --------------------------------------- Verlet Algorithm ----------------------------------------- #
@ti.data_oriented
class SolverVerlet(TimeIntegrationMPM):
    def __init__(self, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
                self.dem.Engine.IntegrationInit()
                self.dem.Engine.IntegrationPredictor()
                self.dem.neighborList.InitWallList()
                self.dem.neighborList.SumParticles()
                self.dem.neighborList.BoardNeighborList()
                self.dem.neighborList.BoardSearchP2W()
                self.dem.neighborList.BoardSearchP2P()
                self.dem.neighborList.FineSearchP2W()
                self.dem.neighborList.FineSearchP2P()
                self.dem.Engine.IntegrationCorrector()
                self.dem.Engine.SphereRotation()
            else:
                self.dem.Engine.IntegrationPredictor()
                max_disp = self.dem.partList.ComputeMaximumParticleDisp()
                if max_disp > 0.5 * self.dem.verlet_distance:
                    self.dem.neighborList.InitWallList()
                    self.dem.neighborList.SumParticles()
                    self.dem.neighborList.BoardNeighborList()
                    self.dem.neighborList.BoardSearchP2W()
                    self.dem.neighborList.BoardSearchP2P()
                    self.dem.partList.ResetParicleDisp()
                self.dem.neighborList.FineSearchP2W()
                self.dem.neighborList.FineSearchP2P()
                self.dem.Engine.IntegrationCorrector()
                self.dem.Engine.SphereRotation()

            if self.saveTime - self.t % self.saveTime < self.dem.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.dem.Engine.Reset()
            self.dem.contPair.Reset()
 
            self.dem.AddParticlesInRun(self.t)

            self.t += self.dem.Dt[None]
            self.step += 1

        print('Physical time = ', time.time() - start)


@ti.data_oriented
class SolverDEMPM:
    def __init__(self, dempm, dem):
        self.dempm = dempm
        self.dem = dem

    def Flow(self):
        pass


@ti.data_oriented
class FlowEuler(SolverDEMPM):
    def __init__(self, dempm, dem):
        super().__init__(dempm, dem)

    def Time0(self):
        self.dem.neighborList.InitWallList()
        self.dem.neighborList.SumParticles()
        self.dem.neighborList.BoardNeighborList()
        self.dem.neighborList.BoardSearchP2W()
        self.dem.neighborList.BoardSearchP2P()
        self.dem.neighborList.FineSearchP2W()
        self.dem.neighborList.FineSearchP2P()
        self.dempm.DEMEngine.Integration()

    def Flow(self):
        max_disp = self.dem.partList.ComputeMaximumParticleDisp()
        if max_disp > 0.5 * self.dem.verlet_distance:
            self.dem.neighborList.InitWallList()
            self.dem.neighborList.SumParticles()
            self.dem.neighborList.BoardNeighborList()
            self.dem.neighborList.BoardSearchP2W()
            self.dem.neighborList.BoardSearchP2P()
            self.dem.partList.ResetParicleDisp()
        self.dem.neighborList.FineSearchP2W()
        self.dem.neighborList.FineSearchP2P()
        self.dempm.DEMEngine.Integration()
        

@ti.data_oriented
class FlowVerlet(SolverDEMPM):
    def __init__(self, dempm, dem):
        super().__init__(dempm, dem)

    def Time0(self):
        self.dempm.DEMEngine.IntegrationInit()
        self.dempm.DEMEngine.IntegrationPredictor()
        self.dem.neighborList.InitWallList()
        self.dem.neighborList.SumParticles()
        self.dem.neighborList.BoardNeighborList()
        self.dem.neighborList.BoardSearchP2W()
        self.dem.neighborList.BoardSearchP2P()
        self.dem.neighborList.FineSearchP2W()
        self.dem.neighborList.FineSearchP2P()
        self.dempm.DEMEngine.IntegrationCorrector()
        self.dempm.DEMEngine.SphereRotation()

    def Flow(self):
        self.dempm.DEMEngine.IntegrationPredictor()
        max_disp = self.dem.partList.ComputeMaximumParticleDisp()
        if max_disp > 0.5 * self.dem.verlet_distance:
            self.dem.neighborList.InitWallList()
            self.dem.neighborList.SumParticles()
            self.dem.neighborList.BoardNeighborList()
            self.dem.neighborList.BoardSearchP2W()
            self.dem.neighborList.BoardSearchP2P()
            self.dem.partList.ResetParicleDisp()
        self.dem.neighborList.FineSearchP2W()
        self.dem.neighborList.FineSearchP2P()
        self.dempm.DEMEngine.IntegrationCorrector()
        self.dempm.DEMEngine.SphereRotation()
