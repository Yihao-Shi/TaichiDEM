import taichi as ti
import time
import EulerAlgorithm as EulerAlgorithm
import VerletAlgorithm as VerletAlgorithm
import Graphic as graphic
import Spying as spy


def Output(t, step, printNum, dem, start, vtkPath, ascPath):
    print('------------------------ Time Step = ', step, '------------------------')
    print('Simulation time = ', t)
    print('Time step = ', dem.Dt[None])
    print('Physical time = ', time.time() - start)
    graphic.WriteFileVTK_DEM(dem.lp, printNum, vtkPath)
    spy.MonitorDEM(t, dem.lp, printNum, ascPath)
    print('------------------------------- Running --------------------------------')


# Solvers
def Solver(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    if dem.Algorithm == 0:
        SolverEuler(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif dem.Algorithm == 1:
        SolverVerlet(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


#  --------------------------------------- Euler Algorithm ----------------------------------------- #
def SolverEuler(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    st = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, dem, st, vtkPath, ascPath)
            printNum += 1
        
        EulerAlgorithm.UpdatePosition(dem)
        EulerAlgorithm.NeighborSearching(dem)
        EulerAlgorithm.ContactCal(dem)
        EulerAlgorithm.UpdateVelocity(dem)
        EulerAlgorithm.UpdateAngularVelocity(dem, t)
        
        if t % saveTime < dem.Dt[None]:
            Output(t, step, printNum, dem, st, vtkPath, ascPath)
            printNum += 1

        t += dem.Dt[None]
        step += 1


#  --------------------------------------- Verlet Algorithm ----------------------------------------- #
def SolverVerlet(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    st = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, dem, st, vtkPath, ascPath)
            printNum += 1

        VerletAlgorithm.UpdateQuaternion(dem)
        VerletAlgorithm.NeighborSearching(dem)
        VerletAlgorithm.ContactCal(dem)
        VerletAlgorithm.UpdateVelocity(dem)
        VerletAlgorithm.UpdateAngularVelocity(dem)
        VerletAlgorithm.UpdatePosition(dem)

 
        if t % saveTime < dem.Dt[None]:
            Output(t, step, printNum, dem, st, vtkPath, ascPath)
            printNum += 1

        t += dem.Dt[None]
        step += 1
