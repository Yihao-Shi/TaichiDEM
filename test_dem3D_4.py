from __init__ import *
import DEMLib3D.DEM as DEM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False)


# Test for Linear contact models // Rotation
if __name__ == "__main__":
    # DEM domain
    dem = DEM.DEM(cmType=2,                                                 # /Contact model type/ 0 for linear model; 1 for hertz model; 2 for linear rolling model; 3 for linear bond model
                  domain=ti.Vector([0.31, 0.03, 0.24]),                     # domain size
                  boundary=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  algorithm=1,                                              # /Integration scheme/ 0 for Euler; 1 for Verlet; 2 for sympletic
                  gravity=ti.Vector([0., 0., -9.8]),                        # Gravity (body force)
                  timeStep=5.e-5,                                           # Time step
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  wallNum=5,                                                # Taichi field of material parameters *Number of Wall*
                  searchAlgorithm=1                                         # /Search Algorithm/ 0 for sorted based; 1 for linked cell; 
                  )
    
    # Physical parameters of particles
    dem.MatInfo[0].Kn = 1e3                                                 # Contact normal stiffness
    dem.MatInfo[0].Ks = 1e3                                                 # Contact tangential stiffness
    dem.MatInfo[0].Kr = 100                                                 # Contact rolling stiffness
    dem.MatInfo[0].Kt =100                                                  # Contact twisting stiffness
    dem.MatInfo[0].Mu = 0.5                                                 # Friction coefficient
    dem.MatInfo[0].Rmu = 0.05                                               # Rolling Friction coefficient
    dem.MatInfo[0].Tmu = 0.05                                               # Twisting Friction coefficient
    dem.MatInfo[0].ForceLocalDamping = 0.05                                 # /Local damping/ 
    dem.MatInfo[0].TorqueLocalDamping = 0.05                                # /Local damping/ 
    dem.MatInfo[0].NormalViscousDamping = 0.2                               # /Viscous damping/ 
    dem.MatInfo[0].TangViscousDamping = 0.2                                 # /Viscous damping/ 
    dem.MatInfo[0].ParticleRho = 957                                        # Particle density

    # DEM body domain
    dem.BodyInfo[0].ID = 0                                                  # Body ID
    dem.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[0].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[0].GenerateType = 1                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box
    dem.BodyInfo[0].pos0 = ti.Vector([0.14, 0., 0.05])                        # Initial position or center of sphere of Body
    dem.BodyInfo[0].len = ti.Vector([0.03, 0.03, 0.03])                     # Initial position or center of sphere of Body
    dem.BodyInfo[0].rhi = 0.001836                                          # Particle radius
    dem.BodyInfo[0].rlo = 0.001836                                          # Particle radius
    dem.BodyInfo[0].pnum = 300                                              # The number of particle in the group
    dem.BodyInfo[0].v0 = ti.Vector([0, 0, 0])                               # Initial velocity
    dem.BodyInfo[0].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[0].fixedW = ti.Vector([0, 0, 0])                           # Fixed angular velocity
    dem.BodyInfo[0].orientation = ti.Vector([0, 0, 1])                      # Initial orientation
    dem.BodyInfo[0].DT = ti.Vector([0, 0.3, 3.0])                           # DT

    dem.WallInfo[0].ID = 0                                                  # Body ID
    dem.WallInfo[0].Mat = 0                                                 # Material Name of Body
    dem.WallInfo[0].point1 = ti.Vector([0., 0., 0.])                        # The vertex of the wall
    dem.WallInfo[0].point2 = ti.Vector([0.31, 0., 0.])                      # The vertex of the wall
    dem.WallInfo[0].point3 = ti.Vector([0.31, 0.03, 0.])                    # The vertex of the wall
    dem.WallInfo[0].point4 = ti.Vector([0., 0.03, 0.])                      # The vertex of the wall
    dem.WallInfo[0].norm = ti.Vector([0, 0, 1])                             # The norm of the wall

    dem.WallInfo[1].ID = 1                                                  # Body ID
    dem.WallInfo[1].Mat = 0                                                 # Material Name of Body
    dem.WallInfo[1].point1 = ti.Vector([0., 0., 0.])                        # The vertex of the wall
    dem.WallInfo[1].point2 = ti.Vector([0., 0.03, 0.])                      # The vertex of the wall
    dem.WallInfo[1].point3 = ti.Vector([0., 0.03, 0.24])                    # The vertex of the wall
    dem.WallInfo[1].point4 = ti.Vector([0., 0., 0.24])                      # The vertex of the wall
    dem.WallInfo[1].norm = ti.Vector([1, 0, 0])                             # The norm of the wall

    dem.WallInfo[2].ID = 2                                                  # Body ID
    dem.WallInfo[2].Mat = 0                                                 # Material Name of Body
    dem.WallInfo[2].point1 = ti.Vector([0.31, 0., 0.])                      # The vertex of the wall
    dem.WallInfo[2].point2 = ti.Vector([0.31, 0.03, 0.])                    # The vertex of the wall
    dem.WallInfo[2].point3 = ti.Vector([0.31, 0.03, 0.24])                  # The vertex of the wall
    dem.WallInfo[2].point4 = ti.Vector([0.31, 0., 0.24])                    # The vertex of the wall
    dem.WallInfo[2].norm = ti.Vector([-1, 0, 0])                            # The norm of the wall

    dem.WallInfo[3].ID = 3                                                  # Body ID
    dem.WallInfo[3].Mat = 0                                                 # Material Name of Body
    dem.WallInfo[3].point1 = ti.Vector([0., 0., 0.])                        # The vertex of the wall
    dem.WallInfo[3].point2 = ti.Vector([0.31, 0., 0.])                      # The vertex of the wall
    dem.WallInfo[3].point3 = ti.Vector([0.31, 0., 0.24])                    # The vertex of the wall
    dem.WallInfo[3].point4 = ti.Vector([0., 0., 0.24])                      # The vertex of the wall
    dem.WallInfo[3].norm = ti.Vector([0, 1, 0])                             # The norm of the wall

    dem.WallInfo[4].ID = 4                                                  # Body ID
    dem.WallInfo[4].Mat = 0                                                 # Material Name of Body
    dem.WallInfo[4].point1 = ti.Vector([0., 0.03, 0.])                      # The vertex of the wall
    dem.WallInfo[4].point2 = ti.Vector([0.31, 0.03, 0.])                    # The vertex of the wall
    dem.WallInfo[4].point3 = ti.Vector([0.31, 0.03, 0.24])                  # The vertex of the wall
    dem.WallInfo[4].point4 = ti.Vector([0., 0.03, 0.24])                    # The vertex of the wall
    dem.WallInfo[4].norm = ti.Vector([0, -1, 0])                            # The norm of the wall

   
    # Create MPM domain
    dem.AddContactModel()
    dem.AddBodies(max_particle_num=4000)
    dem.AddWall(max_facet_num=5)
    dem.AddContactPair(max_contact_num=23993)
    dem.AddNeighborList(multiplier=4, max_potential_particle_pairs=150000, max_potential_wall_pairs=50000)

    # Solve
    TIME: float = 5                                                         # Total simulation time
    saveTime: float = 0.05                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest6'                                              # VTK output path
    ascPath = './vtkDataTest6/postProcessing'                               # Monitoring data path

    dem.Solver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)