import taichi as ti
import DEM as DEM
import TimeIntegrationDEM as TimeIntegrationDEM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


# Test for Linear contact models // Rotation
if __name__ == "__main__":
    # DEM domain
    dem = DEM.DEM(cmType=0,                                                 # /Contact model type/ 0 for linear model; 1 for hertz model; 2 for linear rolling model; 3 for linear bond model
                  domain=ti.Vector([0.5, 0.5, 45]),                             # domain size
                  periodic=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  algorithm=0,                                              # /Integration scheme/ 0 for Euler; 1 for Verlet; 2 for sympletic
                  gravity=ti.Vector([0., 0., 0.]),                        # Gravity (body force)
                  timeStep=1.e-5,                                           # Time step
                  bodyNum=3,                                               # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  wallNum=1,                                                # Taichi field of material parameters *Number of Wall*
                  searchAlgorithm=2,                                        # /Search Algorithm/ 0 for sorted based; 1 for hash cell; 
                  max_particle_num=3,                                       # the number of particles
                  max_contact_num=3,                                        # the number of contacts
                  )
    
    # Physical parameters of particles
    dem.MatInfo[0].Kn = 1e6                                                 # Contact normal stiffness
    dem.MatInfo[0].Ks = 1e6                                                 # Contact tangential stiffness
    dem.MatInfo[0].Mu = 0.5                                                 # Friction coefficient
    dem.MatInfo[0].localDamping = [0., 0.]                                  # /Local damping/ Translation & Rolling
    dem.MatInfo[0].visDamping = [0., 0.]                                    # /Viscous damping/ Normal & Tangential
    dem.MatInfo[0].ParticleRho = 2650                                       # Particle density

    # DEM body domain
    rad = 0.025
    pos = ti.Vector([0.25, 0.25, 21.975])
    dem.BodyInfo[0].ID = 0                                                  # Body ID
    dem.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[0].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[0].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[0].pos0 = pos                                              # Initial position or center of sphere of Body
    dem.BodyInfo[0].rhi = rad                                                 # Particle radius
    dem.BodyInfo[0].rlo = rad                                                 # Particle radius
    dem.BodyInfo[0].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[0].v0 = ti.Vector([0, 0, 1])                              # Initial velocity
    dem.BodyInfo[0].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[0].fixedW = ti.Vector([0, 0, 0])                           # Fixed angular velocity
    dem.BodyInfo[0].orientation = ti.Vector([0, 0, 1])                      # Initial angular orientation
    dem.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT
    
    pos = ti.Vector([0.25, 0.25, 22.025])
    dem.BodyInfo[1].ID = 1                                                  # Body ID
    dem.BodyInfo[1].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[1].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[1].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[1].pos0 = pos                                              # Initial position or center of sphere of Body
    dem.BodyInfo[1].rhi = rad                                                 # Particle radius
    dem.BodyInfo[1].rlo = rad                                                 # Particle radius
    dem.BodyInfo[1].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[1].v0 = ti.Vector([0, 0, -1])                              # Initial velocity
    dem.BodyInfo[1].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[1].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[1].fixedW = ti.Vector([0, 0, 0])                           # Fixed angular velocity
    dem.BodyInfo[1].orientation = ti.Vector([0, 0, 1])                      # Initial angular orientation
    dem.BodyInfo[1].DT = ti.Vector([0, 1e6, 5])                             # DT

    pos = ti.Vector([0.2933, 0.25, 22])
    dem.BodyInfo[2].ID = 1                                                  # Body ID
    dem.BodyInfo[2].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[2].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[2].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[2].pos0 = pos                                              # Initial position or center of sphere of Body
    dem.BodyInfo[2].rhi = rad                                                 # Particle radius
    dem.BodyInfo[2].rlo = rad                                                 # Particle radius
    dem.BodyInfo[2].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[2].v0 = ti.Vector([-1, 0, 0])                              # Initial velocity
    dem.BodyInfo[2].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[2].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[2].fixedW = ti.Vector([0, 0, 0])                           # Fixed angular velocity
    dem.BodyInfo[2].orientation = ti.Vector([0, 0, 1])                      # Initial angular orientation
    dem.BodyInfo[2].DT = ti.Vector([0, 1e6, 5])                             # DT

    # Create MPM domain
    dem.AddMaterial()
    dem.AddBodies()
    dem.AddWall()
    dem.AddContactList()
    dem.AddNeighborList(GridSize=0.05, max_particle_per_cell=10)

    # Solve
    TIME: float = 10                                                         # Total simulation time
    saveTime: float = 1e-2                                                   # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest1'                                              # VTK output path
    ascPath = './vtkDataTest1/postProcessing'                               # Monitoring data path

    TimeIntegrationDEM.Solver(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

