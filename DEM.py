import taichi as ti
import DEMLib3D.DEMWalls as Wall
import DEMLib3D.ContactModels as ContactModels
import DEMLib3D.DEMParticles as Particle
import DEMLib3D.NeighborSearchSorted as NeighborSearchSorted
import DEMLib3D.NeighborSearchLinkedCell as NeighborSearchLinkedCell
import DEMLib3D.ContactPairs as ContactPairs
import DEMLib3D.DEMEngine as Engine
import DEMLib3D.TimeIntegrationDEM as TimeIntegrationDEM

@ti.data_oriented
class DEM:
    def __init__(self, cmType, domain, boundary, algorithm, 
                 gravity, timeStep, bodyNum, wallNum, matNum, searchAlgorithm):
        self.CmType = cmType
        self.Domain = domain
        self.Boundary = boundary
        self.Algorithm = algorithm
        self.Gravity = gravity
        self.Dt = ti.field(float, shape=())
        self.Dt[None] = timeStep
        self.SearchAlgorithm = searchAlgorithm

        self.bodyNum = int(bodyNum)
        self.wallNum = int(wallNum)
        self.matNum = int(matNum)

        self.MainLoop = None
        
        if self.bodyNum > 0:
            self.BodyInfo = ti.Struct.field({                                         # List of body parameters
                "ID": int,                                                            # Body ID
                "Mat": int,                                                           # Material name
                "shapeType": int,                                                     # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for Cuboid; 2 for Tetrahedron 
                "GenerateType": int,                                                  # /Generate type/ 0 for create; 1 for generate; 2 for distribute; 3 for fill in box;

                "pos0": ti.types.vector(3, float),                                    # Initial position 
                "len": ti.types.vector(3, float),                                     # Size of box
                "rlo": float,                                                         # the minimum Radius
                "rhi": float,                                                         # the maximum Radius
                "pnum": int,                                                          # the number of particle in the group
                "VoidRatio": float,                                                   # The macro void ratio of granular assembly

                "fex":ti.types.vector(3, float),                                      # The externel force
                "tex":ti.types.vector(3, float),                                      # The externel torque
                "v0": ti.types.vector(3, float),                                      # Initial velocity
                "orientation": ti.types.vector(3, float),                             # Initial orientation
                "w0": ti.types.vector(3, float),                                      # Initial angular velocity
                "fixedV": ti.types.vector(3, int),                                    # Fixed velocity
                "fixedW": ti.types.vector(3, int),                                    # Fixed angular velocity
                "DT": ti.types.vector(3, float)                                       # DT
            }, shape=(self.bodyNum,))
        
        if self.wallNum > 0:
            self.WallInfo = ti.Struct.field({                                         # List of wall parameters
                "ID": int,                                                            # Body ID
                "Mat": int,                                                           # Material name
                "point1": ti.types.vector(3, float),                                  # The vertex of the wall
                "point2": ti.types.vector(3, float),                                  # The vertex of the wall
                "point3": ti.types.vector(3, float),                                  # The vertex of the wall
                "point4": ti.types.vector(3, float),                                  # The vertex of the wall
                "norm": ti.types.vector(3, float),                                    # The wall normal (actived side)
                "fex":ti.types.vector(3, float),                                      # The externel force
                "tex":ti.types.vector(3, float),                                      # The externel torque
                "v0": ti.types.vector(3, float),                                      # Initial velocity
                "w0": ti.types.vector(3, float),                                      # Initial angular velocity
                "fixedV": ti.types.vector(3, int),                                    # Fixed velocity
                "fixedW": ti.types.vector(3, int),                                    # Fixed angular velocity
                "DT": ti.types.vector(3, float)                                       # DT
                }, shape=(self.wallNum,))
        
        if matNum > 0:
            self.MatInfo = ti.Struct.field({                                          # List of material parameters
                "ParticleRho": float,                                                 # Particle density 
                "Modulus": float,                                                     # Shear Modulus
                "possion": float,                                                     # Possion ratio
                "Kn": float,                                                          # Hardening
                "Ks": float,                                                          # Cohesion coefficient
                "Kr": float,                                                          # Hardening
                "Kt": float,                                                          # Cohesion coefficient
                "Mu": float,                                                          # Angle of internal friction
                "Rmu": float,                                                         # Angle of dilatation
                "Tmu": float,                                                         # Angle of dilatation
                "cohesion": float,                                                    # Bond cohesion
                "ForceLocalDamping": float,                                           # Local Damping
                "TorqueLocalDamping": float,                                          # Local Damping
                "NormalViscousDamping": float,                                        # Viscous Damping
                "TangViscousDamping": float,                                          # Viscous Damping
                "Restitution": float                                                  # Restitution coefficient
            }, shape=(self.matNum,))

    
    def AddContactModel(self):
        print('------------------------ ContactModels Initialization ------------------------')
        if ti.static(self.CmType == 0):
            self.contModel = ContactModels.LinearContactModel(self.matNum, self.Dt[None])
        elif ti.static(self.CmType == 1):
            self.contModel = ContactModels.HertzMindlinContactModel(self.matNum, self.Dt[None])
        elif ti.static(self.CmType == 2):
            self.contModel = ContactModels.LinearRollingResistanceContactModel(self.matNum, self.Dt[None])
        elif ti.static(self.CmType == 3):
            self.contModel = ContactModels.LinearBondContactModel(self.matNum, self.Dt[None])
        elif ti.static(self.CmType == 4):
            self.contModel = ContactModels.LinearParallelBondContactModel(self.matNum, self.Dt[None])
        for nm in range(self.MatInfo.shape[0]):
            self.contModel.ParticleMaterialInit(nm, self.MatInfo)

    def AddBodies(self, max_particle_num):
        print('------------------------ Body Initialization ------------------------')
        self.max_particle_num = max_particle_num
        self.isClump = 0
        self.partList = Particle.DEMParticle(self.Domain, max_particle_num, self.contModel, self.Gravity)
        if self.bodyNum > 0:
            for nb in range(self.BodyInfo.shape[0]):
                if ti.static(self.BodyInfo[nb].GenerateType == 0):
                    self.partList.CreateSphere(nb, self.BodyInfo)
                elif ti.static(self.BodyInfo[nb].GenerateType == 1):
                    self.partList.GenerateSphere(nb, self.BodyInfo)
                elif ti.static(self.BodyInfo[nb].GenerateType == 2):
                    self.partList.DistributeSphere(nb, self.BodyInfo)
                elif ti.static(self.BodyInfo[nb].GenerateType == 3):
                    self.partList.FillBallInBox(nb, self.BodyInfo)
                if ti.static(self.BodyInfo[nb].shapeType > 0): self.isClump = 1
        self.Rad_max = self.partList.FindMaxRadius()

    def AddWall(self, max_facet_num):
        print('------------------------ Wall Initialization ------------------------')
        self.max_facet_num = max_facet_num
        self.wallList = Wall.DEMWall(self.max_facet_num)
        if self.wallNum > 0:
            for nw in range(self.WallInfo.shape[0]):
                self.wallList.CreatePlane(nw, self.WallInfo)
    
    def AddContactPair(self, max_contact_num):
        print('------------------------ Contact Pair Initialization ------------------------')
        if ti.static(self.CmType == 0):
            self.contPair = ContactPairs.Linear(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 1):
            self.contPair = ContactPairs.HertzMindlin(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 2):
            self.contPair = ContactPairs.LinearRollingResistance(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 3):
            self.contPair = ContactPairs.LinearContactBond(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 4):
            self.contPair = ContactPairs.LinearParallelBond(max_contact_num, self.partList, self.wallList, self.contModel)

    def AddNeighborList(self, multiplier, max_potential_particle_pairs, max_potential_wall_pairs):
        print('------------------------ Neighbor Initialization ------------------------')
        self.multiplier = multiplier
        self.verlet_distance = (multiplier - 2) * self.Rad_max
        self.DEMGridSize = self.multiplier * self.Rad_max 
        if ti.static(self.SearchAlgorithm == 0):
            print("Neighbor Searching Algorithm: Axis-sligned Bounding Box\n")
            self.neighborList = NeighborSearchSorted.NeighborSearchSorted(self.Domain, max_potential_particle_pairs, max_potential_wall_pairs, self.partList, self.wallList, self.contPair)
        if ti.static(self.SearchAlgorithm == 1):
            print("Neighbor Searching Algorithm: Linked Cell\n")
            self.neighborList = NeighborSearchLinkedCell.NeighborSearchLinkedCell2(self.Domain, self.DEMGridSize, self.Rad_max, self.verlet_distance, self.max_facet_num, max_potential_particle_pairs, max_potential_wall_pairs, self.partList, self.wallList, self.contPair)
            self.neighborList.CellInit()

    def AddParticlesInRun(self, t):
        for nb in range(self.BodyInfo.shape[0]):
            if t > 0. and t % self.BodyInfo[nb].DT[1] < self.Dt[None] and self.BodyInfo[nb].DT[0] <= t <= self.BodyInfo[nb].DT[2]:
                print('------------------------ Add Body  ------------------------')
                if ti.static(self.BodyInfo[nb].GenerateType == 0):
                    self.partList.CreateSphere(nb, self.BodyInfo)
                elif ti.static(self.BodyInfo[nb].GenerateType == 1):
                    self.partList.GenerateSphere(nb, self.BodyInfo)
                elif ti.static(self.BodyInfo[nb].GenerateType == 2):
                    self.partList.DistributeSphere(nb, self.BodyInfo)
                elif ti.static(self.BodyInfo[nb].GenerateType == 3):
                    self.partList.FillBallInBox(nb, self.BodyInfo)
            self.Rad_max = self.partList.FindMaxRadius()


    # ============================================== Solver ================================================= #
    def Solver(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        print('------------------------ DEM Solver ------------------------')
        if self.MainLoop:
            self.MainLoop.UpdateSolver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
            self.MainLoop.Solver()
        else:
            if ti.static(self.Algorithm == 0):
                if self.isClump == 1: 
                    print("Clump particles must choose Verlet Integration Scheme")
                    assert 0
                print("Integration Scheme: Euler\n")
                self.Engine = Engine.Euler(self.Gravity, self.Dt[None], self.partList, self.contModel)
                self.MainLoop = TimeIntegrationDEM.SolverEuler(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 1):
                print("Integration Scheme: Verlet\n")
                self.Engine = Engine.Verlet(self.Gravity, self.Dt[None], self.partList, self.contModel)
                self.MainLoop = TimeIntegrationDEM.SolverVerlet(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 3):
                print("Integration Scheme: RungeKutta\n")
                self.Engine = Engine.RungeKutta(self.Gravity, self.Dt[None], self.partList, self.contModel)
                self.MainLoop = TimeIntegrationDEM.SolverVerlet(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
