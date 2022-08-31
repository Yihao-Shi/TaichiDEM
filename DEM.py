import taichi as ti
import DEM_walls as Wall
import DEM_materials as Material
import DEM_particles as Particle
import NeighborSearchProjection as NeighborSearchProjection
import NeighborSearchInCell as NeighborSearchInCell
import NeighborSearchInList as NeighborSearchInList
import ContactBodies as ContactBodies
import TimeIntegrationDEM as Solver

@ti.data_oriented
class DEM:
    def __init__(self, cmType, domain, periodic, algorithm, 
                 gravity, timeStep, bodyNum, wallNum, matNum, 
                 searchAlgorithm, max_particle_num, max_contact_num):
        self.CmType = cmType
        self.Domain = domain
        self.Periodic = periodic
        self.Algorithm = algorithm
        self.Gravity = gravity
        self.Dt = ti.field(float, shape=())
        self.Dt[None] = timeStep
        self.SearchAlgorithm = searchAlgorithm

        self.max_particle_num = int(max_particle_num)
        self.max_contact_num = int(max_contact_num)
        self.bodyNum = int(bodyNum)
        self.wallNum = int(wallNum)
        self.matNum = int(matNum)

        if self.Algorithm == 0:
            print("Integration Scheme: Euler")
        elif self.Algorithm == 1:
            print("Integration Scheme: Verlet")

        self.BodyInfo = ti.Struct.field({                                         # List of body parameters
            "ID": int,                                                            # Body ID
            "Mat": int,                                                           # Material name
            "shapeType": int,                                                     # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for Cuboid; 2 for Tetrahedron 
            "GenerateType": int,                                                  # /Generate type/ 0 for create; 1 for generate; 2 for fill in box;

            # Generate collections of balls
            "pos0": ti.types.vector(3, float),                                    # Initial position 
            "len": ti.types.vector(3, float),                                     # Size of box
            "rlo": float,                                                         # the minimum Radius
            "rhi": float,                                                         # the maximum Radius
            "pnum": int,                                                          # the number of particle in the group

            "fex":ti.types.vector(3, float),                                      # The externel force
            "tex":ti.types.vector(3, float),                                      # The externel torque
            "v0": ti.types.vector(3, float),                                      # Initial velocity
            "orientation": ti.types.vector(3, float),                             # Initial orientation
            "w0": ti.types.vector(3, float),                                      # Initial angular velocity
            "fixedV": ti.types.vector(3, int),                                    # Fixed velocity
            "fixedW": ti.types.vector(3, int),                                    # Fixed angular velocity
            "DT": ti.types.vector(3, float)                                       # DT
        }, shape=(self.bodyNum,))

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
        
        self.MatInfo = ti.Struct.field({                                          # List of material parameters
            "Modulus": float,                                                     # Shear Modulus
            "possion": float,                                                     # Possion ratio
            "Kn": float,                                                          # Hardening
            "Ks": float,                                                          # Cohesion coefficient
            "Mu": float,                                                          # Angle of internal friction
            "Rmu": float,                                                         # Angle of dilatation
            "cohesion": float,                                                    # Bond cohesion
            "ForceLocalDamping": float,                                           # Local Damping
            "TorqueLocalDamping": float,                                          # Local Damping
            "NormalViscousDamping": float,                                        # Viscous Damping
            "TangViscousDamping": float,                                          # Viscous Damping
            "ParticleRho": float                                                  # Particle density !NOT MACRO DENSITY
        }, shape=(self.matNum,))

    
    def AddMaterial(self):
        print('------------------------------------- Material Initialization -----------------------------------------')
        self.lm = Material.DEMMaterial(self.matNum)
        for nm in range(self.MatInfo.shape[0]):
            if self.CmType == 0:
                self.lm.ParticleLinearInit(nm, self.MatInfo)
            elif self.CmType == 1:
                self.lm.ParticleHertzInit(nm, self.MatInfo)
            elif self.CmType == 2:
                self.lm.ParticleLinearRollingInit(nm, self.MatInfo)
            if self.CmType == 3:
                self.lm.ParticleBondInit(nm, self.MatInfo)

    def AddBodies(self):
        print('---------------------------------------- Body Initialization ------------------------------------------')
        self.lp = Particle.DEMParticle(self.max_particle_num)
        for nb in range(self.BodyInfo.shape[0]):
            if self.BodyInfo[nb].GenerateType == 0:
                self.lp.CreateSphere(nb, self.BodyInfo, self.lm, self.Gravity, self.Dt[None])
            elif self.BodyInfo[nb].GenerateType == 1:
                self.lp.GenerateSphere(nb, self.BodyInfo, self.lm, self.Gravity, self.Dt[None])
            elif self.BodyInfo[nb].GenerateType == 2:
                self.lp.FillBallInBox(nb, self.BodyInfo, self.lm, self.Gravity, self.Dt[None])

    def AddWall(self):
        print('---------------------------------------- Wall Initialization ------------------------------------------\n')
        self.lw = Wall.DEMWall(self.wallNum)
        for nw in range(self.WallInfo.shape[0]):
            self.lw.CreatePlane(nw, self.WallInfo)
    
    def AddContactList(self):
        print('---------------------------------------- Contact Initialization ------------------------------------------\n')
        self.lc = ContactBodies.DEMContact(self.max_contact_num, self.CmType)

    def AddNeighborList(self, GridSize, max_particle_per_cell):
        print('---------------------------------------- Neighbor Initialization ------------------------------------------')
        self.ln = None
        if self.SearchAlgorithm == 0:
            self.ln = NeighborSearchProjection.NeighborSearchProjection(self.Domain, GridSize, self.max_contact_num, self.max_particle_num, self.wallNum)
        if self.SearchAlgorithm == 1:
            self.ln = NeighborSearchInList.NeighborSearchInList(self.Domain, GridSize, self.max_contact_num, self.max_particle_num, self.wallNum)
        if self.SearchAlgorithm == 2:
            self.ln = NeighborSearchInCell.NeighborSearchInCell(self.Domain, GridSize, self.max_contact_num, self.max_particle_num, self.wallNum, max_particle_per_cell)
            self.ln.CellInit()
        self.ln.SetupWallElementBox(self.lw)
