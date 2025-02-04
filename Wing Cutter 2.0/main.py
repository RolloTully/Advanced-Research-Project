import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
from tqdm import tqdm
from matplotlib import cm
from skopt import gp_minimize
from scipy.stats import qmc
plt.rcParams["font.family"] = "Times New Roman"

'''
Questions to be answered.

What is the surface energy of the foil.

What is the program, How does it work.
'''
class Trajectory(object):
    def __init__(self, surface, wire_trajectory, wire_velocity = []):
        self.surface_panels = surface
        self.wire_trajectory_panels = wire_trajectory
        self.wire_velocity = wire_velocity
        self.critical_surface_energy = 1 #<<<<<<-----------This needs to be calibrated from experiemental analysis.
        self.free_surface_critical_energy = 1
        self.close_surface_critical_energy = 1

    def predict_sde(self, verbose = False):
        '''predict based on the energy deposited over the foil surface'''
        '''returns the surface energy on the given foil surface'''
        '''assumes that all surface panels will recive the same energy'''

        self.surface_midpoints = jnp.array([(self.surface_panels[n+1] + self.surface_panels[n])/2 for n, _ in enumerate(self.surface_panels)])
        self.surface_vectors = np.diff(self.surface_panels,axis=0)[0]
        self.surface_areas = [np.hypot(x,y) for x,y in  self.surface_vectors]

        self.wire_midpoints = jnp.array([(self.wire_trajectory_panels[n+1] + self.wire_trajectory_panels[n])/2 for n, _ in enumerate(self.wire_trajectory_panels)])
        self.wire_vectors = np.diff(self.wire_trajectory_panels,axis=0)[0]
        self.wire_areas = [np.hypot(x,y) for x,y in  self.wire_vectors]


        self.surface_irradance = jnp.zeros_like(self.surface_panels) #Keeps track of how much energy a panel has been exposed to.
        '''
        to accelerate this calculation we can calculate what regions of the trajectory are visable from each each panel, this means we dont have to compute all point for every panel
        this step is hugely computationally expensive.
        '''
        #self.wire_midpoints = jnp.array([panel.mid_point for panel in self.wire_trajectory_panels])  # Shape: (num_wires, 2)
        #self.wire_vectors = jnp.array([panel.panel_vector for panel in self.wire_trajectory_panels])  # Shape: (num_wires, 2)
        #self.surface_midpoints = jnp.array([panel.mid_point for panel in self.surface_panels])  # Shape: (num_surfaces, 2)
        #self.surface_vectors = jnp.array([panel.panel_vector for panel in self.surface_panels])  # Shape: (num_surfaces, 2)
        #self.surface_areas = jnp.array([panel.area for panel in self.surface_panels])  # Shape: (num_surfaces,)
        #self.wire_areas = jnp.array([panel.area for panel in self.wire_trajectory_panels])
        # Compute differences and distances between all wire and surface panels
        self.diff = self.wire_midpoints[:, np.newaxis, :] - self.surface_midpoints[np.newaxis, :, :]  # Shape: (num_wires, num_surfaces, 2)
        self.distances = jnp.linalg.norm(self.diff, axis=2)  # Shape: (num_wires, num_surfaces)
        # Compute visibility matrix using broadcasting
        # Cross product in 2D: z-component of (a x b) = a[0]*b[1] - a[1]*b[0]
        self.cross_products = (
        self.surface_vectors[np.newaxis, :, 0] * self.diff[:, :, 1] -
        self.surface_vectors[np.newaxis, :, 1] * self.diff[:, :, 0]
        )  # Shape: (num_wires, num_surfaces)
        self.visibility_matrix = (0 >= self.cross_products.T)  # Transpose for desired shape
        # Compute perspective matrix using broadcasting
        self.dot_products = jnp.abs(
        self.surface_vectors[np.newaxis, :, 0] * self.wire_vectors[:, 0][:, np.newaxis] +
        self.surface_vectors[np.newaxis, :, 1] * self.wire_vectors[:, 1][:, np.newaxis]
        )  # Shape: (num_wires, num_surfaces)
        self.perspective_matrix = (self.dot_products.T * self.wire_areas * self.surface_areas * jnp.tri(*self.visibility_matrix.shape,k=1)) / (self.distances.T**2)  # Transpose for shape alignment
        # Calculate surface exposure
        self.surface_exposure = jnp.sum(self.visibility_matrix * self.perspective_matrix, axis=1)#
        if verbose:
            plt.imshow(self.visibility_matrix, interpolation='spline16', cmap = "binary")
            plt.colorbar()
            plt.grid()
            plt.ylabel("Surface panel index")
            plt.xlabel("Wire panel index")
            plt.title("Wire Surface Visibility")
            plt.gca().invert_yaxis()
            plt.show()
            #plt.imshow(self.visibility_matrix)
            #plt.imshow(self.distances)
            plt.imshow(self.perspective_matrix*self.visibility_matrix, interpolation='nearest')
            #plt.imshow(self.visibility_matrix*self.perspective_matrix*np.tri(*self.visibility_matrix.shape,k=1),interpolation='nearest', cmap='jet')
            plt.gca().invert_yaxis()
            plt.ylabel("Surface panel index")
            plt.xlabel("Wire panel index")
            plt.colorbar()
            plt.show()
            plt.plot(self.surface_exposure, c = 'black')
            plt.xlabel("Surface panel index")
            plt.ylabel("Dimensionless heating parameter")
            plt.grid()
            plt.show()
            self.fig,self.axs = plt.subplots()
            for n, panel in enumerate(self.surface_panels):
                self.axs.plot(panel.points[:,0],panel.points[:,1],c=cm.jet(self.surface_exposure[n]/np.max(self.surface_exposure)))
            plt.gca().set_aspect('equal')
            plt.xlabel("X position [mm]")
            plt.ylabel("Y position [mm]")
            plt.grid()
            plt.show()
        return self.surface_exposure

class Panel(object):
    def __init__(self, p1,p2):
        self.points = np.vstack((p1,p2))
        self.panel_vector = np.diff(self.points,axis=0)[0]
        self.area = np.hypot(self.panel_vector[0],self.panel_vector[1])
        self.mid_point = np.mean(self.points,axis=0)
        self.direction = np.sign(np.diff(self.points[:,0],axis = 0)[0]) #Is the direction of travel of curve

    def update(self):
        self.mid_point = np.mean(self.points,axis=0)
        self.direction = np.sign(np.diff(self.points[:,0],axis = 0)[0]) #Is the direction of travel of curve
        self.area = np.hypot(self.panel_vector[0],self.panel_vector[1])
        self.line()

class Shape(object):
    def __init__(self, root, tip, root_chord, tip_chord, root_alpha, tip_alpha, sweep_angle):
        self.root = root
        self.tip = tip
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.root_alpha = root_alpha
        self.tip_alpha = tip_alpha
        self.sweep = sweep_angle
        self.Compute_goemetry()
        self.Fit_bspline_surface()
    def Discretize(self, points):
        return np.array([Panel(points[0,n], points[0,n+1]) for n in range(0,points.shape[1]-1)]) #Turns the continuious surface into a set of discrete panels, points are distributed evenly along the parametrisation

    def Compute_goemetry(self):
        '''Generated a set of points that describes the 2 directries'''
        '''Working'''
        self.root_discrete_directrices = self.rotate_data(self.root*self.root_chord, self.root_alpha, np.array([self.root_chord*0.3,0]))#The 2D discrete directrie for the wing root
        self.tip_discrete_directrices = self.rotate_data(self.tip*self.tip_chord, self.tip_alpha, np.array([self.tip_chord*0.3,0])) #The 2D discrete directrie for the wing root

    def rotate_data(self,data,alpha,rot_axis):
        '''Wow this is so inefficient, have you heard of a matricie, no? fine!'''
        self.rotated_dat= []
        for self.point in data:
            self.rel_point = self.point-rot_axis
            self.p_point = [np.sqrt(self.rel_point[0]**2+self.rel_point[1]**2),np.arctan2(self.rel_point[1],self.rel_point[0])]
            self.p_point[1]-= np.radians([alpha])
            self.c_points = np.array([self.p_point[0]*np.cos(self.p_point[1]),self.p_point[0]*np.sin(self.p_point[1])]).flatten()
            self.rotated_dat.append(self.c_points+rot_axis)
        self.rotated_dat = np.array(self.rotated_dat)
        return self.rotated_dat

    def Fit_bspline_surface(self):
        '''Fits parametric b-splines to the discrete directries'''
        '''these b-splines are parametrised in x and y seperatly with paramter t'''
        '''A second function defines the rate at which the curve is traversed.'''
        self.r_tck, _ = interpolate.splprep([self.root_discrete_directrices[:,0],self.root_discrete_directrices[:,1]],k=5,s=0.1,per=True) #Continuious directrie for root rail
        self.t_tck, _ = interpolate.splprep([self.tip_discrete_directrices[:,0],self.tip_discrete_directrices[:,1]],k=5,s=0.1,per=True) #Continuious directrie for tip rail
        '''
        self.i_x, self.i_y = interpolate.splev(linspace(0,1,samples),self.tck)
        when ready we use this to sample the curve at desired points.
        this returns the spline parameteried in u.
        '''

    def Tip_knots(self):
        #print(self.t_tck[0])
        return interpolate.splev(self.t_tck[0],self.t_tck)

    def Tip_geometry(self):
        self.Samples = np.linspace(0,1,1000, endpoint = False)
        return interpolate.splev(self.Samples,self.t_tck)

    def Root_geometry(self):
        self.Samples = np.linspace(0,1,1000, endpoint = False)
        return interpolate.splev(self.Samples,self.r_tck)

    def constituants(self):
        '''breaks the foil surface in to discrete panel elements'''
        pass
class main(object):
    def __init__(self):
        '''Loading in foil data'''
        self.foil_addr = "Airfoils//s1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]
        self.Optimisation_Iteration_Limit = 100
        self.Particles = 20

        '''Optimisation Parameters'''
        self.C1 = self.C2  = 0.1
        self.w = 0.8


        self.mainloop()

    def Discretize(self, points):
        '''to correctly discretise we have to close the path'''
        points = np.concatenate([points[0],[points[0,0]]],axis=0)#this closes the loop
        #print(points.shape)
        return np.array([[points[n], points[n+1]] for n in range(0,points.shape[0]-1)]) #Turns the continuious surface into a set of discrete panels, points are distributed evenly along the parametrisation

    def MSE_Loss(self, array, target):
        '''Define loss function'''
        '''Distance between desired shape and predicted shape'''
        return (1/len(array))*np.sum((array-target)**2)
        pass

    def format_dat(self,data):
        # TODO: Needs Improvment for greater compatibility
        #Grandfarthered from old version
        self.data = data.split("\n")
        self.formatted = [self.el.split(" ") for self.el in self.data]
        self.formatted  = [[float(self.num) for self.num in list(filter(lambda x:x!='',self.coord))]for self.coord in self.formatted]#list(map(float,self.formatted))
        self.formatted = list(filter(lambda x:x!=[],self.formatted))
        return self.formatted

    def panel_intersections(self, panels):
        self.points = []
        for n in range(0, panels.shape[0]-1):
            self.x_intersection = (panels[n+1].c-panels[n].c)/(panels[n].m-panels[n+1].m)
            self.y_intersection = panels[n].m*self.x_intersection+panels[n].c
            panels[n+1].points[0] = np.r_[*[self.x_intersection,self.y_intersection]]
            panels[n].points[1] = np.r_[*[self.x_intersection,self.y_intersection]]
            self.points.append([self.x_intersection,self.y_intersection])
        #We now deal with the edge case of first and last elements
        self.x_intersection = (panels[0].c-panels[-1].c)/(panels[-1].m-panels[0].m)
        self.y_intersection = panels[-1].m*self.x_intersection+panels[-1].c
        panels[0].points[0] = np.r_[*[self.x_intersection,self.y_intersection]]
        panels[-1].points[1] = np.r_[*[self.x_intersection,self.y_intersection]]
        self.points.append([self.x_intersection,self.y_intersection])
        return panels

    def Offset(self, nodes, offsets):
        for n, node in enumerate(nodes):
            

    def Objective_function(self, offsets):
        '''Exists just to simplify and discretize away this step in a nice way.'''
        self.Local_Panels = self.Panels.copy()
        '''Extract nodes from'''
        self.offset_panels = np.array([panel.offset(offsets[n]) for n, panel in enumerate(self.Local_Panels)])#offsets the panel
        '''We now calculate the new intersections between the panels'''
        self.trajectory_Panels = self.panel_intersections(self.offset_panels) #calculates the new meeting points between panels and redefines the panels, this is the trajectory path
        '''^^^^^^^^^^^''''
        '''Need to change this to be panel node offsets'''
        self.trajectory = Trajectory(self.Panels, self.trajectory_Panels)
        self.surface_exposure = self.trajectory.predict_sde()
        self.Loss = self.MSE_Loss(self.surface_exposure, self.trajectory.critical_surface_energy)
        return self.Loss

    def mainloop(self):
        '''Load in Foil Data'''
        self.shape = Shape(self.foil_dat, self.foil_dat, 300,200,2,0,5) #Instanciated the foil
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.foil_dat[:,0],self.foil_dat[:,1])
        plt.title("S1223 Foil")
        plt.gca().set_aspect('equal')
        plt.show()
        '''Tip Geometry is extracted from the composite B-Spline'''
        self.x, self.y = self.shape.Tip_geometry()
        self.Tip_points = np.dstack((self.x,self.y))

        '''The cutting path now needs to be initialised, this is done by offsetting the the surface points and then defining a new B-Spline'''
        self.offset_guess = 3
        self.offset_points = self.offset(self.Tip_points, np.full_like(self.Tip_points,self.offset_guess))
        '''We now calculate the new intersections between the panels'''
        self.trajectory = Trajectory(self.Tip_points, self.offset_points)

        self.fig,self.axs = plt.subplots()
        self.curve = []
        for panel in self.Panels:
            self.axs.plot(panel.points[:,0],panel.points[:,1],c="black",linestyle = (0, (3, 1, 1, 1)))
        for panel in self.trajectory_Panels:
            self.curve.append([panel.points[1,0],panel.points[1,1]])
        self.curve = np.array(self.curve)
        self.axs.plot(self.curve[:,0],self.curve[:,1],c="black",linestyle = (0, (1, 2)))
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()

        '''OPTIMIZATION STEP'''
        '''Doing Autodiff is going to be basicly impossible here'''
        '''so we are going to go gradientles'''
        '''Look whos that, is that Nelder-mead, is that Baysian Optimisation! NO, its Particle Swarm !!!'''
        '''Draw N Samples'''
        self.LHS = qmc.LatinHercube(d=100) #define latinhypercube sampler for an arrray of 100D design varaible, LHS is used becuase it will give a better understanding of the design space
        self.Particle_Positions = self.LHS.random(n = self.Particles)*10  #draw N samples of the design space, the LHS samples are scaled to in the range +5mm, the design variable defined the off set of each node in the wire trajectory directorie
        self.Particle_Velocity = (self.LHS.random(n = self.Particles)-0.5) #draw N samples of the design space for the velocity this is scaled and given a negative component to promote search in both directions
        self.Particle_best_performance = [self.Objective_function(particle) for particle in self.Particle_Positions]
        '''We can now start the Optimisation'''
        for _ in range(self.Optimisation_Iteration_Limit):
            '''We evalute the objective function at each particles position'''
            self.Particle_performance = [self.Objective_function(particle) for particle in self.Particle_Positions]#Calculate the performance of each particle, this can be done as a parellized process, Much speed, go fast.
            self.improved = self.Particle_performance < self.Particle_best_performance
            self.Particle_best_positions[self.improved] = self.Particle_Positions[self.improved]
            self.Particle_best_performance[self.improved] = self.Particle_performance[self.improved]
            self.best_index = np.arg_min(self.Particle_performance)
            if self.Particle_performance[self.best_index] < self.Global_best:
                self.Global_best_performance = self.Particle_performance[self.best_index]#Global Best Solution
                self.Global_best_position = self.Particle_Positions[self.best_index]
            '''We now update the particle velocity'''
            self.Particle_Velocity = self.W*self.Particle_Velocity +
                                     self.C1*np.random.rand()*(self.Particle_best_performance - self.Particle_Positions) +
                                     self.C2*np.random.rand()*(self.Global_best_performance - self.Particle_Positions)
            self.Particle_Positions +=  self.Particle_Velocity

        '''TAAAAA DAAAA'''
        '''Like magic'''



if __name__ == "__main__":
    main()
