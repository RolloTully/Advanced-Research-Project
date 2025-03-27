import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
from tqdm import tqdm
from matplotlib import cm
from shapely.geometry import LineString
plt.rcParams["font.family"] = "Times New Roman"

'''
Questions to be answered.

What is the surface energy of the foil.

What is the program, How does it work.
'''
class Trajectory(object):
    def __init__(self, surface, wire_trajectory, wire_velocity = []):
        self.surface_nodes = surface
        self.wire_trajectory_nodes = wire_trajectory
        self.wire_velocity = wire_velocity
        self.critical_surface_energy = 22 #<<<<<<-----------This needs to be calibrated from experiemental analysis.

    def predict_sde(self, verbose = False):
        '''predict based on the energy deposited over the foil surface'''
        '''returns the surface energy on the given foil surface'''
        '''assumes that all surface panels will recive the same energy'''
        self.surface_midpoints = np.array([(self.surface_nodes[n+1] + self.surface_nodes[n])/2 for n in range(self.surface_nodes.shape[0]-1)])
        self.surface_vectors = np.diff(self.surface_nodes,axis=0)
        self.surface_areas = [np.hypot(x,y) for x,y in  self.surface_vectors]
        self.wire_midpoints = np.array([(self.wire_trajectory_nodes[n+1] + self.wire_trajectory_nodes[n])/2 for n in range(self.wire_trajectory_nodes.shape[0]-1)])
        self.wire_vectors = np.diff(self.wire_trajectory_nodes,axis=0)
        self.wire_areas = [np.hypot(x,y) for x,y in  self.wire_vectors]
        self.surface_irradance = np.zeros_like(self.surface_nodes) #Keeps track of how much energy a panel has been exposed to.
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
        self.distances = np.linalg.norm(self.diff, axis=2)  # Shape: (num_wires, num_surfaces)
        # Compute visibility matrix using broadcasting
        # Cross product in 2D: z-component of (a x b) = a[0]*b[1] - a[1]*b[0]
        self.cross_products = (
        self.surface_vectors[np.newaxis, :, 0] * self.diff[:, :, 1] -
        self.surface_vectors[np.newaxis, :, 1] * self.diff[:, :, 0]
        )  # Shape: (num_wires, num_surfaces)
        self.visibility_matrix = (0 >= self.cross_products.T)  # Transpose for desired shape
        # Compute perspective matrix using broadcasting
        self.dot_products = np.abs(
        self.surface_vectors[np.newaxis, :, 0] * self.wire_vectors[:, 0][:, np.newaxis] +
        self.surface_vectors[np.newaxis, :, 1] * self.wire_vectors[:, 1][:, np.newaxis]
        )  # Shape: (num_wires, num_surfaces)
        self.perspective_matrix = (self.dot_products.T * self.wire_areas * self.surface_areas * np.tri(*self.visibility_matrix.shape,k=1)) / (self.distances.T**2)  # Transpose for shape alignment
        # Calculate surface exposure
        self.surface_exposure = np.sum(self.visibility_matrix * self.perspective_matrix, axis=1)#
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
            for n in range(0,len(self.surface_nodes)-1):
                print(self.surface_nodes[n:n+2,0],self.surface_nodes[n:n+2,1])
                self.axs.plot(self.surface_nodes[n:n+2,0],self.surface_nodes[n:n+2,1],c=cm.jet(self.surface_exposure[n]/np.max(self.surface_exposure)))
            plt.gca().set_aspect('equal')
            plt.title("Surface heating")
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
    def Tip_geometry(self):
        self.Samples = np.linspace(0,1,200, endpoint = False)
        return interpolate.splev(self.Samples,self.t_tck)

    def Root_geometry(self):
        self.Samples = np.linspace(0,1,200, endpoint = False)
        return interpolate.splev(self.Samples,self.r_tck)


class main(object):
    def __init__(self):
        '''Loading in foil data'''
        self.foil_addr = "Airfoils//s1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]
        self.k = 3
        self.n_samples  = 400
        self.Optimisation_Iteration_Limit = 200


        self.mainloop()

    def Discretize(self, points):
        '''to correctly discretise we have to close the path'''
        points = np.concatenate([points[0],[points[0,0]]],axis=0)#this closes the loop
        return np.array([[points[n], points[n+1]] for n in range(0,points.shape[0]-1)]) #Turns the continuious surface into a set of discrete panels, points are distributed evenly along the parametrisation

    def MSE_Loss(self, array, target):
        '''Define loss function'''
        '''Distance between desired shape and predicted shape'''
        return np.mean((array-target)**2)

    def format_dat(self,data):
        # TODO: Needs Improvment for greater compatibility
        #Grandfarthered from old version
        self.data = data.split("\n")
        self.formatted = [self.el.split(" ") for self.el in self.data]
        self.formatted = [[float(self.num) for self.num in list(filter(lambda x:x!='',self.coord))]for self.coord in self.formatted]
        self.formatted = list(filter(lambda x:x!=[],self.formatted))
        return self.formatted

    def Offset(self, nodes, offsets):
        '''Fast as fuck bois, like 10 microseconds for 100 points '''
        self.nodes = np.concatenate([nodes[-1:],nodes,nodes[:1]],axis=0)#Formats array, copies for and last elements to the last and first positions
        self.derivatives = self.nodes[1:]-self.nodes[:-1]# The difference between neighbouring nodes
        self.Unit_derivaitves = self.derivatives/np.sqrt(np.sum(self.derivatives**2,axis=1))[:,None]#calculates the plane unit vector
        self.Offset_unit_vectors = self.Unit_derivaitves[:,::-1]#flips cooordinate
        self.Offset_unit_vectors[:,0] *= -1#multiplies 1st element by minus 1
        self.Offset_unit_vectors = (self.Offset_unit_vectors[1:]+self.Offset_unit_vectors[:-1])/2 #calculated the average of neighbouring unit vectors
        self.Offset_vectors = self.Offset_unit_vectors*offsets[:,None]*-1#scales offset vector
        return nodes+self.Offset_vectors#adds offset vector to nodes and returns

    def Objective_function(self, BSpline):
        '''Exists just to simplify and discretize away this step in a nice way.'''
        self.trajectory_nodes = interpolate.splev(np.linspace(0,1,200, endpoint = False),BSpline)
        self.Trajectory_line = LineString([(self.trajectory_nodes[0][i],self.trajectory_nodes[1][i]) for i,_ in enumerate(self.trajectory_nodes[0])])
        '''Violation loss'''
        '''great care must be taken with fixed losses, they add uncertainty and can ruin the optimisation'''
        self.Violation_loss = 0
        if self.Tip_line.intersection(self.Trajectory_line).is_empty != True:#if the 2 paths intersect there is a huge penalty
            print("Invalid trajectory")
            self.Exposure_loss = 400#*len(self.Tip_line.intersection(self.Trajectory_line).geoms)

        '''Exposure loss'''
        '''is the surface getting correctly irradiated'''
        self.trajectory_nodes = np.dstack((self.trajectory_nodes[0],self.trajectory_nodes[1]))[0]
        self.trajectory = Trajectory(self.Tip_points, self.trajectory_nodes)
        self.surface_exposure = self.trajectory.predict_sde()
        self.Exposure_loss = self.MSE_Loss(self.surface_exposure, self.trajectory.critical_surface_energy)
        '''offset losses'''
        '''the further the wire gets from the surface the larger this los gets'''
        self.offset_loss = np.sqrt(np.sum((self.Tip_points-self.trajectory_nodes)**2))
        return self.Exposure_loss+self.offset_loss+self.Violation_loss

    def mainloop(self):
        '''Load in Foil Data'''
        self.shape = Shape(self.foil_dat, self.foil_dat, 300,200,2,0,5) #Instanciated the foil
        '''Tip Geometry is extracted from the composite B-Spline'''
        self.x, self.y = self.shape.Tip_geometry()#Extract the b-spline geometry as a set of points
        self.Tip_points = np.dstack((self.x,self.y))[0]#Stack the 2 arrays [N,] into an [N,2] array
        self.Tip_line = LineString([(self.Tip_points[i,0],self.Tip_points[i,1]) for i,_ in enumerate(self.Tip_points)])
        '''The cutting path now needs to be initialised, this is done by offsetting the the surface points and then defining a new B-Spline'''
        '''we want to minimise the work done by the optimiser, so we find the best starting path'''
        self.offset_to_beat = 0
        self.minimum_loss = 1e9
        for offset_guess in np.linspace(0.01,10,100):
            self.offset_points = self.Offset(self.Tip_points, np.full_like(self.Tip_points[:,0],offset_guess)) #Initialising trajectory for the inital simulation
            self.offset_points[0] +=[offset_guess,0]
            self.tck, _ = interpolate.splprep([self.offset_points[:,0],self.offset_points[:,1]],k=self.k,s=0.1,per=False)
            self.t, self.c, self.k  = self.tck
            self.loss = self.Objective_function([self.t,self.c,self.k])
            if self.loss < self.minimum_loss:
                self.offset_to_beat = offset_guess
                self.minimum_loss = self.loss
        '''we have found a good starting point'''
        self.offset_points = self.Offset(self.Tip_points, np.full_like(self.Tip_points[:,0],self.offset_to_beat)) #Initialising trajectory for the inital simulation
        self.offset_points[0] +=[self.offset_to_beat,0]
        self.tck, _ = interpolate.splprep([self.offset_points[:,0],self.offset_points[:,1]],k=self.k,s=0.1,per=False)
        self.t, self.c, self.k  = self.tck
        self.n_points = interpolate.splev(np.linspace(0,1,200, endpoint = True),self.tck)
        self.fig=plt.figure()
        plt.axes().set_aspect('equal')
        plt.title("Tip foil profile and inital wire trajectory")
        plt.plot(self.Tip_points[:,0],self.Tip_points[:,1])
        plt.plot(self.n_points[0],self.n_points[1])
        plt.plot(self.offset_points[:,0],self.offset_points[:,1])
        plt.show()

        self.c = torch.as_tensor(np.array(self.c).flatten())#np.dstack((self.c[0],self.c[1]))[0]
        #print(self.c)
        #input()
        '''OPTIMIZATION STEP'''
        '''Doing Autodiff is going to be basicly impossible here'''
        '''so we are going to go gradientles'''
        '''Look whos that, is that Nelder-mead, is that Baysian Optimisation! NO, its Particle Swarm !!!'''
        '''Draw N Samples'''
        '''Drawing samples'''


        print(self.c.shape)
        self.train_x = self.c + (torch.rand(self.n_samples,self.c.shape[0],dtype = torch.float64)-1)*2
        self.train_y = torch.as_tensor(np.array([self.Objective_function([self.t,sample.reshape((2,24)),self.k]) for sample in self.train_x])).reshape((400,1))
        print(self.train_x)
        print(self.train_y)

        self.fig, self.ax = plt.subplots(1,2)
        #self.ax.set_aspect('equal')
        self.ax[0].plot(self.Tip_points[:,0],self.Tip_points[:,1])
        self.ax[0].plot(self.offset_points[:,0],self.offset_points[:,1])
        self.ax[0].set_aspect('equal')
        self.ax[0].set_xlim([-15,215])
        self.ax[1].set_ylim([0,100])

        self.bounds = torch.stack([torch.zeros(self.c.shape), torch.ones(self.c.shape)])

        self.bounds = torch.stack([
                        self.c-5,  # Lower bound
                        self.c+5    # Upper bound
                        ])
        for _ in tqdm(range(self.Optimisation_Iteration_Limit)):
            '''Define surrogate model'''
            self.model = SingleTaskGP(self.train_x, self.train_y)
            '''define model mean and std'''
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            '''fit model'''
            fit_gpytorch_mll(self.mll)
            '''define aquisition function'''
            self.acq_func = qLogExpectedImprovement(model=self.model, best_f=self.train_y.max()) #we want to maximise improvment
            self.candidate, self.acq_value = optimize_acqf(
                acq_function=self.acq_func,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )

            self.train_x = torch.vstack((self.train_x,self.candidate))
            self.new_y = self.Objective_function([self.t,self.candidate.reshape((2,24)),self.k])
            self.train_y = torch.vstack((self.train_y, torch.tensor(self.new_y)))

            self.ax[1].plot(self.surface_exposure)
            self.min_y_value, self.min_index = torch.min(self.train_y, dim=0)  # Get min value & index
            print("Best performance:", self.min_y_value)
            self.min_position = self.train_x[self.min_index]
            self.trajectory_nodes = interpolate.splev(np.linspace(0,1,100, endpoint = False),[self.t,self.min_position.reshape((2,24)),self.k])
            #print(self.trajectory_nodes)
            self.ax[0].plot(self.trajectory_nodes[0],self.trajectory_nodes[1])
            plt.pause(0.001)

            print(self.candidate)
            print(self.new_y)


        '''TAAAAA DAAAA'''
        '''Like magic'''
        '''I swear this should have been harder'''





if __name__ == "__main__":
    main()
