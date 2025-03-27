import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
from tqdm import tqdm
from matplotlib import cm
<<<<<<< Updated upstream
=======
from shapely.geometry import LineString
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import neuralfoil as nf
>>>>>>> Stashed changes
plt.rcParams["font.family"] = "Times New Roman"

'''
Questions to be answered.

What is the surface energy of the foil.

What is the program, How does it work...




'''
class Trajectory(object):
    def __init__(self, surface, wire_trajectory, wire_velocity = []):
        self.surface_panels = surface
        self.wire_trajectory_panels = wire_trajectory
        self.wire_velocity = wire_velocity
<<<<<<< Updated upstream
        self.critical_surface_energy = 1 #<<<<<<-----------This needs to be calibrated from experiemental analysis.
        self.free_surface_critical_energy = 1
        self.close_surface_critical_energy = 1
=======
        self.critical_surface_energy = 22 #<<<<<<-----------This needs to be calibrated from experiemental analysis.
>>>>>>> Stashed changes

    def dkr(self,sfm):
        pass
    def predict_dkr(self):
        '''Predicts based on a direct relation between cutter speed and kerf width'''
        '''returns the predicted surface shape from the given trajectory'''
        pass

    def predict_sde(self, verbose = False):
        '''predict based on the energy deposited over the foil surface'''
        '''returns the surface energy on the given foil surface'''
        '''assumes that all surface panels will recive the same energy'''
<<<<<<< Updated upstream
        [panel.update() for panel in self.surface_panels]
        [panel.update() for panel in self.wire_trajectory_panels]
        self.mid_points = np.array([panel.mid_point for panel in self.surface_panels])
        self.surface_irradance = np.zeros_like(self.surface_panels) #Keeps track of how much energy a panel has been exposed to.
=======
        self.surface_midpoints = np.array([(self.surface_nodes[n+1] + self.surface_nodes[n])/2 for n in range(self.surface_nodes.shape[0]-1)])
        self.surface_vectors = np.diff(self.surface_nodes,axis=0)
        self.surface_areas = [np.hypot(x,y) for x,y in  self.surface_vectors]
        self.wire_midpoints = np.array([(self.wire_trajectory_nodes[n+1] + self.wire_trajectory_nodes[n])/2 for n in range(self.wire_trajectory_nodes.shape[0]-1)])
        self.wire_vectors = np.diff(self.wire_trajectory_nodes,axis=0)
        self.wire_areas = [np.hypot(x,y) for x,y in  self.wire_vectors]
        self.surface_irradance = np.zeros_like(self.surface_nodes) #Keeps track of how much energy a panel has been exposed to.
>>>>>>> Stashed changes
        '''
        to accelerate this calculation we can calculate what regions of the trajectory are visable from each each panel, this means we dont have to compute all point for every panel
        this step is hugely computationally expensive.
        '''
<<<<<<< Updated upstream
        self.wire_midpoints = np.array([panel.mid_point for panel in self.wire_trajectory_panels])  # Shape: (num_wires, 2)
        self.wire_vectors = np.array([panel.panel_vector for panel in self.wire_trajectory_panels])  # Shape: (num_wires, 2)
        self.surface_midpoints = np.array([panel.mid_point for panel in self.surface_panels])  # Shape: (num_surfaces, 2)
        self.surface_vectors = np.array([panel.panel_vector for panel in self.surface_panels])  # Shape: (num_surfaces, 2)
        self.surface_areas = np.array([panel.area for panel in self.surface_panels])  # Shape: (num_surfaces,)
        self.wire_areas = np.array([panel.area for panel in self.wire_trajectory_panels])
        # Compute differences and distances between all wire and surface panels
=======

>>>>>>> Stashed changes
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
            self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='3d'))
            print(self.surface_exposure)
            self.ax.stem(self.surface_midpoints[:,0],self.surface_midpoints[:,1], self.surface_exposure)
            self.ax.set_proj_type('persp')
            plt.show()
        return self.surface_exposure
    def Predict_geom(self):
        '''Predicts the final geometry allowing for performance aware optimisation'''




class Panel(object):
    def __init__(self, p1,p2):
        self.points = np.vstack((p1,p2))
        self.panel_vector = np.diff(self.points,axis=0)[0]
        self.panel_normal_vector = self.perpendicular(self.panel_vector).astype(np.float32)
        self.area = np.hypot(self.panel_vector[0],self.panel_vector[1])
        self.mid_point = np.mean(self.points,axis=0)
        self.direction = np.sign(np.diff(self.points[:,0],axis = 0)[0]) #Is the direction of travel of curve
        self.line()

    def perpendicular(self, a) :
        self.b = np.empty_like(a)
        self.b[0] = -a[1]
        self.b[1] = a[0]
        return self.b
    def line(self):
        self.m = np.diff(self.points[:,1])/np.diff(self.points[:,0])
        self.c =  self.points[0,1]-self.points[0,0]*self.m

    def offset(self,d):
        '''returns a copy that has been offset by d mm'''
        '''
        This has some problems with certain foil geometries, needs to be refactored.
        '''
        self.offset_self = copy.deepcopy(self)
        self.offset_self.c = self.c-self.direction*d*np.sqrt(1+self.m**2)
        return self.offset_self

    def does_intersect(self ,m ,c):
        '''Does a line intersect with the panel and how far from the source to the panel and the angle of intersection'''
        '''returns boolean intersection check'''
        #finds the point of intersection
        self.x_intersection = (c-self.c)/(self.m-m)
        self.y_intersection = self.x_intersection*self.m+self.c
        #parameterises the intersection point
        #print(np.array([self.x_intersection, self.y_intersection]).T,self.points[1],self.panel_vector)
        self.parameteriation = (np.array([self.x_intersection, self.y_intersection]).T-self.points[1])/self.panel_vector
        #print(self.parameteriation)
        if np.all(0 <= self.parameteriation) and np.all(self.parameteriation <=1):
            #the intersection lies within the bounds of panel
            return True
        else:
            return False
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
        #print(points.shape)
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
        self.foil_addr = "Airfoils//S1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]
<<<<<<< Updated upstream
=======
        self.Particles = 10
        self.k = 5
        self.Smoothing = 0.01

        self.Max_iterations = 800

        '''Global Optimisation Parameters'''
        self.G_C1 = 0.2 #Local
        self.G_C2  = 0.8 #Global
        self.G_w = 0.8#Momentum
        '''Local Optimisation Parameters'''
        self.L_C1 = 0.8
        self.L_C2 = 0.2
        self.L_w = 0.8
>>>>>>> Stashed changes
        self.mainloop()

    def add_arrow_to_line2D(self, axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],arrowstyle='-|>', arrowsize=1, transform=None):
        """
        Add arrows to a matplotlib.lines.Line2D at selected locations.

        Parameters:
        -----------
        axes:
        line: Line2D object as returned by plot command
        arrow_locs: list of locations where to insert arrows, % of total length
        arrowstyle: style of the arrow
        arrowsize: size of the arrow
        transform: a matplotlib transform instance, default to data coordinates

        Returns:
        --------
        arrows: list of arrows
        """
        if not isinstance(line, mlines.Line2D):
            raise ValueError("expected a matplotlib.lines.Line2D object")
        x, y = line.get_xdata(), line.get_ydata()

        arrow_kw = {
            "arrowstyle": arrowstyle,
            "mutation_scale": 10 * arrowsize,
        }

        color = line.get_color()
        use_multicolor_lines = isinstance(color, np.ndarray)
        if use_multicolor_lines:
            raise NotImplementedError("multicolor lines not supported")
        else:
            arrow_kw['color'] = color

        linewidth = line.get_linewidth()
        if isinstance(linewidth, np.ndarray):
            raise NotImplementedError("multiwidth lines not supported")
        else:
            arrow_kw['linewidth'] = linewidth

        if transform is None:
            transform = axes.transData

        arrows = []
        for loc in arrow_locs:
            s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
            n = np.searchsorted(s, s[-1] * loc)
            arrow_tail = (x[n], y[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
            p = mpatches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform,
                **arrow_kw)
            axes.add_patch(p)
            arrows.append(p)
        return arrows


    def Discretize(self, points):
        '''to correctly discretise we have to close the path'''
        points = np.concatenate([points[0],[points[0,0]]],axis=0)#this closes the loop
<<<<<<< Updated upstream
        #print(points.shape)
        return np.array([Panel(points[n], points[n+1]) for n in range(0,points.shape[0]-1)]) #Turns the continuious surface into a set of discrete panels, points are distributed evenly along the parametrisation

    def _curve(self, panels, offset_vector):
        self.offset_points = []
        for  n in range(0, points.shape[0]-1):
            '''We need to calculate the surface vector at the point, this is the rotational average of the 2 neighboring panels'''
=======
        return np.array([[points[n], points[n+1]] for n in range(0,points.shape[0]-1)]) #Turns the continuious surface into a set of discrete panels, points are distributed evenly along the parametrisation
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream

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

    def Auto_Differentiation(self):
        pass
=======
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

    def Objective_function(self, BSpline, verbose = False):
        '''Exists just to simplify and discretize away this step in a nice way.'''
        self.trajectory_nodes = interpolate.splev(np.linspace(0,1,200, endpoint = False),BSpline)
        self.Trajectory_line = LineString([(self.trajectory_nodes[0][i],self.trajectory_nodes[1][i]) for i,_ in enumerate(self.trajectory_nodes[0])])
        '''Violation loss'''
        '''great care must be taken with fixed losses, they add uncertainty and can ruin the optimisation'''
        self.Violation_loss = 0
        if self.Tip_line.intersection(self.Trajectory_line).is_empty != True:#if the 2 paths intersect there is a huge penalty
            #print("Violation")
            self.Violation_loss = 1e15#*len(self.Tip_line.intersection(self.Trajectory_line).geoms)

        '''Exposure loss'''
        '''is the surface getting correctly irradiated'''
        self.trajectory_nodes = np.dstack((self.trajectory_nodes[0],self.trajectory_nodes[1]))[0]
        self.trajectory = Trajectory(self.Tip_points, self.trajectory_nodes)
        self.surface_exposure = self.trajectory.predict_sde(verbose)
        self.Exposure_loss = self.MSE_Loss(self.surface_exposure, self.trajectory.critical_surface_energy)
        '''offset losses'''
        '''the further the wire gets from the surface the larger this los gets'''
        self.offset_loss = 0#np.sqrt(np.sum((self.Tip_points-self.trajectory_nodes)**2))
        return abs(self.Exposure_loss)+abs(self.offset_loss)+abs(self.Violation_loss)
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        self.x, self.y = self.shape.Tip_geometry()
        self.Tip_points = np.dstack((self.x,self.y))

        '''These points are turned in to panels'''
        self.Panels = self.Discretize(self.Tip_points)

        '''The cutting path now needs to be initialised, this is done by offsetting the the surface points and then defining a new B-Spline'''
        self.x = np.linspace(0,np.pi,len(self.Panels))
        self.offset_guess = 3
        plt.plot(self.offset_guess)
        plt.show()
        self.offset_panels = np.array([panel.offset(self.offset_guess) for n, panel in enumerate(self.Panels)])#offsets the panel
        '''We now calculate the new intersections between the panels'''
        self.trajectory_Panels = self.panel_intersections(self.offset_panels) #calculates the new meeting points between panels and redefines the panels, this is the trajectory path
        self.trajectory = Trajectory(self.Panels, self.trajectory_Panels)

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

        '''We now have the hotwires inital trajectory, we now need to calculate how well this trajectory performes'''
        self.surface_exposure = self.trajectory.predict_sde()
        self.Loss = self.MSE_Loss(self.surface_exposure, self.trajectory.critical_surface_energy)

        print("MSE Loss", self.Loss)
        '''We now need to do auto differentiation, fml'''

        '''This is suprisingly important, dont get rid of just yet'''
        self.fig,self.axs = plt.subplots()
        for panel in self.offset_panels:
            self.x = np.linspace(panel.points[0,0],panel.points[1,0],5)
            self.y = panel.m*self.x+panel.c
            self.axs.plot(self.x,self.y, c= "Green")
        plt.gca().set_aspect('equal')
        plt.show()

=======
        self.x, self.y = self.shape.Tip_geometry()#Extract the b-spline geometry as a set of points
        self.Tip_points = np.dstack((self.x,self.y))[0]#Stack the 2 arrays [N,] into an [N,2] array
        self.Tip_line = LineString([(self.Tip_points[i,0],self.Tip_points[i,1]) for i,_ in enumerate(self.Tip_points)])
        '''The cutting path now needs to be initialised, this is done by offsetting the the surface points and then defining a new B-Spline'''
        '''we want to minimise the work done by the optimiser, so we find the best starting path'''
        self.offset_to_beat = 0
        self.minimum_loss = 1e9
        for offset_guess in np.linspace(0.001,10,1000):
            self.offset_points = self.Offset(self.Tip_points, np.full_like(self.Tip_points[:,0],offset_guess)) #Initialising trajectory for the inital simulation
            self.offset_points[0] +=[offset_guess,0]
            self.tck, _ = interpolate.splprep([self.offset_points[:,0],self.offset_points[:,1]],k=self.k,s=0.02,per=False)
            self.t, self.c, self.k  = self.tck
            self.loss = self.Objective_function([self.t,self.c,self.k])
            if self.loss < self.minimum_loss:
                self.offset_to_beat = offset_guess
                self.minimum_loss = self.loss
        '''we have found a good starting point'''
        self.offset_points = self.Offset(self.Tip_points, np.full_like(self.Tip_points[:,0],self.offset_to_beat)) #Initialising trajectory for the inital simulation
        self.offset_points[0] +=[self.offset_to_beat,0]

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_title("Tip foil profile and inital wire trajectory")
        self.ax.set_ylabel("Y position [mm]")
        self.ax.set_xlabel("X position [mm]")
        self.ax.plot(self.Tip_points[:,0],self.Tip_points[:,1], color = "black")
        self.line, = self.ax.plot(self.offset_points[:,0],self.offset_points[:,1],color = "red")
        self.add_arrow_to_line2D(self.ax, self.line , arrow_locs=np.linspace(0., 1., 200),arrowstyle='->')
        #plt.plot(self.offset_points[:,0],self.offset_points[:,1], color = "red")
        plt.show()
        self.tck = interpolate.splprep([self.offset_points[:,0],self.offset_points[:,1]],k=self.k,s=self.Smoothing,per=False)[0]
        self.t, self.c, self.k  = self.tck
        self.v = np.dstack((self.c[0],self.c[1]))[0]
        #self.trajectory = Trajectory(self.Tip_points, self.t, self.c, self.k) #Instanciate this Trajectory calls

        '''OPTIMIZATION STEP'''
        '''Doing Autodiff is going to be basicly impossible here'''
        '''so we are going to go gradientles'''
        '''Look whos that, is that Nelder-mead, is that Baysian Optimisation! NO, its Particle Swarm !!!'''
        '''Draw N Samples'''
        self.Particle_Positions = np.array([self.v for _ in range(self.Particles)])#draw N samples of the design space, the LHS samples are scaled to in the range +5mm, the design variable defined the off set of each node in the wire trajectory directorie
        '''
        for pos in self.Particle_Positions[0:4]:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.ax.set_title("Potential new trajectories")
            self.ax.set_ylabel("Y position [mm]")
            self.ax.set_xlabel("X position [mm]")
            self.ax.plot(self.Tip_points[:,0],self.Tip_points[:,1], color = "black")

            self.nodes = interpolate.splev(np.linspace(0,1,1000, endpoint = False),[self.t,[pos[:,0], pos[:,1]],self.k])
            print(self.nodes)

            self.line, = self.ax.plot(self.nodes[0],self.nodes[1],color = "red")
            self.add_arrow_to_line2D(self.ax, self.line , arrow_locs=np.linspace(0., 1., 200),arrowstyle='->')
            #plt.plot(self.offset_points[:,0],self.offset_points[:,1], color = "red")
            plt.show()
        '''
        self.Particle_Velocity =  np.array([(np.random.randn(*self.v.shape)-0.5)*4 for _ in range(self.Particles)])#draw N samples of the design space for the velocity this is scaled and given a negative component to promote search in both directions
        self.Particle_best_positions = np.copy(self.Particle_Positions)
        self.Particle_performance = np.array([self.Objective_function([self.t,[particle[:,0],particle[:,1]],self.k]) for particle in self.Particle_Positions])
        self.Particle_best_performance = np.copy(self.Particle_performance)
        self.Global_best = np.argmin(self.Particle_performance)
        self.Global_best_position = np.copy(self.Particle_Positions[self.Global_best])
        self.Global_best_performance = self.Particle_performance[self.Global_best]

        '''We can now start the Optimisation'''
        self.fig, self.ax = plt.subplots()
        self.loss_array = []
        self.Last_global_best = 0
        self.no_change_count = 0
        self.best_index = np.argmin(self.Particle_performance)
        for _ in range(self.Max_iterations):
            '''This allows the optimisation to go between exploration at the start finding local optimums and then switching to global solutions'''
            print("Iteration: ", _, "Global best performance: ",self.Global_best_performance, " Iteration best: ",self.Particle_performance[self.best_index] )

            self.Particle_Positions +=  self.Particle_Velocity
            self.loss_array.append(self.Global_best_performance)

            '''We evalute the objective function at each particles position'''
            self.Particle_performance = np.array([self.Objective_function([self.t,[particle[:,0],particle[:,1]],self.k]) for particle in self.Particle_Positions])#Calculate the performance of each particle, this can be done as a parellized process, Much speed, go fast.
            self.improved = self.Particle_performance < self.Particle_best_performance
            self.Particle_best_positions[self.improved] = self.Particle_Positions[self.improved]
            self.Particle_best_performance[self.improved] = self.Particle_performance[self.improved]
            self.best_index = np.argmin(self.Particle_performance)
            if self.Particle_performance[self.best_index] < self.Global_best_performance:
                self.Global_best_performance = self.Particle_performance[self.best_index]#Global Best Solution
                self.Global_best_position = self.Particle_Positions[self.best_index]


            '''Depending on the optimiations behaviour we want to alternate between a global search and a local search'''
            '''After 10 steps of no improvment the PS switches to a local search mode and the particle positions are reinitialised'''
            self.C1_r = (1-np.heaviside(self.no_change_count-50,1))*self.L_C1 + np.heaviside(self.no_change_count-50,1)*self.G_C1
            self.C2_r = (1-np.heaviside(self.no_change_count-50,1))*self.L_C2 + np.heaviside(self.no_change_count-50,1)*self.G_C2
            self.w_r = (1-np.heaviside(self.no_change_count-100,1))*self.G_w + np.heaviside(self.no_change_count-100,1)*self.L_w
            print("C1: ",self.C1_r, " C2: ",self.C2_r, " W:", self.w_r, "No change: ",self.no_change_count)
            '''We now update the particle velocity'''
            self.r1, self.r2 = np.random.rand(), np.random.rand()

            self.Particle_Velocity = self.w_r*self.Particle_Velocity + self.C1_r*self.r1*(self.Particle_best_positions - self.Particle_Positions) + self.C2_r*self.r2*(self.Global_best_position - self.Particle_Positions) + self.Particle_Velocity*0.001*(np.random.rand(*self.Particle_Velocity.shape)-0.5)*2
            self.ax.plot(self.Global_best_position.flatten()-self.v.flatten())
            plt.pause(0.1)
            '''Defining a break condition if the optimisation is not showing any improvment'''
            self.Last_global_best = self.Global_best_performance
            if abs(self.Last_global_best-self.Global_best_performance) < 0.001:
                self.no_change_count +=1
            else:
                self.no_change_count = 0
            if self.no_change_count > 50:
                self.no_change_count = 0
                self.Particle_Positions = np.array([self.Global_best_position  for _ in range(self.Particles)])
                self.Particle_Velocity =  np.array([(np.random.randn(*self.v.shape)-0.5)*4 for _ in range(self.Particles)])

        '''Define output plot'''
        self.Objective_function([self.t,[self.Global_best_position[:,0],self.Global_best_position[:,1]],self.k], True)
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_ylabel("Dimensionless surface exposure")
        self.ax.set_xlabel("Foil position")
        self.ax.set_ylim([0, 50])
        self.ax.axhline(22, 0,207)
        self.ax.plot(self.surface_exposure)
        plt.show()

        self.fig, self.ax = plt.subplots()
        plt.title("Optimum solution")
        self.ax.set_aspect('equal')
        self.ax.plot(self.Tip_points[:,0],self.Tip_points[:,1])
        self.nodes = interpolate.splev(np.linspace(0,1,200, endpoint = False),[self.t,[self.Global_best_position[:,0],self.Global_best_position[:,1]],self.k])
        self.ax.plot(self.nodes[0],self.nodes[1])
        self.ax.plot(self.offset_points[:,0],self.offset_points[:,1])
        plt.show()

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.loss_array)
        plt.title("Optimisation loss")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        plt.show()

        self.trajectory = Trajectory(self.Tip_points, self.trajectory_nodes)
        self.surface_exposure = self.trajectory.predict_sde()
        plt.plot(self.surface_exposure)#self.trajectory.perspective_matrix*self.trajectory.visibility_matrix)
        plt.show()



        '''TAAAAA DAAAA'''
        '''Like magic'''
        '''I swear this should have been harder'''



>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
