import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
from tqdm import tqdm
import numba as nb
from matplotlib import cm
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
        self.critical_surface_energy = 1 #<<<<<<-----------This needs to be calibrated from experiemental analysis.
        self.free_surface_critical_energy = 1
        self.close_surface_critical_energy = 1

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
        [panel.update() for panel in self.surface_panels]
        [panel.update() for panel in self.wire_trajectory_panels]
        self.mid_points = np.array([panel.mid_point for panel in self.surface_panels])
        self.surface_irradance = np.zeros_like(self.surface_panels) #Keeps track of how much energy a panel has been exposed to.
        '''
        to accelerate this calculation we can calculate what regions of the trajectory are visable from each each panel, this means we dont have to compute all point for every panel
        this step is hugely computationally expensive.
        '''
        self.wire_midpoints = np.array([panel.mid_point for panel in self.wire_trajectory_panels])  # Shape: (num_wires, 2)
        self.wire_vectors = np.array([panel.panel_vector for panel in self.wire_trajectory_panels])  # Shape: (num_wires, 2)
        self.surface_midpoints = np.array([panel.mid_point for panel in self.surface_panels])  # Shape: (num_surfaces, 2)
        self.surface_vectors = np.array([panel.panel_vector for panel in self.surface_panels])  # Shape: (num_surfaces, 2)
        self.surface_areas = np.array([panel.area for panel in self.surface_panels])  # Shape: (num_surfaces,)
        self.wire_areas = np.array([panel.area for panel in self.wire_trajectory_panels])
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
            for n, panel in enumerate(self.surface_panels):
                self.axs.plot(panel.points[:,0],panel.points[:,1],c=cm.jet(self.surface_exposure[n]/np.max(self.surface_exposure)))
            plt.gca().set_aspect('equal')
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
        self.foil_addr = "Airfoils//s1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]
        self.mainloop()

    def Discretize(self, points):
        '''to correctly discretise we have to close the path'''
        points = np.concatenate([points[0],[points[0,0]]],axis=0)#this closes the loop
        #print(points.shape)
        return np.array([Panel(points[n], points[n+1]) for n in range(0,points.shape[0]-1)]) #Turns the continuious surface into a set of discrete panels, points are distributed evenly along the parametrisation

    def _curve(self, panels, offset_vector):
        self.offset_points = []
        for  n in range(0, points.shape[0]-1):
            '''We need to calculate the surface vector at the point, this is the rotational average of the 2 neighboring panels'''

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

    def Auto_Differentiation(self):
        pass

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
        plt.legend()
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
            #print(panel)
            self.x = np.linspace(panel.points[0,0],panel.points[1,0],5)
            #print("M, C")
            #print(panel.m, panel.c)
            self.y = panel.m*self.x+panel.c
            self.axs.plot(self.x,self.y, c= "Green")
        plt.gca().set_aspect('equal')
        plt.show()



if __name__ == "__main__":
    main()
