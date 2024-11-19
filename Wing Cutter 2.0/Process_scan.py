import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


'''
Problems to fix?

How do we quantitativly compare point cloud data?




'''
class main():
    def __init__(self):
        self.foil_address = "C:/Users/rollo/Documents/University Content/Year 4/Advanced Research Project/Scans"
        self.corrected_foil = mesh.Mesh.from_file(self.foil_address+'/MeshBody1-Corr.stl')
        self.dumb_foil = mesh.Mesh.from_file(self.foil_address+'/MeshBody1-Dumb.stl')
        self.mainloop()
    def mainloop(self):
        self.points = self.corrected_foil.points[:,1:3]
        self.pointss = self.dumb_foil[:,1:3]

        self.centroid = np.mean(self.points, axis=0)
        self.angles = np.arctan2(self.points[:, 1] - self.centroid[1], self.points[:, 0] - self.centroid[0])
        self.t_values = (self.angles - self.angles.min()) / (self.angles.max() - self.angles.min())
        print(self.points.shape)
        self.spline, _ = splprep([self.points[:,0],self.points[:,1]], u=self.t_values, k=3, s=0.01, per=True)
        plt.scatter(self.points[:,0],self.points[:,1],s=1)
        plt.scatter(self.pointss[:,0],self.pointss[:,1],s=1)
        plt.show()


if __name__=="__main__":
    main()
