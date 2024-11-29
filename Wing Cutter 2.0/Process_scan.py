import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import alphashape
from scipy.spatial.distance import directed_hausdorff
from scipy import interpolate

'''
Problems to fix?

How do we quantitativly compare point cloud data?




'''
class main():
    def __init__(self):
        plt.rcParams["font.family"] = "Times New Roman"
        self.foil_address = "C:/Users/rollo/Documents/University Content/Year 4/Advanced Research Project/Scans"
        self.corrected_foil = mesh.Mesh.from_file(self.foil_address+'/MeshBody1.stl')
        self.dumb_foil = mesh.Mesh.from_file(self.foil_address+'/MeshBody2.stl')
        self.foil_addr = "Airfoils//S1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]*200
        self.mainloop()

    def format_dat(self,data):
        # TODO: Needs Improvment for greater compatibility
        #Grandfarthered from old version
        self.data = data.split("\n")
        self.formatted = [self.el.split(" ") for self.el in self.data]
        self.formatted  = [[float(self.num) for self.num in list(filter(lambda x:x!='',self.coord))]for self.coord in self.formatted]#list(map(float,self.formatted))
        self.formatted = list(filter(lambda x:x!=[],self.formatted))
        return self.formatted

    def export(self):
        #print(self.points_hull_1)
        self.tck, _ = interpolate.splprep([self.points_hull_1[0],self.points_hull_1[1]],k=5,s=0.1,per=True) #Continuious directrie for root rail
        self.Samples = np.linspace(0,1,60, endpoint = True)
        self.p = interpolate.splev(self.Samples,self.tck)
        for i in range(0, len(self.p[0])):
            print(round(self.p[0][i],2),"   ",round(self.p[1][i],2))

        #print(self.points_hull_2)

    def mainloop(self):
        self.points = self.corrected_foil.points[:,1:3]
        self.pointss = self.dumb_foil[:,1:3]
        self.points = self.points - np.min(self.points,axis = 0)
        self.pointss = self.pointss - np.min(self.pointss,axis = 0)
        self.foil_dat = self.foil_dat - np.min(self.foil_dat,axis = 0)
        self.points_2 = []
        for point in self.points:

            if (11<point[1]<14) and (179 <point[0]<183):
                print(point)
                pass
            else:
                self.points_2.append(point/10)
        self.points = np.array(self.points_2)
        '''form concave hulls'''
        self.points_gt = [(point[0], point[1])for point in self.foil_dat]
        self.points_2d = [(point[0], point[1])for point in self.points]
        self.pointss_2d = [(point[0], point[1])for point in self.pointss]
        self.hull_1 = alphashape.alphashape(self.points_2d, 0.2)
        self.hull_2 = alphashape.alphashape(self.pointss_2d, 0.2)
        self.points_hull_1 = self.hull_1.exterior.coords.xy
        self.points_hull_2 = self.hull_2.exterior.coords.xy
        self.export()
        print("Corrected path ",max(directed_hausdorff(self.points_gt,self.points_2d)[0],directed_hausdorff(self.points_2d,self.points_gt)[0])," mm")
        print("Fixed path ",max(directed_hausdorff(self.points_gt,self.pointss_2d)[0],directed_hausdorff(self.pointss_2d,self.points_gt)[0])," mm")

        #plt.scatter(self.points[200:,0],self.points[200:,1])
        plt.plot(self.points_hull_1[0],self.points_hull_1[1],label = "Adaptive kerf Path", linestyle = (0, (3, 1, 1, 1)), c = "black")
        plt.plot(self.points_hull_2[0],self.points_hull_2[1],label = "Fixed kerf Path",linestyle = (0, (5, 10)), c = "black")
        plt.plot(self.foil_dat[:,0],self.foil_dat[:,1],label = "Ground Truth",linestyle = (0,()),c = "black")
        plt.gca().set_aspect('equal')
        plt.xlabel("X position [mm]")
        plt.ylabel("Y position [mm]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        self.figManager = plt.get_current_fig_manager()
        self.figManager.window.showMaximized()
        plt.savefig('Foil_Comparison.png', dpi = 450)
        plt.show()


if __name__=="__main__":
    main()
