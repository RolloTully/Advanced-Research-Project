import numpy as np
import matplotlib.pyplot as plt
import timeit
class Main():
    def __init__(self):
        self.foil_addr = "Airfoils//s1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]
        self.mainloop()

    def format_dat(self,data):
        # TODO: Needs Improvment for greater compatibility
        #Grandfarthered from old version
        self.data = data.split("\n")
        self.formatted = [self.el.split(" ") for self.el in self.data]
        self.formatted  = [[float(self.num) for self.num in list(filter(lambda x:x!='',self.coord))]for self.coord in self.formatted]#list(map(float,self.formatted))
        self.formatted = list(filter(lambda x:x!=[],self.formatted))
        return self.formatted

    def Offset(self, nodes, offsets):
        self.nodes = np.concatenate([nodes[-1:],nodes,nodes[:1]],axis=0)
        self.derivatives = self.nodes[1:]-self.nodes[:-1]#np.diff(nodes,axis=0)
        self.Unit_derivaitves = self.derivatives/np.sqrt(np.sum(self.derivatives**2,axis=1))[:,None]
        self.Offset_unit_vectors = self.Unit_derivaitves[:,::-1]
        self.Offset_unit_vectors[:,0] *= -1
        self.Offset_unit_vectors = (self.Offset_unit_vectors[1:]+self.Offset_unit_vectors[:-1])/2
        self.Offset_vectors = self.Offset_unit_vectors*self.offsets[:,None]
        return nodes+self.Offset_vectors

    def mainloop(self):

        self.nodes = self.foil_dat.copy()*100
        self.offsets = np.random.rand(self.nodes.shape[0])*-10
        print(self.offsets)
        self.data = self.Offset(self.nodes, self.offsets)

        plt.plot(self.nodes[:,0],self.nodes[:,1])
        plt.plot(self.data[:,0],self.data[:,1])
        plt.gca().set_aspect('equal')
        plt.show()


if __name__ == "__main__":
    Main()
