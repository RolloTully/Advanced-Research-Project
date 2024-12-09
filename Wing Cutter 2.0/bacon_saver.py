np.sqrt(np.sum(np.square(wire_panel.mid_point-surface_panel.mid_point)))

self.m = wire_panel.mid_point[1]-surface_panel.mid_point[1]/wire_panel.mid_point[0]-surface_panel.mid_point[0]
      self.c = wire_panel.points[0,1]-wire_panel.points[0,0]*self.m
      self.panel_distance = np.sqrt(np.sum(np.square(wire_panel.mid_point-surface_panel.mid_point)))#distance between the current surface panel of interest and the trajectory panel of interest
      '''we know this ray intersect the current panel, but does it intersect another panel?, and is this panel closer to the wire?'''
      self.add = True
      for n, sp in enumerate(self.surface_panels):
          if sp.does_intersect(self.m,self.c):
              self.new_panel_distance  = np.sqrt(np.sum(np.square(wire_panel.mid_point-sp.mid_point)))
              if self.new_panel_distance<self.panel_distance:
                  #not first surface
                  self.add = False
              '''the ray intersects so we calcuate the distance'''
              self.visibility_matrix[s_n,w_n] = True
      self.visibility_matrix[s_n,w_n] = self.add
'''We know what panels are visable from where but we need to know what area each panel presents to the wire'''
'''we now calculate how much each surface panel is irradated by each visible trajectory panel'''
