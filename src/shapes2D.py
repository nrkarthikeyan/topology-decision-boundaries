import numpy as np
class disk2D:
    """ Disk class for 2D """
    def __init__(self, c, r):
        """
            c - center (tuple)
            r - radius
        """
        self.c = c
        self.r = r
        
    def samples(self, ns):
        thetas = np.random.uniform(low = 0.0, high = 2.0*np.pi, size = ns)
        rs = np.random.uniform(low = 0.0, high = self.r, size = ns)
        samp = np.vstack((rs*np.cos(thetas), rs*np.sin(thetas))).T +np.array(self.c)
        return samp
    
class annulus2D:
    """ Annulus class for 2D """
    def __init__(self, c, rlow, rhigh):
        """
            c - center (tuple)
            rlow, rhigh - low and high radii
        """
        self.c = c
        self.rlow = rlow
        self.rhigh = rhigh
        
    def samples(self, ns):
        thetas = np.random.uniform(low = 0.0, high = 2.0*np.pi, size = ns)
        rs = np.random.uniform(low = self.rlow, high = self.rhigh, size = ns)
        samp = np.vstack((rs*np.cos(thetas), rs*np.sin(thetas))).T +np.array(self.c)
        return samp