import numpy as np
from mayavi import mlab

# data preparation
orgPC = np.random.normal(size=(1000, 3))
roadPC = np.random.normal(size=(500, 3))
vehiclePC = np.random.normal(size=(500, 3))

# merge all clouds
coloredPoints = []
coloredPoints.append(orgPC)
coloredPoints.append(roadPC)
coloredPoints.append(vehiclePC)
colorizedPC = np.array(coloredPoints)

# define a colormap
colormap = np.random.random_sample((100,3))

mlab.figure(bgcolor=(1,1,1))
for pc in range(0, 3):
    colorIndex = pc
    mlab.points3d(colorizedPC[pc][:,0],colorizedPC[pc][:,1],colorizedPC[pc][:,2], np.ones(len(colorizedPC[pc])), color=tuple(colormap[colorIndex]), colormap="spectral", scale_factor=0.25)
mlab.show()



