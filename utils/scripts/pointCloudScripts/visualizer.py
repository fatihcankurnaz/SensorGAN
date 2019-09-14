import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt

# set parameters
zoomFactor = 4.0  # 2.0
pointSize = 0.3  # 0.2
azimuthAngle = 180
width = 1600
height = 1200
skip_points = 10
azimuthVal=-135
elevationVal=55
distanceVal=640
focalpointVal=[8.49302292, -5.28041458,  2.11626339]

class Vis(object):

    def __init__(self, bcgColor=(1,1,1), pointSize = 0.2):
        self.bcgColor = bcgColor
        self.pointSize = pointSize
        self.colorOrgPC = [0.8, 0.8, 0.8]  # original raw point cloud color
        self.colorFovPC = [69/255.0, 69/255.0, 69/255.0]  # field of view point cloud color
        self.colorRoadPC = [128/255.0, 64/255.0, 128/255.0]  # road point cloud color
        self.colorVehPC = [0, 0, 142/255.0]  # vehicle point cloud color
        self.colorPedPC = [219/255.0, 19/255.0, 60/255.0]  # pedestrian point cloud color
        self.colorCycPC = [119/255.0, 10/255.0, 32/255.0]  # cyclist point cloud color

    def showGrayImage(self, img=[]):

        mlab.imshow(img, colormap='gist_earth')
        mlab.show()

    def showRGBImage(self, img=[], bgr_2_rgb=True):

        if bgr_2_rgb:
            img = img[:, :, ::-1] # convert color from BGR to RGB

        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def saveRGBImage(self, img, frameID, imgPath, imgName="outputImg", bgr_2_rgb=True):

        if bgr_2_rgb:
            img = img[:, :, ::-1]  # convert color from BGR to RGB

        fileName = "%s%s_%06i.png" % (imgPath,imgName,frameID)
        plt.imsave(fileName,img)

    def showClouds(self, orgPC=[], fovPC=[], roadPC=[], vehPC=[], pedPC=[], cycPC=[], make_sparse=True):

        mlab.figure(bgcolor=self.bcgColor)

        # make the original point cloud sparse by removing every second point
        if make_sparse:
            if len(orgPC):
                orgPC = np.delete(orgPC, list(range(0, orgPC.shape[0], 2)), axis=0)

            if len(orgPC):
                orgPC = np.delete(orgPC, list(range(0, orgPC.shape[0], 2)), axis=0)

            if len(orgPC):
                orgPC = np.delete(orgPC, list(range(0, orgPC.shape[0], 2)), axis=0)

        # colorize point clouds
        if len(orgPC):
            mlab.points3d(orgPC[:, 0], orgPC[:, 1], orgPC[:, 2], np.ones(len(orgPC)), color=tuple(self.colorOrgPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(fovPC):
            mlab.points3d(fovPC[:, 0], fovPC[:, 1], fovPC[:, 2], np.ones(len(fovPC)), color=tuple(self.colorFovPC),
                          colormap="spectral", scale_factor=self.pointSize)
        if len(roadPC):
            mlab.points3d(roadPC[:, 0], roadPC[:, 1], roadPC[:, 2], np.ones(len(roadPC)), color=tuple(self.colorRoadPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(vehPC):
            mlab.points3d(vehPC[:, 0], vehPC[:, 1], vehPC[:, 2], np.ones(len(vehPC)), color=tuple(self.colorVehPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(pedPC):
            mlab.points3d(pedPC[:, 0], pedPC[:, 1], pedPC[:, 2], np.ones(len(pedPC)), color=tuple(self.colorPedPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(cycPC):
            mlab.points3d(cycPC[:, 0], cycPC[:, 1], cycPC[:, 2], np.ones(len(cycPC)), color=tuple(self.colorCycPC),
                          colormap="spectral", scale_factor=self.pointSize)

        mlab.show()

    def showCloudsWithBBs(self, orgPC=[], fovPC=[], roadPC=[], vehPC=[], pedPC=[], cycPC=[], bb3D=[], fileName="", make_sparse=True):

        mlab.figure(bgcolor=(1, 1, 1), size=(width, height))
        figure = mlab.gcf()

        # make the original point cloud denser by removing every second point
        if make_sparse:
            if len(orgPC):
                orgPC = np.delete(orgPC, list(range(0, orgPC.shape[0], 2)), axis=0)

            if len(orgPC):
                orgPC = np.delete(orgPC, list(range(0, orgPC.shape[0], 2)), axis=0)

            if len(orgPC):
                orgPC = np.delete(orgPC, list(range(0, orgPC.shape[0], 2)), axis=0)

        # colorize point clouds
        if len(orgPC):
            x, y, z = self.getCoordinates(orgPC)
            mlab.points3d(x, y, z, np.ones(len(orgPC)), color=tuple(self.colorOrgPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(fovPC):
            x, y, z = self.getCoordinates(fovPC)
            mlab.points3d(x, y, z, np.ones(len(fovPC)), color=tuple(self.colorFovPC),
                          colormap="spectral", scale_factor=self.pointSize)
        if len(roadPC):
            x, y, z = self.getCoordinates(roadPC)
            mlab.points3d(x, y, z, np.ones(len(roadPC)), color=tuple(self.colorRoadPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(vehPC):
            x, y, z = self.getCoordinates(vehPC)
            mlab.points3d(x, y, z, np.ones(len(vehPC)), color=tuple(self.colorVehPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(pedPC):
            x, y, z = self.getCoordinates(pedPC)
            mlab.points3d(x, y, z, np.ones(len(pedPC)), color=tuple(self.colorPedPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if len(cycPC):
            x, y, z = self.getCoordinates(cycPC)
            mlab.points3d(x, y, z, np.ones(len(cycPC)), color=tuple(self.colorCycPC),
                          colormap="spectral", scale_factor=self.pointSize)

        if bb3D:
            self.draw_3D_bboxes(bb3D, figure, draw_text=False)

        # set camera parameters
        figure.scene.camera.zoom(zoomFactor)
        figure.scene.camera.azimuth(azimuthAngle)
        mlab.view(azimuth=azimuthVal, elevation=elevationVal, distance=distanceVal, focalpoint=focalpointVal)

        if fileName:
        # save the figure
            print(" saving result " + fileName)
            mlab.savefig(fileName)
            mlab.close()
        else:
            mlab.show()

    def draw_3D_bboxes(self, bboxes3d, fig, line_width=2, draw_text=True, text_scale=(1, 1, 1)):

        num = len(bboxes3d)
        for n in range(num):
            b = bboxes3d[n]
            if b[0] != "None":

                if b[0] == "car":
                    color = tuple(self.colorVehPC)
                elif b[0] == "person":
                    color = tuple(self.colorPedPC)
                elif b[0] == "cyclist":
                    color = tuple(self.colorCycPC)
                else:
                    color = (0, 0, 0)

                if draw_text:
                    mlab.text3d(b[1][0,0], b[1][1,0], b[1][2,0], '%s'%b[0], scale=text_scale, color=color, figure=fig)
                for k in range(0,4):

                    i,j=k,(k+1)%4
                    #mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
                    mlab.plot3d([b[1][0,i], b[1][0,j]], [b[1][1,i], b[1][1,j]], [b[1][2,i], b[1][2,j]], color=color, tube_radius=None, line_width=line_width, figure=fig)

                    i,j=k+4,(k+1)%4 + 4
                    #mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
                    mlab.plot3d([b[1][0,i], b[1][0,j]], [b[1][1,i], b[1][1,j]], [b[1][2,i], b[1][2,j]], color=color, tube_radius=None, line_width=line_width, figure=fig)

                    i,j=k,k+4
                    #mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
                    mlab.plot3d([b[1][0,i], b[1][0,j]], [b[1][1,i], b[1][1,j]], [b[1][2,i], b[1][2,j]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        return fig

    def getCoordinates(self, pointCloud):

        x = []
        y = []
        z = []
        for p in range(0, len(pointCloud)):
            x.append(pointCloud[p, 0])
            y.append(pointCloud[p, 1])
            z.append(pointCloud[p, 2])
        return x,y,z

