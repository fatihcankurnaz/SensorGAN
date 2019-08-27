import os.path
import time

import numpy as np
import pickle
import utils.helpers.PC2ImageConverter as PC2ImageConverter
import matplotlib.pyplot as plt


from utils.helpers.visualizer import Vis

def decomposeCloud(rawCloud, verbose=False):
    # decompose cloud
    backgrdPoints = []
    roadPoints = []
    vehPoints = []
    pedPoints = []
    cycPoints = []

    for i in range(0, len(rawCloud)):
        objClass = rawCloud[i, 4]
        if objClass == "road":
            roadPoints.append(rawCloud[i,:])
        elif objClass == "car":
            vehPoints.append(rawCloud[i,:])
        elif objClass == "person":
            pedPoints.append(rawCloud[i,:])
        elif objClass == "cyclist":
            cycPoints.append(rawCloud[i,:])
        elif objClass == "None":
            backgrdPoints.append(rawCloud[i,:])

    backgrdCloud = np.asarray(backgrdPoints)
    roadCloud = np.asarray(roadPoints)
    vehCloud = np.asarray(vehPoints)
    pedCloud = np.asarray(pedPoints)
    cycCloud = np.asarray(cycPoints)

    if verbose:
        print ("background cloud: " + str(backgrdCloud.shape))
        print ("roadCloud cloud: " + str(roadCloud.shape))
        print ("vehCloud cloud: " + str(vehCloud.shape))
        print ("pedCloud cloud: " + str(pedCloud.shape))
        print ("cycCloud cloud: " + str(cycCloud.shape))


    return backgrdCloud, roadCloud, vehCloud, pedCloud, cycCloud

def loadBoundingBox(boundingBox):
    with open(boundingBox,'rb') as f:
        return pickle.load(f,encoding='bytes')

def parseBB3D(curr_path, bb3D_path):
    '''
    _BOundingbox : n* [ label_type,
                        [ [x1,x2,x3,x4,x5,x6,x7,x8],
                          [y1, ,,, ,,, ,,, ,,, ,y8],
                          [z1, ... ... ... ... ,z8]
                        ]
                      ]
                      for BoundingBox, x,y,z are in image coordinate
    '''
    pathName, tempName = os.path.split(curr_path)
    currFileName, _ = tempName.split(".")
    bbFileName = bb3D_path + currFileName.replace('full_label', 'bb3d') + ".bin"
    print(bbFileName)
    boundingbox_3d = []
    if os.path.exists(bbFileName):
        boundingbox_3d = loadBoundingBox(bbFileName)
    else:
        print ("ERROR: BB_3D file does not exist " + str(bbFileName))
        return None

    return boundingbox_3d

def insertLabelColumn(inputCloud):
    """ we insert an additional column representing the label id as int"""
    columnList = []
    for p in range(0, len(inputCloud)):

        label = inputCloud[p, 4]

        if label == 'None':
            columnList.append(0)
        elif label == 'road':
            columnList.append(1)
        elif label == 'car':
            columnList.append(2)
        elif label == 'person':
            columnList.append(3)
        elif label == 'cyclist':
            columnList.append(4)

    newColumn = np.asarray(columnList)
    inputCloud = np.insert(inputCloud, 5, newColumn, axis=1)

    return inputCloud

def processData(cloudName, bb3D_path, PC2ImgConv, outputFileName):

    timeStart = time.time()

    # load pc
    colorizedPC = np.load(cloudName,allow_pickle=True)  # colorizedPC has 7 columns [x y z i 'LabelStr' image_x image_y]
    # add label column
    labeledPC = insertLabelColumn(colorizedPC)  # labeledPC has now 8 columns [x y z i 'LabelStr' LabelID image_x image_y]

    # generate bird eye view image and cloud
    bevImage, bevCloud = PC2ImgConv.getBirdEyeViewImage(labeledPC, segValColIndex=5)
    #print(" bevImage size  " + str(bevImage.shape))

    timeElapsed = time.time() - timeStart
    #print(" bevImage generation took  " + str(timeElapsed))

    if True:
        visualizer = Vis()
        # get bounding boxes
        bb3D = [1] #parseBB3D(cloudName,bb3D_path)#bb3D_path

        if bb3D is not None:
            # decompose cloud into object clouds
            backgrdCloud, roadCloud, vehCloud, pedCloud, cycCloud = decomposeCloud(colorizedPC, verbose=False)

            visualizer.showCloudsWithBBs(orgPC=backgrdCloud, fovPC=bevCloud,  roadPC=roadCloud,
                                         vehPC=vehCloud, pedPC=pedCloud, cycPC=cycCloud, bb3D=[],
                                         fileName=outputFileName)

    if False:
        fig, axes = plt.subplots(6, 1, sharey=True)
        for r in range(0, 6):
            axes[r].imshow(bevImage[:, :, r])
            axes[r].set_axis_off()
            plt.axis('off')

        plt.show()

    if False:
        # print(type(bevImage[:,:,5]))
        np.savez_compressed(outputFileName, data=bevImage[:, :, 5])
        # plt.axis("off")
        # plt.imshow(bevImage[:,:,5])
        # plt.show()
def main():

    velo_path = "/SPACE/DATA/KITTI_Data/KITTI_object_labeled/Velo_labeled/"
    bb3D_path = "/SPACE/DATA/KITTI_Data/KITTI_object_labeled/BB_3d/"
    output_path = "/home/fatih/my_git/sensorgan/scripts/output/"

    fileList = os.listdir(velo_path)
    fileList.sort()  # good initial sort but doesnt sort numerically very well
    sorted(fileList)  # sort numerically in ascending order

    cloud_list = []  # list of image filenames

    # count the total number of frames in all subdirectories
    for file in fileList:

        if 'full_label_' in file and '.npy' in file:
            cloud_list.append(velo_path + file)
        else:
            print ("\n ------ ERROR unknown file format: " + velo_path +  file + "\n")

    print (" input path: " + str(velo_path))
    print (" total Cloud: " + str(len(cloud_list)))

    """ ---------- point cloud to image converter instance -------"""
    # version 02
    PC2ImgConv = PC2ImageConverter.PC2ImgConverter(imgChannel=5, xRange=[0, 25], yRange=[-6, 12], zRange=[-10, 8],
                                                   xGridSize=0.1, yGridSize=0.15, zGridSize=0.3, maxImgHeight=128,
                                                   maxImgWidth=256, maxImgDepth=64)


    frame_no=200
    currCloudFullPathName = cloud_list[frame_no]
    print(currCloudFullPathName)
    _, fullCloudName = os.path.split(currCloudFullPathName)
    currCloudName, _ = fullCloudName.split(".")
    outputFileName = output_path + currCloudName.replace('full_label', 'cloud') + ".png"

    print ("processing frame: " + str(fullCloudName) + " (" + str(frame_no) + " / " + str(len(cloud_list)) + ")" )
    processData(currCloudFullPathName, bb3D_path, PC2ImgConv, outputFileName)

if __name__ == '__main__':

    main()
