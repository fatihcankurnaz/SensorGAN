from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import math


def sd_calc(data):
    n = len(data)

    if n <= 1:
        return 0.0

    mean, sd = avg_calc(data), 0.0

    # calculate stan. dev.
    for el in data:
        sd += (float(el) - mean)**2
    sd = math.sqrt(sd / float(n-1))

    return sd

def avg_calc(ls):
    n, mean = len(ls), 0.0

    if n <= 1:
        return ls[0]

    # calculate average
    for el in ls:
        mean = mean + float(el)
    mean = mean / float(n)

    return mean

def majority_vote(lst):

    final_val = 0

    if len(lst):
        #most_common_val  = max(set(lst), key=lst.count)
        most_common_val = max(lst)

        if most_common_val == 1:   # roadSegmentLabel
            final_val = 1
        elif most_common_val == 2: #vehicleSegmentLabel
            final_val = 2
        elif most_common_val == 3: #pedestrianSegmentLabel
            final_val = 3
        elif most_common_val == 4: #cycleSegmentLabel
            final_val = 4

    return final_val

def majority_vote_with_constraints(lst):

    final_val = 0
    totalPoint = len(lst)
    thresholdRoad = 80
    threshold = 50

    if totalPoint:
        rateBackground =100*lst.count(0)/totalPoint
        rateRoad = 100*lst.count(1)/totalPoint  # roadSegmentLabel
        rateVehicle = 100*lst.count(2)/totalPoint #vehicleSegmentLabel
        ratePedestrian = 100*lst.count(3)/totalPoint #pedestrianSegmentLabel
        rateCyclist = 100*lst.count(4)/totalPoint  #cycleSegmentLabel

        if rateBackground >= threshold:
            final_val= 0
        if rateRoad >= thresholdRoad:
            final_val= 1
        if rateVehicle >= threshold:
            final_val= 2
        if ratePedestrian >= threshold:
            final_val= 3
        if rateCyclist >= threshold:
            final_val= 4

        #print ("list: " + str(lst) + " b: " + str(rateBackground) + " r: " + str(rateRoad) + " f: " + str(final_val))

    return final_val

class PC2ImgConverter(object):

    def __init__(self, imgChannel=6, xRange=[0, 100], yRange=[-10, 10], zRange=[-10, 10], xGridSize=0.1, yGridSize=0.1,
                 zGridSize=0.1, maxImgWidth=512, maxImgHeight=64, maxImgDepth=64):

        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.zGridSize = zGridSize
        self.topViewImgWidth = np.int((xRange[1] - xRange[0]) / xGridSize)
        self.topViewImgHeight = np.int((yRange[1] - yRange[0]) / yGridSize)
        self.topViewImgDepth = np.int((zRange[1] - zRange[0]) / zGridSize)
        self.frontViewImgWidth = np.int((zRange[1] - zRange[0]) / zGridSize)
        self.frontViewImgHeight = np.int((yRange[1] - yRange[0]) / yGridSize)
        self.imgChannel = imgChannel
        self.maxImgWidth = maxImgWidth
        self.maxImgHeight = maxImgHeight
        self.maxImgDepth = maxImgDepth
        self.maxDim = 5000

        if self.topViewImgWidth > self.maxImgWidth or self.topViewImgHeight > self.maxImgHeight or self.topViewImgDepth > self.maxImgDepth:
            print("ERROR in top view image dimensions mismatch")

    def getBirdEyeViewImage(self, pointCloud, segValColIndex= 6, training=True):

        """ top view x-y projection of the input point cloud"""
        """ max image size maxImgWidth=512 times maxImgHeight=64 """
        timeStart = time.time()
        if training:
            topViewImage = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.imgChannel+1))
            imgGT = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        else:
            topViewImage = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.imgChannel))

        imgMean = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgMin = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgMax = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgStd = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgRef = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgDensity = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))

        tempMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        tempMatrix[:] = np.nan
        refMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        refMatrix[:] = np.nan
        topViewPoints = []
        if training:
            gtMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
            gtMatrix[:] = np.nan

        # compute top view points
        for p in range(0, len(pointCloud)):

            xVal = pointCloud[p][0]
            yVal = pointCloud[p][1]
            zVal = pointCloud[p][2]
            iVal = pointCloud[p][3]  # must be between 0 and 1

            if self.xRange[0] < xVal < self.xRange[1] and self.yRange[0] < yVal < self.yRange[1] and self.zRange[0] < zVal < self.zRange[1]:
                topViewPoints.append([xVal, yVal, zVal])
                pixelX = np.int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
                pixelY = np.int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
                imgDensity[pixelY,pixelX] += 1
                indexVal = np.int(imgDensity[pixelY,pixelX])
                if indexVal>= self.maxDim:
                    print ("ERROR in top view image computation: indexVal " + str(indexVal) + " is greater than maxDim " + str(self.maxDim))
                tempMatrix[pixelY, pixelX, indexVal] = zVal
                refMatrix[pixelY, pixelX, indexVal] = iVal
                if training:
                    gtMatrix[pixelY, pixelX, indexVal] = pointCloud[p][segValColIndex] #segVal

        # compute statistics
        for i in range(0, self.maxImgHeight):
            for j in range(0, self.maxImgWidth):
                currPixel = tempMatrix[i,j,:]
                currPixel = currPixel[~np.isnan(currPixel)]   # remove nans

                currRef = refMatrix[i,j,:]
                currRef = currRef[~np.isnan(currRef)]   # remove nans

                if training:
                    currGT = gtMatrix[i,j,:]
                    currGT = currGT[~np.isnan(currGT)]   # remove nans

                if len(currPixel):
                    imgMean[i,j] = np.mean(currPixel)
                    imgMin[i,j] = np.min(currPixel)
                    imgMax[i,j] = np.max(currPixel)
                    imgStd[i,j] = sd_calc(currPixel) #np.std(currPixel, ddof=0)
                    imgRef[i,j] = np.mean(currRef)
                    if training:
                        imgGT[i,j] = majority_vote_with_constraints(currGT.tolist())

        # convert to gray scale
        grayMean = convertMean(imgMean)
        grayMin = convertMean(imgMin)
        grayMax = convertMean(imgMax)
        grayStd = convertStd(imgStd)
        grayDensity = convertDensity(imgDensity)
        grayRef = convertReflectivity(imgRef)

        # place all computed images in a specific order
        topViewImage[:, :, 0] = grayMean
        topViewImage[:, :, 1] = grayMin
        topViewImage[:, :, 2] = grayMax
        topViewImage[:, :, 3] = grayRef
        topViewImage[:, :, 4] = grayDensity
        if training:
            topViewImage[:,:, 5] = imgGT

        topViewCloud = np.asarray(topViewPoints)

        timeElapsed = time.time() - timeStart
        print("Top view image computation took " + str(timeElapsed) + " ms ")
        return topViewImage, topViewCloud

    def getBirdEyeViewLayeredImage(self, pointCloud, segValColIndex= 6, training=True):

        """ bird-eye view x-y projection of the input point cloud"""
        """ max image size maxImgWidth=512 times maxImgHeight=64 times maxImgDepth=20 """
        timeStart = time.time()
        if training:
            topViewImage = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.maxImgDepth + 1))
            imgGT = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        else:
            topViewImage = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.maxImgDepth))

        densityMatrix = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.maxImgDepth))
        refMatrix = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.maxImgDepth))
        topViewPoints = []

        if training:
            gtMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxImgDepth, self.maxDim),
                                dtype=np.float32)
            gtMatrix[:] = np.nan

        # compute top view points
        for p in range(0, len(pointCloud)):

            xVal = pointCloud[p][0]
            yVal = pointCloud[p][1]
            zVal = pointCloud[p][2]
            iVal = pointCloud[p][3]  # must be between 0 and 1

            if self.xRange[0] < xVal < self.xRange[1] and self.yRange[0] < yVal < self.yRange[1] and self.zRange[
                0] < zVal < self.zRange[1]:
                topViewPoints.append([xVal, yVal, zVal])
                pixelX = np.int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
                pixelY = np.int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
                pixelZ = np.int(np.floor((zVal - self.zRange[0]) / self.zGridSize))
                densityMatrix[pixelY, pixelX, pixelZ] += 1
                refMatrix[pixelY, pixelX, pixelZ] += iVal
                indexVal = np.int(densityMatrix[pixelY, pixelX, pixelZ])
                if indexVal >= self.maxDim:
                    print("ERROR in bird eye view image computation: indexVal " + str(
                        indexVal) + " is greater than maxDim " + str(self.maxDim))

                if training:
                    gtMatrix[pixelY, pixelX, pixelZ, indexVal] = pointCloud[p][segValColIndex] # segVal

        # compute statistics
        for i in range(0, self.maxImgHeight):
            for j in range(0, self.maxImgWidth):

                if training:
                    labels = []

                for f in range(0, self.maxImgDepth):

                    if densityMatrix[i, j, f] > 0:
                        topViewImage[i, j, f] = convertReflectivity(refMatrix[i, j, f] / densityMatrix[i, j, f])

                    if training:
                        currGT = gtMatrix[i, j, f]
                        currGT = currGT[~np.isnan(currGT)]  # remove nans
                        labels = np.append(labels,currGT)

                if training:
                    imgGT[i, j] = majority_vote_with_constraints(labels.tolist())
                    topViewImage[i, j, self.maxImgDepth] = imgGT[i, j]

        topViewCloud = np.asarray(topViewPoints)

        timeElapsed = time.time() - timeStart
        print("Bird eye view image computation took " + str(timeElapsed) + " ms ")
        return topViewImage, topViewCloud

    def getFrontViewImage(self, pointCloud, segValColIndex= 6):

        """ front view y-z projection of the input point cloud"""
        """ max image size maxImgHeight=64 times maxImgHeight=64 """

        timeStart = time.time()
        frontViewImage = np.zeros(shape=(self.maxImgHeight, self.maxImgHeight, self.imgChannel))
        imgMean = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))
        imgMin = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))
        imgMax = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))
        imgStd = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))
        imgRef = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))
        imgDensity = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))
        imgGT = np.zeros(shape=(self.frontViewImgHeight, self.frontViewImgWidth))

        tempMatrix = np.empty(shape=(self.frontViewImgHeight, self.frontViewImgWidth, self.maxDim), dtype=np.float32)
        tempMatrix[:] = np.nan
        refMatrix = np.empty(shape=(self.frontViewImgHeight, self.frontViewImgWidth, self.maxDim), dtype=np.float32)
        refMatrix[:] = np.nan
        gtMatrix = np.empty(shape=(self.frontViewImgHeight, self.frontViewImgWidth, self.maxDim), dtype=np.float32)
        gtMatrix[:] = np.nan

        # compute top view points
        for p in range(0, len(pointCloud)):

            xVal = pointCloud[p][0]
            yVal = pointCloud[p][1]
            zVal = pointCloud[p][2]
            iVal = pointCloud[p][3]  # must be between 0 and 1
            segVal = pointCloud[p][segValColIndex]

            if  self.zRange[0] < zVal < self.zRange[1] and  self.yRange[0] < yVal < self.yRange[1]:
                pixelX = np.int(np.floor((zVal - self.zRange[0]) / self.zGridSize)-1)
                pixelY = np.int(self.frontViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize)-1)
                imgDensity[pixelY,pixelX] += 1
                indexVal = np.int(imgDensity[pixelY,pixelX])
                if indexVal>= self.maxDim:
                    print ("ERROR in front view image computation: indexVal " + str(indexVal) + " is greater than maxDim " + str(self.maxDim))
                tempMatrix[pixelY, pixelX, indexVal] = zVal
                refMatrix[pixelY, pixelX, indexVal] = iVal
                gtMatrix[pixelY, pixelX, indexVal] = segVal

        # compute statistics
        for i in range(0, self.frontViewImgHeight):
            for j in range(0, self.frontViewImgWidth):
                currPixel = tempMatrix[i,j,:]
                currPixel = currPixel[~np.isnan(currPixel)]   # remove nans

                currRef = refMatrix[i,j,:]
                currRef = currRef[~np.isnan(currRef)]   # remove nans

                currGT = gtMatrix[i,j,:]
                currGT = currGT[~np.isnan(currGT)]   # remove nans

                if len(currPixel):
                    imgMean[i,j] = np.mean(currPixel)
                    imgMin[i,j] = np.min(currPixel)
                    imgMax[i,j] = np.max(currPixel)
                    imgStd[i,j] = sd_calc(currPixel) #np.std(currPixel, ddof=0)
                    imgRef[i,j] = np.mean(currRef)
                    imgGT[i, j] = majority_vote(currGT.tolist())

        # convert to gray scale
        grayMean = convertMean(imgMean)
        grayMin = convertMean(imgMin)
        grayMax = convertMean(imgMax)
        grayStd = convertStd(imgStd)
        grayDensity = convertDensity(imgDensity)
        grayRef = convertReflectivity(imgRef)

        # place all computed images in a specific order
        frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 0] = grayMean
        frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 1] = grayMin
        frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 2] = grayMax
        frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 3] = grayRef
        frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 4] = grayDensity
        frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 5] = imgGT
        #frontViewImage[0:self.frontViewImgHeight, 0:self.frontViewImgWidth, 5] = grayStd

        timeElapsed = time.time() - timeStart
        print("Front view image computation took " + str(timeElapsed) + " ms ")
        return frontViewImage

    def getSphericalViewImage(self, pointCloud):

        """ spherical view projection of the input point cloud"""
        """ max image size maxImgWidth=512 times maxImgHeight=64 """
        timeStart = time.time()
        sphericalImage = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.imgChannel))
        imgX = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgY = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgZ = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgRef = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgDensity = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgRange = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))
        imgGT = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth))

        tempXMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        tempXMatrix[:] = np.nan

        tempYMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        tempYMatrix[:] = np.nan

        tempZMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        tempZMatrix[:] = np.nan

        refMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        refMatrix[:] = np.nan

        rangeMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        rangeMatrix[:] = np.nan

        gtMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        gtMatrix[:] = np.nan

        fovHor = [-0.55, 0.55]#[-0.785308, 0.785330]  # 90 degree
        fovVer = [1.45, 1.65]
        resHor = (fovHor[1] - fovHor[0]) / self.maxImgWidth
        resVer = (fovVer[1] - fovVer[0]) / self.maxImgHeight

        # compute spherical view points
        for p in range(0, len(pointCloud)):

            xVal = pointCloud[p][0]
            yVal = pointCloud[p][1]
            zVal = pointCloud[p][2]
            iVal = pointCloud[p][3]  # must be between 0 and 1
            segVal = pointCloud[p][segValColIndex]

            rVal = math.sqrt(pow(xVal,2)+pow(yVal,2)+pow(zVal,2))
            tVal = math.sqrt(pow(xVal,2)+pow(yVal,2))

            theta = math.atan2(tVal,zVal)
            phi = math.atan2(yVal,xVal)

            imgPhi = round(self.maxImgWidth-(phi-fovHor[0])/resHor)
            imgTheta = round((theta-fovVer[0])/resVer)

            if 0 <= imgTheta < self.maxImgHeight and 0 <= imgPhi < self.maxImgWidth:
                pixelX = np.int(imgPhi)
                pixelY = np.int(imgTheta)
                imgDensity[pixelY,pixelX] += 1
                indexVal = np.int(imgDensity[pixelY,pixelX])
                if indexVal>= self.maxDim:
                    print ("ERROR in top view image computation: indexVal " + str(indexVal) + " is greater than maxDim " + str(self.maxDim))

                tempXMatrix[pixelY, pixelX, indexVal] = xVal
                tempYMatrix[pixelY, pixelX, indexVal] = yVal
                tempZMatrix[pixelY, pixelX, indexVal] = zVal
                rangeMatrix[pixelY, pixelX, indexVal] = rVal
                refMatrix[pixelY, pixelX, indexVal] = iVal
                gtMatrix[pixelY, pixelX, indexVal] = segVal

        # compute statistics
        for i in range(0, self.maxImgHeight):
            for j in range(0, self.maxImgWidth):
                currX = tempXMatrix[i,j,:]
                currX = currX[~np.isnan(currX)]   # remove nans

                currY = tempYMatrix[i,j,:]
                currY = currY[~np.isnan(currY)]   # remove nans

                currZ = tempZMatrix[i,j,:]
                currZ = currZ[~np.isnan(currZ)]   # remove nans

                currRef = refMatrix[i,j,:]
                currRef = currRef[~np.isnan(currRef)]   # remove nans

                currRan = rangeMatrix[i,j,:]
                currRan = currRan[~np.isnan(currRan)]   # remove nans

                currGT = gtMatrix[i,j,:]
                currGT = currGT[~np.isnan(currGT)]   # remove nans

                if len(currX):
                    imgX[i,j] = np.mean(currX)
                    imgY[i,j] = np.mean(currY)
                    imgZ[i,j] = np.mean(currZ)
                    imgRef[i,j] = np.mean(currRef)
                    imgRange[i,j] = np.mean(currRan)
                    imgGT[i,j] = majority_vote(currGT.tolist())

        # convert to gray scale
        grayDensity = convertDensity(imgDensity)
        grayRef = convertReflectivity(imgRef)
        grayRange = convertReflectivity(imgRange)

        # place all computed images in a specific order
        sphericalImage[:, :, 0] = imgX
        sphericalImage[:, :, 1] = imgY
        sphericalImage[:, :, 2] = imgZ
        sphericalImage[:, :, 3] = grayRef
        sphericalImage[:, :, 4] = grayDensity
        sphericalImage[:, :, 5] = imgGT

        timeElapsed = time.time() - timeStart
        print("Spherical view image computation took " + str(timeElapsed) + " ms ")
        return sphericalImage

    def convertSphericalImageToCloud(self, predImg,inputImg):

        timeStart = time.time()
        vehiclePoints = []
        pedestrianPoints = []
        cyclistPoints = []

        for (x, y), value in np.ndenumerate(predImg):
            objID = predImg[x, y]
            xVal = inputImg[x, y, 0]
            yVal = inputImg[x, y, 1]
            zVal = inputImg[x, y, 2]
            if objID == 1:
                vehiclePoints.append([xVal, yVal, zVal])
            elif objID == 2:
                pedestrianPoints.append([xVal, yVal, zVal])
            elif objID == 3:
                cyclistPoints.append([xVal, yVal, zVal])

        vehicleCloud = np.asarray(vehiclePoints)
        pedestrianCloud = np.asarray(pedestrianPoints)
        cyclistCloud = np.asarray(cyclistPoints)

        timeElapsed = time.time() - timeStart
        print(" [OS] Object point cloud projection took " + str(timeElapsed) + " ms ")
        return vehicleCloud, pedestrianCloud, cyclistCloud

    def getCloudsFromTopViewImage(self, predImg, topViewCloud, postProcessing = False):
        """ crop topviewcloud based on the network prediction image  """
        roadPoints = []
        vehPoints = []

        for p in range(0, len(topViewCloud)):

            xVal = topViewCloud[p][0]
            yVal = topViewCloud[p][1]
            zVal = topViewCloud[p][2]
            pixelX = np.int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
            pixelY = np.int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
            classVal = predImg[pixelY, pixelX]


            if classVal == 1:
                roadPoints.append([xVal, yVal, zVal])
            elif classVal == 2:
                vehPoints.append([xVal, yVal, zVal])

        roadCloud = np.asarray(roadPoints)
        vehCloud = np.asarray(vehPoints)

        if postProcessing:

            # first global thresholding, make all points above 3  m as background
            globalThreshold = 3
            if len(roadCloud):
                roadCloud = roadCloud[roadCloud[:, 2] < globalThreshold]
            if len(vehCloud):
                vehCloud = vehCloud[vehCloud[:, 2] < globalThreshold]

            # second, apply thresholding only to road points
            # e.g. compute the mean of road points and remove those that are above
            if len(roadCloud):
                meanRoadZ = roadCloud[:, 2].mean()  # mean of third column, i.e. z values
                stdRoadZ = roadCloud[:, 2].std()  # mean of third column, i.e. z values
                roadThreshold = meanRoadZ + (1.0 * stdRoadZ)

                #print ("meanRoadZ: " + str(meanRoadZ) + " stdRoadZ: " + str(stdRoadZ) + " roadThreshold: " + str(roadThreshold))
                roadCloud = roadCloud[roadCloud[:, 2] < roadThreshold]


        return roadCloud, vehCloud

    def getCloudsFromBirdEyeViewImage(self, predImg, topViewCloud, postProcessing = False):
        """ crop topviewcloud based on the network prediction image  """
        roadPoints = []
        vehPoints = []

        for p in range(0, len(topViewCloud)):

            xVal = topViewCloud[p][0]
            yVal = topViewCloud[p][1]
            zVal = topViewCloud[p][2]
            pixelX = np.int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
            pixelY = np.int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
            pixelZ = np.int(np.floor((zVal - self.zRange[0]) / self.zGridSize))
            classVal = predImg[pixelY, pixelX]

            if classVal == 1:
                roadPoints.append([xVal, yVal, zVal])
            elif classVal == 2:
                vehPoints.append([xVal, yVal, zVal])

        roadCloud = np.asarray(roadPoints)
        vehCloud = np.asarray(vehPoints)

        if postProcessing:

            # first global thresholding, make all points above 3  m as background
            globalThreshold = 3
            if len(roadCloud):
                roadCloud = roadCloud[roadCloud[:, 2] < globalThreshold]
            if len(vehCloud):
                vehCloud = vehCloud[vehCloud[:, 2] < globalThreshold]

            # second, apply thresholding only to road points
            # e.g. compute the mean of road points and remove those that are above
            if len(roadCloud):
                meanRoadZ = roadCloud[:, 2].mean()  # mean of third column, i.e. z values
                stdRoadZ = roadCloud[:, 2].std()  # mean of third column, i.e. z values
                roadThreshold = meanRoadZ + (1.0 * stdRoadZ)

                #print ("meanRoadZ: " + str(meanRoadZ) + " stdRoadZ: " + str(stdRoadZ) + " roadThreshold: " + str(roadThreshold))
                roadCloud = roadCloud[roadCloud[:, 2] < roadThreshold]


        return roadCloud, vehCloud


"""------------- util functions -------------"""
def convertMean(input):
    output = input

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            p = input[i, j]
            output[i,j] = MapHeightToGrayscale(p)

    return output

def MapHeightToGrayscale(currHeight):

    medianRoadHeight = -1.6
    minHeight = -3
    maxHeight = 3
    delta = (maxHeight - minHeight) / 256
    deltaHeight = currHeight - medianRoadHeight;
    grayLevel = 0

    if deltaHeight >= maxHeight:
        grayLevel = 255
    elif deltaHeight <= minHeight:
        grayLevel = 0
    else:
        grayLevel = np.floor((deltaHeight - minHeight) / delta)

    if currHeight == 0:
        grayLevel = 0

    return grayLevel

def convertStd(input):
    output = input

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            p = input[i, j]
            output[i,j] = MapStdToGrayscale(p)

    return output

def MapStdToGrayscale(std):

    minStd = 0
    maxStd = 1
    delta = (maxStd - minStd) / 256
    grayLevel = 0

    if std >= maxStd:
        grayLevel = 255
    elif std <= minStd:
        grayLevel = 0
    else:
        grayLevel = np.floor((std - minStd) / delta)

    return grayLevel

def convertDensity(input):
    output = input

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            p = input[i, j]
            output[i,j] = MapDensityToGrayscale(p)

    return output

def MapDensityToGrayscale(density):

    minDensity = 0
    maxDensity = 16
    delta = (maxDensity - minDensity) / 256
    grayLevel = 0

    if density >= maxDensity:
        grayLevel = 255
    elif density <= minDensity:
        grayLevel = 0
    else:
        grayLevel = np.floor((density - minDensity) / delta)

    return grayLevel

def convertReflectivity(input):

    output = np.round(input*255)

    return output

