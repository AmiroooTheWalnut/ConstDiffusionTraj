import math

from fontTools.unicodedata import block
from sympy import false
from sympy.physics.quantum.circuitplot import pyplot

from TrajCell import TrajCell
from Traj import SubTraj
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import copy

class TilesSynthetic:
    def __init__(self, fullImage,numRows,numCols):
        img = Image.open(fullImage)
        pixels = img.load()
        self.fullImage = pixels
        self.numRows = numRows
        self.numCols = numCols
        self.width, self.height = img.size
        self.trajGrid = []
        # self.trajGridReady = []
        self.origStreetPoints=[]

    def makeSyntheticPaths(self,numPaths,numRows,numCols,marginPercentage,targetLength, seed, isVisualize=False):
        numpy.random.seed(seed)
        potentialPixels=[]
        for i in range(1,self.width):# FOR SOME REASON A-STAR FAILS WHEN TARGET OR START IS AT 0
            for j in range(1,self.height):# FOR SOME REASON A-STAR FAILS WHEN TARGET OR START IS AT 0
                if self.fullImage[i,j][0]<200:
                    potentialPixels.append([i,j])
        allPaths=[]
        for i in range(numPaths):
            for tryIndex in range(10000):
                startIndex = numpy.random.randint(0, high=len(potentialPixels))
                endIndex = numpy.random.randint(0, high=len(potentialPixels))
                startX = potentialPixels[startIndex][0]
                startY = potentialPixels[startIndex][1]
                endX = potentialPixels[endIndex][0]
                endY = potentialPixels[endIndex][1]
                if math.sqrt(math.pow(startX-endX,2)+math.pow(startY-endY,2))<10:
                    continue
                path = self.shortestPathImage(startX, startY, endX, endY, isVisualize=isVisualize)
                allPaths.append(path)
                break
        self.splitByGridAndMargin(allPaths, numRows, numCols, marginPercentage)
        self.upDownSample(targetLength)
        streetPoints=potentialPixels
        self.origStreetPoints=numpy.array(streetPoints,dtype=numpy.float32)
        maxX = numpy.max(self.origStreetPoints[:,0])
        minX = numpy.min(self.origStreetPoints[:,0])
        maxY = numpy.max(self.origStreetPoints[:, 1])
        minY = numpy.min(self.origStreetPoints[:, 1])
        adjustedStreetPoints=self.addExtraStreetPoints(streetPoints)
        adjustedStreetPoints=numpy.array(adjustedStreetPoints,dtype=numpy.float32)
        adjustedStreetPoints[:,0] = (adjustedStreetPoints[:, 0] - minX) / (maxX - minX)
        adjustedStreetPoints[:, 1] = (adjustedStreetPoints[:, 1] - minY) / (maxY - minY)
        adjustedStreetPoints = adjustedStreetPoints.transpose()
        # adjustedStreetPoints=adjustedStreetPoints.transpose()*scale

        allPathsAdjusted=[]
        cutLats = []
        cutLons = []
        for r in range(len(self.trajGrid)):
            for c in range(len(self.trajGrid[r])):
                for i in range(len(self.trajGrid[r][c])):
                    self.trajGrid[r][c][i].points=numpy.array(self.trajGrid[r][c][i].points,dtype=numpy.float32)
                    self.trajGrid[r][c][i].streetPoints = numpy.array(self.trajGrid[r][c][i].streetPoints,
                                                                      dtype=numpy.float32)
                    if maxX == minX:
                        self.trajGrid[r][c][i].points[:, 0] = 0.5
                        self.trajGrid[r][c][i].streetPoints[:, 0] = 0.5
                    else:
                        self.trajGrid[r][c][i].points[:, 0] = (self.trajGrid[r][c][i].points[:, 0] - minX) / (
                                    maxX - minX)
                        self.trajGrid[r][c][i].streetPoints[:, 0] = (self.trajGrid[r][c][i].streetPoints[:,
                                                                     0] - minX) / (maxX - minX)
                    if maxY == minY:
                        self.trajGrid[r][c][i].points[:, 1] = 0.5
                        self.trajGrid[r][c][i].streetPoints[:, 1] = 0.5
                    else:
                        self.trajGrid[r][c][i].points[:, 1] = (self.trajGrid[r][c][i].points[:, 1] - minY) / (
                                    maxY - minY)
                        self.trajGrid[r][c][i].streetPoints[:, 1] = (self.trajGrid[r][c][i].streetPoints[:,
                                                                     1] - minY) / (maxY - minY)

                    # self.trajGrid[r][c][i].points=self.trajGrid[r][c][i].points*scale


                    # self.trajGrid[r][c][i].streetPoints = self.trajGrid[r][c][i].streetPoints.transpose() * scale
                    self.trajGrid[r][c][i].streetPoints = self.trajGrid[r][c][i].streetPoints.transpose()
                    # self.trajGrid[r][c][i].initLat=(self.trajGrid[r][c][i].initLat - minX) / (maxX - minX)
                    # self.trajGrid[r][c][i].initLon=(self.trajGrid[r][c][i].initLon - minY) / (maxY - minY)
                    if self.trajGrid[r][c][i].cutLat==-1:
                        if maxX == minX:
                            self.trajGrid[r][c][i].cutLat = 0.5
                        else:
                            self.trajGrid[r][c][i].cutLat = (self.trajGrid[r][c][i].initLat - minX) / (maxX - minX)
                        if maxY == minY:
                            self.trajGrid[r][c][i].cutLon = 0.5
                        else:
                            self.trajGrid[r][c][i].cutLon = (self.trajGrid[r][c][i].initLon - minY) / (maxY - minY)

                    else:
                        if maxX == minX:
                            self.trajGrid[r][c][i].cutLat = 0.5
                        else:
                            self.trajGrid[r][c][i].cutLat = (self.trajGrid[r][c][i].cutLat - minX) / (maxX - minX)
                        if maxY == minY:
                            self.trajGrid[r][c][i].cutLon = 0.5
                        else:
                            self.trajGrid[r][c][i].cutLon = (self.trajGrid[r][c][i].cutLon - minY) / (maxY - minY)

                    cutLats.append(self.trajGrid[r][c][i].cutLat)
                    cutLons.append(self.trajGrid[r][c][i].cutLon)
                    allPathsAdjusted.append(self.trajGrid[r][c][i].points)

        cutLats = numpy.array(cutLats, dtype=numpy.float32)
        cutLons = numpy.array(cutLons, dtype=numpy.float32)
        initLats = numpy.array(allPathsAdjusted,dtype=numpy.float32)[:, 0, 0]
        initLons = numpy.array(allPathsAdjusted,dtype=numpy.float32)[:, 0, 1]

        return allPathsAdjusted, adjustedStreetPoints, initLats, initLons, cutLats, cutLons

    def shortestPathImage(self, startX, startY, endX, endY, isVisualize=False):
        currentX = startX
        currentY = startY
        f = numpy.zeros((self.width, self.height))-1
        g = numpy.zeros((self.width, self.height))-1
        h = numpy.zeros((self.width, self.height))-1
        isChecked = numpy.zeros((self.width, self.height))
        g[currentX, currentY] = 0
        h[currentX, currentY] = self.dist(currentX,currentY,endX,endY)*0.11
        f[currentX, currentY] = g[currentX, currentY]+h[currentX, currentY]
        isChecked[currentX, currentY] = 1
        neighbors = []
        # showCounter=0
        while currentX!=endX or currentY!=endY:
            # if currentX==47 and currentY==56:
            #     print("!!!")
            if currentX+1<self.width:
                if self.fullImage[currentX + 1, currentY][0] < 200:
                    newG = g[currentX, currentY] + 1
                    if newG < g[currentX + 1, currentY] or g[currentX + 1, currentY] < 0:
                        neighbors.append([currentX + 1, currentY])
                        g[currentX + 1, currentY] = newG
                        h[currentX + 1, currentY] = self.dist(currentX + 1, currentY, endX, endY) * 0.11
                        f[currentX + 1, currentY] = g[currentX + 1, currentY] + h[currentX + 1, currentY]
            if currentX + 1 < self.width and currentY+1<self.height:
                if self.fullImage[currentX + 1, currentY + 1][0] < 200:
                    newG = g[currentX, currentY] + math.sqrt(2)
                    if newG < g[currentX + 1, currentY + 1] or g[currentX + 1, currentY + 1] < 0:
                        neighbors.append([currentX + 1, currentY + 1])
                        g[currentX + 1, currentY + 1] = newG
                        h[currentX + 1, currentY + 1] = self.dist(currentX + 1, currentY + 1, endX, endY) * 0.11
                        f[currentX + 1, currentY + 1] = g[currentX + 1, currentY + 1] + h[currentX + 1, currentY + 1]
            if currentY + 1 < self.height:
                if self.fullImage[currentX, currentY + 1][0] < 200:
                    newG = g[currentX, currentY] + 1
                    if newG < g[currentX, currentY + 1] or g[currentX, currentY + 1] < 0:
                        neighbors.append([currentX, currentY + 1])
                        g[currentX, currentY + 1] = newG
                        h[currentX, currentY + 1] = self.dist(currentX, currentY + 1, endX, endY) * 0.11
                        f[currentX, currentY + 1] = g[currentX, currentY + 1] + h[currentX, currentY + 1]
            if currentX - 1 > 0 and currentY + 1 < self.height:
                if self.fullImage[currentX - 1, currentY + 1][0] < 200:
                    newG = g[currentX, currentY] + math.sqrt(2)
                    if newG < g[currentX - 1, currentY + 1] or g[currentX - 1, currentY + 1] < 0:
                        neighbors.append([currentX - 1, currentY + 1])
                        g[currentX - 1, currentY + 1] = newG
                        h[currentX - 1, currentY + 1] = self.dist(currentX - 1, currentY + 1, endX, endY) * 0.11
                        f[currentX - 1, currentY + 1] = g[currentX - 1, currentY + 1] + h[currentX - 1, currentY + 1]
            if currentX - 1 > 0:
                if self.fullImage[currentX - 1, currentY][0] < 200:
                    newG = g[currentX, currentY] + 1
                    if newG < g[currentX - 1, currentY] or g[currentX - 1, currentY] < 0:
                        neighbors.append([currentX - 1, currentY])
                        g[currentX - 1, currentY] = newG
                        h[currentX - 1, currentY] = self.dist(currentX - 1, currentY, endX, endY) * 0.11
                        f[currentX - 1, currentY] = g[currentX - 1, currentY] + h[currentX - 1, currentY]
            if currentX - 1 > 0 and currentY - 1 >0:
                if self.fullImage[currentX - 1, currentY - 1][0] < 200:
                    newG = g[currentX, currentY] + math.sqrt(2)
                    if newG < g[currentX - 1, currentY - 1] or g[currentX - 1, currentY - 1] < 0:
                        neighbors.append([currentX - 1, currentY - 1])
                        g[currentX - 1, currentY - 1] = newG
                        h[currentX - 1, currentY - 1] = self.dist(currentX - 1, currentY - 1, endX, endY) * 0.11
                        f[currentX - 1, currentY - 1] = g[currentX - 1, currentY - 1] + h[currentX - 1, currentY - 1]
            if currentY - 1 > 0:
                if self.fullImage[currentX, currentY - 1][0] < 200:
                    newG = g[currentX, currentY] + 1
                    if newG < g[currentX, currentY - 1] or g[currentX, currentY - 1] < 0:
                        neighbors.append([currentX, currentY - 1])
                        g[currentX, currentY - 1] = newG
                        h[currentX, currentY - 1] = self.dist(currentX, currentY - 1, endX, endY) * 0.11
                        f[currentX, currentY - 1] = g[currentX, currentY - 1] + h[currentX, currentY - 1]
            if currentX + 1 < self.width and currentY - 1 > 0:
                if self.fullImage[currentX + 1, currentY - 1][0] < 200:
                    newG = g[currentX, currentY] + math.sqrt(2)
                    if newG < g[currentX + 1, currentY - 1] or g[currentX + 1, currentY - 1] < 0:
                        neighbors.append([currentX + 1, currentY - 1])
                        g[currentX + 1, currentY - 1] = newG
                        h[currentX + 1, currentY - 1] = self.dist(currentX + 1, currentY - 1, endX, endY) * 0.11
                        f[currentX + 1, currentY - 1] = g[currentX + 1, currentY - 1] + h[currentX + 1, currentY - 1]
            minF=1000000
            minIndex=-1
            for m in range(len(neighbors)):
                if f[neighbors[m][0],neighbors[m][1]]<minF and isChecked[neighbors[m][0],neighbors[m][1]]==0:
                    minF=f[neighbors[m][0],neighbors[m][1]]
                    minIndex=m
            currentX = neighbors[minIndex][0]
            currentY = neighbors[minIndex][1]
            del neighbors[minIndex]
            isChecked[currentX, currentY] = 1

            # if showCounter>50:
            #     plt.imshow(f)
            #     plt.draw()
            #     # plt.pause(0.01)
            #     plt.show()
            #     showCounter=0
            # showCounter=showCounter+1
            # plt.show()
            # print("!!!")
        # plt.imshow(g)
        # plt.draw()
        # # plt.pause(0.01)
        # plt.show()
        # print("!!!")
        currentX=endX
        currentY=endY
        path=[]
        path2DVis=numpy.zeros((self.width, self.height))-1
        while currentX != startX or currentY != startY:
            neighbors = []
            if currentX + 1 < f.shape[0]:
                if f[currentX + 1, currentY] != -1:
                    neighbors.append([currentX + 1, currentY])
            if currentX + 1 < f.shape[0] and currentY + 1 < f.shape[1]:
                if f[currentX + 1, currentY + 1] != -1:
                    neighbors.append([currentX + 1, currentY + 1])
            if currentY + 1 < f.shape[1]:
                if f[currentX, currentY + 1] != -1:
                    neighbors.append([currentX, currentY + 1])
            if currentX - 1 > 0 and currentY + 1 < f.shape[1]:
                if f[currentX - 1, currentY + 1] != -1:
                    neighbors.append([currentX - 1, currentY + 1])
            if currentX - 1 > 0:
                if f[currentX - 1, currentY] != -1:
                    neighbors.append([currentX - 1, currentY])
            if currentX - 1 > 0 and currentY - 1 > 0:
                if f[currentX - 1, currentY - 1] != -1:
                    neighbors.append([currentX - 1, currentY - 1])
            if currentY - 1 > 0:
                if f[currentX, currentY - 1] != -1:
                    neighbors.append([currentX, currentY - 1])
            if currentX + 1 < f.shape[0] and currentY - 1 > 0:
                if f[currentX + 1, currentY - 1] != -1:
                    neighbors.append([currentX + 1, currentY - 1])
            minG = 1000000
            minIndex = -1
            for m in range(len(neighbors)):
                if g[neighbors[m][0], neighbors[m][1]] < minG and isChecked[neighbors[m][0],neighbors[m][1]]==1:
                    minG = g[neighbors[m][0], neighbors[m][1]]
                    currentX = neighbors[m][0]
                    currentY = neighbors[m][1]
                    minIndex=m
            if minIndex==-1:
                # plt.imshow(isChecked)
                # plt.draw()
                # # plt.pause(0.01)
                # plt.show()
                path2DVis[neighbors[-1][0], neighbors[-1][1]] = -1
                path.pop()
                currentX = path[-1][0]
                currentY = path[-1][1]
            else:
                path.append([neighbors[minIndex][0], neighbors[minIndex][1]])
                path2DVis[neighbors[minIndex][0], neighbors[minIndex][1]]=1
                isChecked[neighbors[minIndex][0], neighbors[minIndex][1]]=0
        if isVisualize==True:
            plt.imshow(path2DVis)
            plt.draw()
            # plt.pause(0.01)
            plt.show()
            # print("!!!")
        return numpy.array(path,dtype=numpy.float32)

    def splitByGridAndMargin(self,trajs,numRows,numCols,marginPercentage):
        self.marginPercentage=marginPercentage
        trajGrid=[]
        for r in range(numRows):
            row=[]
            for c in range(numCols):
                row.append([])
            trajGrid.append(row)
        # trajGridReady = []
        # for r in range(numRows):
        #     row = []
        #     for c in range(numCols):
        #         row.append([])
        #     trajGridReady.append(row)

        # totalWidth = numpy.max(trajs[:,:,0])+numpy.min(trajs[:,:,0])
        # totalHeight = numpy.max(trajs[:,:,1])+numpy.min(trajs[:,:,1])
        totalWidth=self.width
        totalHeight=self.height

        width = totalWidth / numRows
        height = totalHeight / numCols
        for r in range(numRows):
            xMin = r * width - width * marginPercentage
            xMax = (r + 1) * width + width * marginPercentage
            for c in range(numCols):
                yMin = c * height - height * marginPercentage
                yMax = (c + 1) * height + height * marginPercentage
                # width=width+width*marginPercentage
                # height = height + height * marginPercentage

                # initialX=-1
                # initialY=-1
                cutLat=-1
                cutLon=-1
                subTrajs=[]
                for i in range(len(trajs)):
                    hasStarted=False
                    hasFinished=True
                    isFromInit=True
                    for t in range(trajs[i].shape[0]):
                        # if r == 0 and c == 0 and i == 0 and t == 700:
                        #     print("!!!")
                        if trajs[i][t,0]<xMax and trajs[i][t,0]>xMin and trajs[i][t,1]<yMax and trajs[i][t,1]>yMin:
                            if hasStarted==False:
                                newTrajCut = []
                                hasStarted=True
                                hasFinished=False
                                newTrajCut.append([trajs[i][t, 0], trajs[i][t, 1]])
                                if t!=0:
                                    isFromInit=False
                                    cutLat = trajs[i][t, 0]
                                    cutLon = trajs[i][t, 1]
                                else:
                                    isFromInit = True
                                    cutLat = -1
                                    cutLon = -1
                            else:
                                newTrajCut.append([trajs[i][t, 0], trajs[i][t, 1]])
                        else:
                            if hasStarted==True:
                                subTrajs.append(SubTraj(trajs[i][0,0],trajs[i][0,1],cutLat,cutLon,newTrajCut,isFromInit,1))
                                newTrajCut = []
                                # initialX = -1
                                # initialY = -1
                                cutLat = -1
                                cutLon = -1
                                hasStarted=False
                                hasFinished=True
                    if hasFinished==False:
                        subTrajs.append(SubTraj(trajs[i][0, 0], trajs[i][0, 1], cutLat, cutLon, newTrajCut, isFromInit, 0))
                trajGrid[r][c]=subTrajs
        self.trajGrid=trajGrid


    def dist(self,x,y,xp,yp):
        return math.sqrt(math.pow(x-xp,2)+math.pow(y-yp,2))

    def upDownSample(self,targetLength, isVizualize=False):
        for r in range(len(self.trajGrid)):
            for c in range(len(self.trajGrid[r])):
                for i in reversed(range(len(self.trajGrid[r][c]))):
                    # if r==0 and c==0:
                    # #     d=numpy.array(self.trajGrid[r][c][i].points)
                    # #     plt.scatter(d[:, 0], d[:, 1])
                    # #     plt.plot(d[:, 0], d[:, 1])
                    # #     plt.show()
                    #     print("DEBUG!!!!")
                    #     if len(self.trajGrid[r][c][i].points) < 45 or len(self.trajGrid[r][c][i].points) > 80:
                    #         del self.trajGrid[r][c][i]
                    #         continue
                    if len(self.trajGrid[r][c][i].points)<3:
                        del self.trajGrid[r][c][i]
                        continue
                    if len(self.trajGrid[r][c][i].points) < targetLength:
                        for m in range(1000000):
                            # if r == 3 and c == 2:
                            #     d = numpy.array(self.trajGrid[r][c][i].points)
                            #     plt.scatter(d[:, 0], d[:, 1])
                            #     plt.plot(d[:, 0], d[:, 1])
                            #     plt.show()
                            #     print("DEBUG!!!!")
                            potentialPoints = []
                            potentialPointsDists = []
                            for t in range(1, len(self.trajGrid[r][c][i].points)):
                                potentialPoints.append(t)
                                dist = math.sqrt(math.pow(
                                    self.trajGrid[r][c][i].points[t - 1][0] - self.trajGrid[r][c][i].points[t][0],2) + math.pow(
                                    self.trajGrid[r][c][i].points[t - 1][1] - self.trajGrid[r][c][i].points[t][1],2))
                                potentialPointsDists.append(dist)
                            maxDist = -1000000
                            maxDistIndex = 'a'
                            for m in range(len(potentialPointsDists)):
                                if potentialPointsDists[m] > maxDist:
                                    maxDist = potentialPointsDists[m]
                                    maxDistIndex = m
                            if len(self.trajGrid[r][c][i].points)<3:
                                midPointX = (self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex] - 1][0] +
                                             self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex]][0]) / 2
                                midPointY = (self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex] - 1][1] +
                                             self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex]][1]) / 2
                            else:
                                midPointX = (self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex]-1][0] +
                                             self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex] + 0][0]) / 2
                                midPointY = (self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex]-1][1] +
                                             self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex] + 0][1]) / 2

                            cutLeft = self.trajGrid[r][c][i].points[0:potentialPoints[maxDistIndex]]
                            cutRight = self.trajGrid[r][c][i].points[potentialPoints[maxDistIndex]:len(self.trajGrid[r][c][i].points)]
                            newPoints = cutLeft + [[midPointX, midPointY]] + cutRight
                            self.trajGrid[r][c][i].points = newPoints
                            if len(self.trajGrid[r][c][i].points)==targetLength:
                                break

                    elif len(self.trajGrid[r][c][i].points) > targetLength:
                        if isVizualize==True:
                            plt.plot(numpy.array(self.trajGrid[r][c][i].points)[:,0],
                                 numpy.array(self.trajGrid[r][c][i].points)[:,1])
                            plt.scatter(numpy.array(self.trajGrid[r][c][i].points)[:,0],
                                 numpy.array(self.trajGrid[r][c][i].points)[:,1])
                            plt.show()
                        offset = 0
                        for tryIndex in range(1000000):
                            potentialPoints = []
                            potentialPointsDists = []
                            # if tryIndex==4:
                            #     print("!")
                            for t in range(1, len(self.trajGrid[r][c][i].points) - 1):
                                dXL = self.trajGrid[r][c][i].points[t - 1][0] - self.trajGrid[r][c][i].points[t][0]
                                dXR = self.trajGrid[r][c][i].points[t][0] - self.trajGrid[r][c][i].points[t + 1][0]

                                dYL = self.trajGrid[r][c][i].points[t - 1][1] - self.trajGrid[r][c][i].points[t][1]
                                dYR = self.trajGrid[r][c][i].points[t][1] - self.trajGrid[r][c][i].points[t + 1][1]
                                angle = math.atan2(dYL, dXL) * 180 / math.pi
                                angleP = math.atan2(dYR, dXR) * 180 / math.pi
                                if math.fabs(angle - angleP) < 3 + offset:
                                    potentialPoints.append(t)
                                    dist = math.sqrt(math.pow(
                                        self.trajGrid[r][c][i].points[t - 1][0] - self.trajGrid[r][c][i].points[t + 1][0],2) + math.pow(
                                        self.trajGrid[r][c][i].points[t - 1][1] - self.trajGrid[r][c][i].points[t + 1][1],2))
                                    potentialPointsDists.append(dist)
                            if len(potentialPoints) == 0:
                                # print('FAILED TO REMOVE POINT!!!')
                                # print('INCREASEING OFFSET!')
                                offset = offset + 1
                                if offset > 50:
                                    # plt.plot(self.trajGrid[r][c].streetPoints[0,:],self.trajGrid[r][c].streetPoints[1,:])
                                    points=numpy.array(self.trajGrid[r][c][i].points)
                                    plt.plot(points[:,0],points[:,1])
                                    plt.show()
                                    print("TOO LARGE ANGLE!!! ERROR THROWN")
                                    raise ValueError("CODE STOPPED FOR HAVING TOO LARGE ANGLE")
                                continue
                            minDist = 1000000
                            minDistIndex = 'a'
                            for m in range(len(potentialPointsDists)):
                                if potentialPointsDists[m] < minDist:
                                    minDist = potentialPointsDists[m]
                                    minDistIndex = m
                            del self.trajGrid[r][c][i].points[potentialPoints[minDistIndex]]
                            # plt.plot(numpy.array(self.trajGrid[r][c][i].points)[:,0],
                            #      numpy.array(self.trajGrid[r][c][i].points)[:,1])
                            # plt.scatter(numpy.array(self.trajGrid[r][c][i].points)[:,0],
                            #      numpy.array(self.trajGrid[r][c][i].points)[:,1])
                            # plt.title(str(len(self.trajGrid[r][c][i].points)))
                            # plt.show()
                            if len(self.trajGrid[r][c][i].points) == targetLength:
                                break
                        if isVizualize == True:
                            plt.plot(numpy.array(self.trajGrid[r][c][i].points)[:, 0],
                                 numpy.array(self.trajGrid[r][c][i].points)[:, 1])
                            plt.scatter(numpy.array(self.trajGrid[r][c][i].points)[:, 0],
                                    numpy.array(self.trajGrid[r][c][i].points)[:, 1])
                            plt.title(str(len(self.trajGrid[r][c][i].points)))
                            plt.show()

                        # print("!!!")

    def addExtraStreetPoints(self,inputStreetPoints):
        outputStreetPoints=copy.deepcopy(inputStreetPoints)

        for r in range(len(self.trajGrid)):
            for c in range(len(self.trajGrid[r])):
                for i in range(len(self.trajGrid[r][c])):
                    for t in range(len(self.trajGrid[r][c][i].points)):
                        isFound=False
                        for n in range(len(outputStreetPoints)):
                            if self.trajGrid[r][c][i].points[t][0]==outputStreetPoints[n][0] and self.trajGrid[r][c][i].points[t][1]==outputStreetPoints[n][1]:
                                isFound=True
                                break
                        if isFound==False:
                            outputStreetPoints.append([self.trajGrid[r][c][i].points[t][0],self.trajGrid[r][c][i].points[t][1]])

        totalWidth = self.width
        totalHeight = self.height
        width = totalWidth / self.numRows
        height = totalHeight / self.numCols
        for r in range(len(self.trajGrid)):
            xMin = r * width - width * self.marginPercentage
            xMax = (r + 1) * width + width * self.marginPercentage
            for c in range(len(self.trajGrid[r])):
                yMin = c * height - height * self.marginPercentage
                yMax = (c + 1) * height + height * self.marginPercentage
                localStreetPoints=[]
                for m in range(len(outputStreetPoints)):
                    if outputStreetPoints[m][0]<xMax and outputStreetPoints[m][0]>xMin and outputStreetPoints[m][1]<yMax and outputStreetPoints[m][1]>yMin:
                        localStreetPoints.append([outputStreetPoints[m][0],outputStreetPoints[m][1]])
                for i in range(len(self.trajGrid[r][c])):
                    self.trajGrid[r][c][i].streetPoints=localStreetPoints

        return outputStreetPoints

    def scaleData(self,value):
        for r in range(len(self.trajGrid)):
            for c in range(len(self.trajGrid[r])):
                for i in range(len(self.trajGrid[r][c])):
                    # plt.scatter(self.trajGrid[r][c][i].streetPoints[0, :],
                    #             self.trajGrid[r][c][i].streetPoints[1, :], s=40, c=[[0, 0, 1]])
                    # # for i in range(self.points.shape[0]):
                    # plt.plot(self.trajGrid[r][c][i].points[:, 0], self.trajGrid[r][c][i].points[:, 1])
                    # plt.scatter(self.trajGrid[r][c][i].points[:, 0], self.trajGrid[r][c][i].points[:, 1], s=20,
                    #             c=[[1, 1, 0]])
                    # # plt.scatter(self.trajGrid[r][c][i].initLat,
                    # #             self.trajGrid[r][c][i].initLon, s=100, c=[[0, 1, 0]])
                    # # if self.trajGrid[r][c][i].cutLat==-1:
                    # plt.scatter(self.trajGrid[r][c][i].cutLat,
                    #             self.trajGrid[r][c][i].cutLon, s=150, c=[[1, 0, 0]])
                    # plt.show()

                    self.trajGrid[r][c][i].points=self.trajGrid[r][c][i].points*value
                    self.trajGrid[r][c][i].streetPoints = self.trajGrid[r][c][i].streetPoints * value
                    # self.trajGrid[r][c][i].initLat = self.trajGrid[r][c][i].initLat * value
                    # self.trajGrid[r][c][i].initLon = self.trajGrid[r][c][i].initLon * value
                    self.trajGrid[r][c][i].cutLat = self.trajGrid[r][c][i].cutLat * value
                    self.trajGrid[r][c][i].cutLon = self.trajGrid[r][c][i].cutLon * value

                    # print(f"r: {r} c: {c} i: {i}")
                    # plt.scatter(self.trajGrid[r][c][i].streetPoints[0, :],
                    #             self.trajGrid[r][c][i].streetPoints[1, :], s=40, c=[[0,0,1]])
                    # # for i in range(self.points.shape[0]):
                    # plt.plot(self.trajGrid[r][c][i].points[:, 0], self.trajGrid[r][c][i].points[:, 1])
                    # plt.scatter(self.trajGrid[r][c][i].points[:, 0], self.trajGrid[r][c][i].points[:, 1], s=20, c=[[1,1,0]])
                    # # plt.scatter(self.trajGrid[r][c][i].initLat,
                    # #             self.trajGrid[r][c][i].initLon, s=100, c=[[0,1,0]])
                    # # if self.trajGrid[r][c][i].cutLat==-1:
                    # plt.scatter(self.trajGrid[r][c][i].cutLat,
                    #             self.trajGrid[r][c][i].cutLon, s=150, c=[[1,0,0]])
                    # plt.show()
                    #
                    # print("!!!")

    def normalize(self):
        for r in range(len(self.trajGrid)):
            for c in range(len(self.trajGrid[r])):
                for i in range(len(self.trajGrid[r][c])):
                    maxX = numpy.max(self.trajGrid[r][c][i].streetPoints[0, :])
                    minX = numpy.min(self.trajGrid[r][c][i].streetPoints[0, :])
                    maxY = numpy.max(self.trajGrid[r][c][i].streetPoints[1, :])
                    minY = numpy.min(self.trajGrid[r][c][i].streetPoints[1, :])
                    # plt.scatter(self.trajGrid[r][c][i].streetPoints[0, :],
                    #             self.trajGrid[r][c][i].streetPoints[1, :])
                    # plt.plot(self.trajGrid[r][c][i].points[:,0], self.trajGrid[r][c][i].points[:,1])
                    # plt.show()
                    # (adjustedStreetPoints[:, 0] - minX) / (maxX - minX)
                    if maxX == minX:
                        self.trajGrid[r][c][i].points[:, 0] = 0.5 + 1
                        self.trajGrid[r][c][i].streetPoints[0, :] = 0.5 + 1
                        self.trajGrid[r][c][i].cutLat = 0.5 + 1
                    else:
                        self.trajGrid[r][c][i].points[:, 0] = ((self.trajGrid[r][c][i].points[:, 0] - minX) / (
                                    maxX - minX)) + 1
                        self.trajGrid[r][c][i].streetPoints[0, :] = ((self.trajGrid[r][c][i].streetPoints[0,
                                                                     :] - minX) / (maxX - minX)) + 1
                        self.trajGrid[r][c][i].cutLat = ((self.trajGrid[r][c][i].cutLat - minX) / (maxX - minX)) + 1
                    if maxY == minY:
                        self.trajGrid[r][c][i].points[:, 1] = 0.5 + 1
                        self.trajGrid[r][c][i].streetPoints[1, :] = 0.5 + 1
                        self.trajGrid[r][c][i].cutLon = 0.5 + 1
                    else:
                        self.trajGrid[r][c][i].points[:, 1] = ((self.trajGrid[r][c][i].points[:, 1] - minY) / (
                                    maxY - minY)) + 1
                        self.trajGrid[r][c][i].streetPoints[1, :] = ((self.trajGrid[r][c][i].streetPoints[1,
                                                                     :] - minY) / (maxY - minY)) + 1
                        self.trajGrid[r][c][i].cutLon = ((self.trajGrid[r][c][i].cutLon - minY) / (maxY - minY)) + 1


                    # self.trajGrid[r][c][i].initLat = (self.trajGrid[r][c][i].initLat - minX) / (maxX - minX)
                    # self.trajGrid[r][c][i].initLon = (self.trajGrid[r][c][i].initLon - minY) / (maxY - minY)
                    # if self.trajGrid[r][c][i].cutLat == -1:

                    # print(f"r: {r} c: {c} i: {i}")
                    # plt.scatter(self.trajGrid[r][c][i].streetPoints[0, :],
                    #             self.trajGrid[r][c][i].streetPoints[1, :], s=40, c=[[0,0,1]])
                    # # for i in range(self.points.shape[0]):
                    # plt.plot(self.trajGrid[r][c][i].points[:, 0], self.trajGrid[r][c][i].points[:, 1])
                    # plt.scatter(self.trajGrid[r][c][i].points[:, 0], self.trajGrid[r][c][i].points[:, 1], s=20, c=[[1,1,0]])
                    # plt.scatter(self.trajGrid[r][c][i].initLat,
                    #             self.trajGrid[r][c][i].initLon, s=100, c=[[0,1,0]])
                    # # if self.trajGrid[r][c][i].cutLat==-1:
                    # plt.scatter(self.trajGrid[r][c][i].cutLat,
                    #             self.trajGrid[r][c][i].cutLon, s=150, c=[[1,0,0]])
                    # plt.show()
                    #
                    # print("!!!")


    @staticmethod
    def gen2DMapFromStreetPoints(points, nGrid):
        gridData = numpy.zeros((nGrid, nGrid))
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                gridX = numpy.floor(points[i, j, 0] * nGrid)
                gridY = numpy.floor(points[i, j, 1] * nGrid)
                if gridX > nGrid - 1:
                    gridX = nGrid - 1
                if gridX < 0:
                    gridX = 0
                if gridY > nGrid - 1:
                    gridY = nGrid - 1
                if gridY < 0:
                    gridY = 0
                gridData[int(gridX), int(gridY)] = gridData[int(gridX), int(gridY)] + 1
        return gridData