# Version 5 with initial condition with 4 inputs
import numpy.random
from sympy.abc import alpha
from torch.optim.lr_scheduler import ExponentialLR

import DataGeneratorFcn
from BatchCumulationCalc import batchCumulationCalc
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import torch
from Tiles import TilesSynthetic
from TrajCell_V20DEBUG import TrajCells
import time

from LossVisualizer import LossVisualizer

# matplotlib.use('TkAgg')
matplotlib.use('QtAgg')
print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(enabled=True)

modelVersion = "V19_10_light_CircularV3FAST_LOSS_FLEXMULTIBRANCH"
isTrainModel = True
continueTrain = True
isChangeWeights = True
isRunOnCPU = False
isCompleteRandomCities = False
fixedOffset = 0.9999
weightingStepSize = 0.3
beforeStepsFixed = 0
numOptIterates = 0
extraStepsFixedBefore = 0
extraStepsFixed = 50
initialLRBefore = 0.00015
initialLR = 0.00002
totalBatchSize = 60
BATCH_SIZE = 50
repeatTimes = 5
metaRepeat = 2
cumulativeIters = batchCumulationCalc(totalBatchSize, BATCH_SIZE)
scale = 10
timesteps = 24
varValScale=1.0
isVisualizeLoss = False
isFreeze=True

testFolder = 'synthetic1_V22Stitch_DEBUGV20'

numTrajectories = 80
maxTrajectoryLength = 30
nRow = 5
nCols = 5

seed = 0
tilesSynthetic = TilesSynthetic("datasets/SyntheticTiles/full.png", nRow, nCols)
allPaths, streetPoints, initLats, initLons, cutLats, cutLons = tilesSynthetic.makeSyntheticPaths(numTrajectories, nRow,
                                                                                                 nCols, 0.05,
                                                                                                 maxTrajectoryLength,
                                                                                                 seed,
                                                                                                 isVisualize=False)
tilesSynthetic.normalize()
tilesSynthetic.scaleData(scale)
# print("!!!")

# meanVal=np.mean(streetPoints,axis=1)
# varVal=np.std(streetPoints,axis=1)*1.0

allCells = TrajCells(nRow, nCols, tilesSynthetic)

allCells.prepareTorchData(isRunOnCPU)
allCells.shuffleData()# MAIN***

timeTemp = datetime.now()#MAIN
np.random.seed(timeTemp.minute + timeTemp.hour + timeTemp.microsecond)#MAIN
# indices=np.random.permutation(len(allPaths))

isTrainedFirstModelDebug = False
if isTrainModel == True:
    for r in range(len(allCells.cells)):
        for c in range(len(allCells.cells[r])):
            if len(allCells.tilesData.trajGrid[r][c]) != 0:
                start_time = time.time()
                if continueTrain == True:
                    print("row: " + str(r) + " col: " + str(c) + " TRY LOADING!")
                    allCells.loadCellModel(r, c, testFolder, modelVersion, initialLR, isRunOnCPU, batch_size=BATCH_SIZE)
                    # if c == 0 and r == 0:
                    # allCells.visSaveNetwork(r, c, testFolder, modelVersion, BATCH_SIZE, maxTrajectoryLength, isRunOnCPU)
                else:
                    allCells.initCellModel(r, c, initialLR, isRunOnCPU, batch_size=BATCH_SIZE)
                # allCells.cells[r][c].model.avgMaxDist = allCells.cells[r][c].maxAvgDist
                # allCells.cells[r][c].model.avgMinDist = allCells.cells[r][c].minAvgDist
                # allCells.cells[r][c].showDatas()

                allCells.visSaveNetwork(r,c,testFolder,modelVersion,BATCH_SIZE,maxTrajectoryLength,isRunOnCPU)
                avgQ = allCells.cells[r][c].assessTrain(modelVersion, 60, maxTrajectoryLength, timesteps, isRunOnCPU)
                # avgQ = 1.0# DEBUG
                print(f"Average quality: {avgQ}")
                averageQuality=avgQ
                for m in range(metaRepeat):
                     if averageQuality>10.0:
                         break
                     # allCells.cells[r][c].train(BATCH_SIZE, meanVal, varVal, testFolder, modelVersion, maxTrajectoryLength,
                     #                            timesteps, repeatTimes, fixedOffset, weightingStepSize, numOptIterates,
                     #                            extraStepsFixed, beforeStepsFixed, cumulativeIters, initialLR,
                     #                            isVisualizeLoss, isRunOnCPU, isChangeWeights,varValScale=varValScale)

                     averageQuality=allCells.cells[r][c].trainV2(BATCH_SIZE, testFolder, modelVersion, maxTrajectoryLength,
                        timesteps, repeatTimes, fixedOffset, weightingStepSize, numOptIterates, extraStepsFixed,
                        extraStepsFixedBefore, beforeStepsFixed, cumulativeIters, initialLR, initialLRBefore,
                        isVisualizeLoss, isRunOnCPU, isChangeWeights,isFreeze=isFreeze,varValScale=varValScale)
                    # allCells.cells[r][c].train_simple(BATCH_SIZE, testFolder, modelVersion, maxTrajectoryLength,
                    #                            timesteps, repeatTimes, fixedOffset, numOptIterates,
                    #                            extraStepsFixed, beforeStepsFixed, cumulativeIters, initialLR,
                    #                            isVisualizeLoss, isRunOnCPU, isChangeWeights,varValScale=varValScale)
                    # initialLR=initialLR*0.995
                print(f"Average quality: {averageQuality}")
                end_time = time.time()
                print("row: " + str(r) + " col: " + str(c) + " FINISHED! RUNTIME: "+str(end_time-start_time))
                allCells.unloadCellModel(r, c)
                isTrainedFirstModelDebug = True
                plt.clf()
                # break  # DEBUGGING
            if isTrainedFirstModelDebug == True:
                break
            # break  # DEBUGGING
        if isTrainedFirstModelDebug == True:
            break

numPredicts = 20
predPerCell = []
time = datetime.now()  # MAIN
numpy.random.seed(time.minute + time.hour + time.microsecond)  # MAIN
# numpy.random.seed(0)# DEBUG
for r in range(nRow):
    row = []
    for c in range(nCols):
        row.append(0)
    predPerCell.append(row)
for tr in range(numPredicts):
    tryCounter = 0
    isFound = False
    while isFound == False or tryCounter > 1000:
        selectedInitRowCell = numpy.random.randint(0, high=nRow)
        selectedInitColCell = numpy.random.randint(0, high=nCols)
        if len(allCells.tilesData.trajGrid[selectedInitRowCell][selectedInitColCell]) != 0:
            predPerCell[selectedInitRowCell][selectedInitColCell] = (
                    predPerCell[selectedInitRowCell][selectedInitColCell] + 1)
            isFound = True
        tryCounter = tryCounter + 1

# \/\/\/DEBUGGING
predPerCell[0][0] = numPredicts
# ^^^DEBUGGING

nGrids = 200
for r in range(nRow):
    for c in range(nCols):
        if predPerCell[r][c] > 0:
            allCells.loadCellModel(r, c, testFolder, modelVersion, initialLR, isRunOnCPU, batch_size=numPredicts)
            # print(allCells.cells[r][c].model.LHS_diff.x0)
            # print(allCells.cells[r][c].model.LHS_L1.x0)
            # print(allCells.cells[r][c].model.LHS_L2.x0)
            # print(allCells.cells[r][c].model.LHS_L3.x0)
            # allCells.predict(r,c,predPerCell[r][c],maxTrajectoryLength,meanVal,varVal,nGrids,scale,
            #                              isRunOnCPU,modelVersion,timesteps)

            meanVal=np.mean(allCells.cells[r][c].streetPoints.cpu().numpy(), axis=1)
            varVal=np.std(allCells.cells[r][c].streetPoints.cpu().numpy(), axis=1) * varValScale
            print(meanVal)
            print(varVal)
            # pred, streetPoints2D = allCells.predict(r, c, predPerCell[r][c], maxTrajectoryLength, meanVal,varVal, nGrids, scale,
            #                  isRunOnCPU, modelVersion, (int)(timesteps*1.0))

            # r,c, modelVersion, numPredicts, maxTrajectoryLength, timesteps, nGrid, scale, meanVal, varVal, isRunOnCPU
            pred, streetPoints2D = allCells.predictDEBUG(r, c, modelVersion, numPredicts, maxTrajectoryLength,
                                                         (int)(timesteps * 1.0), nGrids, scale, meanVal, varVal,
                                                         isRunOnCPU)
            TrajCells.visualize_extent(streetPoints2D, predPerCell[r][c], pred, allCells.cells[r][c].streetPoints,
                                       maxTrajectoryLength)
            allCells.unloadCellModel(r, c)

print("!!!")
