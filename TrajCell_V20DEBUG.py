import numpy
from fontTools.unicodedata import block

import Tiles
import V19_10 as ModelClass
import torch.optim as optim
from pathlib import Path
import torch
from torchview import draw_graph
import gc
import math
from datetime import datetime
from QualityMeasure import JSD, JSD_SingleB
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm.auto import trange
from torch.optim.lr_scheduler import ExponentialLR
from LossVisualizer import LossVisualizer
import matplotlib.cm as cm
import inspect

class TrajCell:
    def __init__(self,r,c):
        self.row=r
        self.column=c
        self.model=None
        self.optimizer=None

        self.points = None
        self.initLats = None
        self.initLons = None
        self.cutLats = None
        self.cutLons = None
        self.streetPoints = None

        self.maxAvgDist = 1
        self.minAvgDist = 0
        self.isContinues = None


    def setTrajs(self,input):
        self.trajs=input

    def train_simple(self, BATCH_SIZE, testFolder, modelVersion, maxTrajectoryLength, timesteps, repeatTimes,
              fixedOffset, numIterates, extraIterates, beforeStepsFixed, cumulativeIters,
              initialLR, isVisualizeLoss, isRunOnCPU, isChangeWeights, varValScale=3.0):
        # Loss and optimizer
        if "LOSS" in modelVersion:
            criterion = ModelClass.CustomLoss(self.streetPoints, self.streetPoints.shape[1],
                                                     maxTrajectoryLength, timesteps)
            # criterion.requires_grad_(True)
        else:
            criterion = nn.L1Loss()
        isAdvancedWeighting=False
        isAdvancedExponent=True
        meanVal=np.mean(self.streetPoints.cpu().numpy(),axis=1)
        varVal=np.std(self.streetPoints.cpu().numpy(),axis=1)*varValScale
        # print(meanVal)
        # print(varVal)

        allLossValues = []
        diff=(1-fixedOffset)
        stepSize=diff/repeatTimes
        for r in range(repeatTimes):
            # print(f"repeatTimes: {r}")
            offsetValue = fixedOffset+r*stepSize
            # print(offsetValue)
            lossValuesRes = self.trainInterior(BATCH_SIZE, meanVal, varVal, testFolder, modelVersion, criterion, numIterates, extraIterates, beforeStepsFixed, cumulativeIters, stepSize, offsetValue, initialLR, isVisualizeLoss,isChangeWeights,isAdvancedWeighting,isAdvancedExponent,isRunOnCPU,timesteps=timesteps)
            allLossValues.append(lossValuesRes)
            initialLR = initialLR * 0.901
        resultAllLoss = np.concatenate(allLossValues)
        plt.plot(resultAllLoss)
        plt.title("All loss values during training")
        plt.savefig(modelVersion + "_LossValues.png")
        plt.show()

    def train(self, BATCH_SIZE, testFolder, modelVersion, maxTrajectoryLength, timesteps, repeatTimes,
              fixedOffset, weightingStepSize, numIterates, extraIterates, beforeStepsFixed, cumulativeIters,
              initialLR, isVisualizeLoss, isRunOnCPU, isChangeWeights, varValScale=3.0):
        if hasattr(self, 'defaultGate')==True:
            self.defaultGate = torch.ones(BATCH_SIZE, device=torch.device('cuda')).unsqueeze(dim=1).unsqueeze(dim=2)
            if isRunOnCPU == False:
                self.defaultGate = self.defaultGate.cuda()

        # Loss and optimizer
        if "LOSS" in modelVersion:
            criterion = ModelClass.CustomLoss(self.streetPoints, self.streetPoints.shape[1],
                                                     maxTrajectoryLength, timesteps)
            # criterion.requires_grad_(True)
        else:
            criterion = nn.L1Loss()
        isAdvancedWeighting=False
        isAdvancedExponent=True
        meanVal=np.mean(self.streetPoints.cpu().numpy(),axis=1)
        varVal=np.std(self.streetPoints.cpu().numpy(),axis=1)*varValScale
        # print(meanVal)
        # print(varVal)

        allLossValues = []
        maxOffsetMovement = 0.0
        for r in range(repeatTimes):
            # print(f"repeatTimes: {r}")
            offsetValues = np.arange(np.minimum(fixedOffset + (r / repeatTimes) * maxOffsetMovement, 0.9999), 1,
                                     weightingStepSize)
            # print(offsetValues)
            for s in range(offsetValues.shape[0]):
                stepSize = np.minimum(weightingStepSize, 1 - offsetValues[s])
                # print(stepSize)
                lossValuesRes = self.trainInterior(BATCH_SIZE, meanVal, varVal, testFolder, modelVersion, criterion, numIterates, extraIterates, beforeStepsFixed, cumulativeIters, stepSize, offsetValues[s], initialLR, isVisualizeLoss,isChangeWeights,isAdvancedWeighting,isAdvancedExponent,isRunOnCPU,timesteps=timesteps)
                allLossValues.append(lossValuesRes)
            initialLR = initialLR * 0.901
            # print(f"LHS_L1: {self.model.LHS_L1.x0.data.detach().cpu().numpy()}")
            # print(f"LHS_L2: {self.model.LHS_L2.x0.data.detach().cpu().numpy()}")
            # # print(f"LHS_L3: {self.model.LHS_L3.x0.data.detach().cpu().numpy()}")
            # print(f"LHS_diff: {self.model.LHS_diff.x0.data.detach().cpu().numpy()}")
        resultAllLoss = np.concatenate(allLossValues)
        plt.plot(resultAllLoss)
        plt.title("All loss values during training")
        plt.savefig(modelVersion + "_LossValues.png")
        # plt.show()
        plt.cla()
        avgQ=self.assessTrain(60,maxTrajectoryLength,timesteps,isRunOnCPU,varValScale=varValScale)
        return avgQ

    def trainV2(self, BATCH_SIZE, testFolder, modelVersion, maxTrajectoryLength, timesteps, repeatTimes,
              fixedOffset, weightingStepSize, numIterates, extraIterates, extraIteratesBefore, beforeStepsFixed,
              cumulativeIters, initialLR, initialLRBefore, isVisualizeLoss, isRunOnCPU, isChangeWeights, isFreeze=False,
              varValScale=3.0):
        if hasattr(self, 'defaultGate')==True:
            self.defaultGate = torch.ones(BATCH_SIZE, device=torch.device('cuda')).unsqueeze(dim=1).unsqueeze(dim=2)
            if isRunOnCPU == False:
                self.defaultGate = self.defaultGate.cuda()

        # Loss and optimizer
        if "LOSS" in modelVersion:
            criterion = ModelClass.CustomLoss(self.streetPoints, self.streetPoints.shape[1],
                                                     maxTrajectoryLength, timesteps)
            # criterion.requires_grad_(True)
        else:
            criterion = nn.L1Loss()
        isAdvancedWeighting=True
        isAdvancedExponent=True
        meanVal=np.mean(self.streetPoints.cpu().numpy(),axis=1)
        varVal=np.std(self.streetPoints.cpu().numpy(),axis=1)*varValScale
        # print(meanVal)
        # print(varVal)

        allLossValues = []
        offsetValues = np.arange(np.minimum(fixedOffset, 0.9999), 1, weightingStepSize)
        # for r in range(repeatTimes):
            # print(f"repeatTimes: {r}")

            # print(offsetValues)

        lossValuesRes = self.trainInterior(BATCH_SIZE, meanVal, varVal, testFolder, modelVersion,
                                           criterion, numIterates, extraIteratesBefore, beforeStepsFixed, cumulativeIters,
                                           0, 0, initialLRBefore, isVisualizeLoss, isChangeWeights,
                                           isAdvancedWeighting, isAdvancedExponent, isRunOnCPU, timesteps=timesteps,isFreeze=isFreeze)

        allLossValues.append(lossValuesRes)
        avgQ = self.assessTrain(modelVersion, 60, maxTrajectoryLength, timesteps, isRunOnCPU, varValScale=varValScale)
        print(f"Average quality internal: {avgQ}")
        for s in range(offsetValues.shape[0]):
            for r in range(repeatTimes):
                stepSize = np.minimum(weightingStepSize, 1 - offsetValues[s])
                    # print(stepSize)
                lossValuesRes = self.trainInterior(BATCH_SIZE, meanVal, varVal, testFolder, modelVersion,
                                    criterion, numIterates, extraIterates, beforeStepsFixed, cumulativeIters,
                                    stepSize, offsetValues[s], initialLR, isVisualizeLoss,isChangeWeights,
                                    isAdvancedWeighting,isAdvancedExponent,isRunOnCPU,timesteps=timesteps,isFreeze=isFreeze)
                allLossValues.append(lossValuesRes)
                avgQ = self.assessTrain(modelVersion, 60, maxTrajectoryLength, timesteps, isRunOnCPU, varValScale=varValScale)
                print(f"Average quality internal: {avgQ}")
                initialLR = initialLR * 0.98
                plt.clf()
            # print(f"LHS_L1: {self.model.LHS_L1.x0.data.detach().cpu().numpy()}")
            # print(f"LHS_L2: {self.model.LHS_L2.x0.data.detach().cpu().numpy()}")
            # # print(f"LHS_L3: {self.model.LHS_L3.x0.data.detach().cpu().numpy()}")
            # print(f"LHS_diff: {self.model.LHS_diff.x0.data.detach().cpu().numpy()}")
        resultAllLoss = np.concatenate(allLossValues)
        plt.plot(resultAllLoss)
        plt.title("All loss values during training")
        plt.savefig(modelVersion + "_LossValues.png")
        # plt.show()
        plt.cla()
        avgQ=self.assessTrain(modelVersion,60,maxTrajectoryLength,timesteps,isRunOnCPU,varValScale=varValScale)
        return avgQ


    def assessTrain(self,modelVersion,numTraj,trajectoryLength,timesteps,isRunOnCPU,varValScale=3.0):
        meanVal=np.mean(self.streetPoints.cpu().numpy(), axis=1)
        varVal=np.std(self.streetPoints.cpu().numpy(), axis=1) * varValScale
        if hasattr(self.model, 'defaultGate') == True:
            self.model.defaultGate = torch.ones(numTraj, device=torch.device('cuda')).unsqueeze(dim=1).unsqueeze(
                dim=2)
            if isRunOnCPU == False:
                self.model.defaultGate = self.model.defaultGate.cuda()
        x_gauss = np.random.normal(loc=meanVal, scale=varVal, size=(numTraj, trajectoryLength, 2))

        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(numTraj, 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(numTraj, trajectoryLength - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVal / 12.0, size=(numTraj, trajectoryLength, 2)) + varVal * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVal / 12.0, size=(numTraj, trajectoryLength, 2)) + meanVal
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVal[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVal[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVal[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVal[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        x = np.concat((allXs, allYs), axis=2)

        x = (x+x_gauss)/2

        x = torch.from_numpy(x).to(torch.float32)
        # self.cells[r][c].model.psl.lenForbidden = serializedSelected.shape[1]
        # self.cells[r][c].model.pslSum.lenForbidden = serializedSelected.shape[1]
        # # if isRunOnCPU == False:
        # #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
        # # else:
        # #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
        if "FULLBRANCH" in modelVersion:
            self.model.branch1.psl.B = self.streetPoints
            self.model.branch1.pslSum.B = self.streetPoints

            self.model.branch2.psl.B = self.streetPoints
            self.model.branch2.pslSum.B = self.streetPoints

            self.model.branch3.psl.B = self.streetPoints
            self.model.branch3.pslSum.B = self.streetPoints
        elif "FLEXMULTIBRANCH" in modelVersion:
            for m in range(self.model.numBranches):
                self.model.branches[m].psl.B = self.streetPoints
                self.model.branches[m].pslSum.B = self.streetPoints
        else:
            self.model.psl.B = self.streetPoints
            self.model.pslSum.B = self.streetPoints

        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, timesteps)

        # fig, axs = plt.subplots(nrows=1, ncols=(int)(timesteps/1))
        with torch.no_grad():
            # indices = np.random.choice(serializedSelected.shape[1], numTraj, replace=True)
            # samples = serializedSelected[:,indices]
            # samples=torch.transpose(samples,0,1)
            # cond1 = torch.unsqueeze(samples[:, 0], dim=1)
            # cond2 = torch.unsqueeze(samples[:, 1], dim=1)

            indices = np.random.choice(self.cutLats.shape[0], numTraj, replace=True)
            cond1 = np.zeros((numTraj, 1))
            cond2 = np.zeros((numTraj, 1))
            for i in range(numTraj):
                cond1[i, 0] = self.cutLats[indices[i]]
                cond2[i, 0] = self.cutLons[indices[i]]
            cond1 = torch.from_numpy(cond1).to(torch.float32)
            cond2 = torch.from_numpy(cond2).to(torch.float32)

            x = TrajCells.predictLoop(self.model, x, numTraj, timesteps, cond1, cond2, isRunOnCPU, cmap, trajectoryLength)

        # plt.savefig(modelVersion + "_Results_gradient.png")
        # # plt.show()  # MAIN

        qualities = TrajCells.getTrajQualities(modelVersion, self.model, x).cpu()
        if hasattr(self.model, 'defaultGate') == True:
            self.model.defaultGate = torch.ones(self.model.batchSize, device=torch.device('cuda')).unsqueeze(dim=1).unsqueeze(
                dim=2)
            if isRunOnCPU == False:
                self.model.defaultGate = self.model.defaultGate.cuda()
        return qualities.mean()

    def trainInterior(self, BATCH_SIZE, meanVal, varVal, testFolder, modelVersion, cri,
                      numIterates, extraIterates, beforeStepsFixed, cumulativeIters,
                      maxWeightingStep, offsetWeighting, initialLR, isVisualizeLoss,
                      isChangeWeights, isAdvancedWeighting,isAdvancedExponent, isRunOnCPU, timesteps=16, isFreeze=False):
        allLoss = np.zeros(numIterates + extraIterates + beforeStepsFixed)
        bar = trange(numIterates + extraIterates + beforeStepsFixed)
        total = cumulativeIters
        self.optimizer.param_groups[0]['lr'] = initialLR
        scheduler1 = ExponentialLR(self.optimizer, gamma=0.99271)
        if isVisualizeLoss == True:
            lv = LossVisualizer(numIterates + extraIterates + beforeStepsFixed)
        optimResetCounter = 0
        usingLR = initialLR
        time = datetime.now()  # MAIN
        numpy.random.seed(time.minute + time.hour + time.microsecond)  # MAIN

        if "FULLBRANCH" in modelVersion:
            self.model.branch1.psl.lenForbidden = self.streetPoints.shape[1]
            self.model.branch1.psl.B = self.streetPoints
            self.model.branch1.pslSum.lenForbidden = self.streetPoints.shape[1]
            self.model.branch1.pslSum.B = self.streetPoints

            self.model.branch2.psl.lenForbidden = self.streetPoints.shape[1]
            self.model.branch2.psl.B = self.streetPoints
            self.model.branch2.pslSum.lenForbidden = self.streetPoints.shape[1]
            self.model.branch2.pslSum.B = self.streetPoints

            self.model.branch3.psl.lenForbidden = self.streetPoints.shape[1]
            self.model.branch3.psl.B = self.streetPoints
            self.model.branch3.pslSum.lenForbidden = self.streetPoints.shape[1]
            self.model.branch3.pslSum.B = self.streetPoints
        elif "FLEXMULTIBRANCH" in modelVersion:
            for m in range(self.model.numBranches):
                self.model.branches[m].psl.lenForbidden = self.streetPoints.shape[1]
                self.model.branches[m].psl.B = self.streetPoints
                self.model.branches[m].pslSum.lenForbidden = self.streetPoints.shape[1]
                self.model.branches[m].pslSum.B = self.streetPoints
        else:
            self.model.psl.lenForbidden = self.streetPoints.shape[1]
            self.model.psl.B = self.streetPoints
            self.model.pslSum.lenForbidden = self.streetPoints.shape[1]
            self.model.pslSum.B = self.streetPoints
        cri.forbiddens = self.streetPoints
        cri.lenForbidden = self.streetPoints.shape[1]
        for i in bar:
            if i >= beforeStepsFixed + numIterates:
                adjustedScheduleValue = offsetWeighting + maxWeightingStep
            elif i <= beforeStepsFixed:
                # adjustedScheduleValue=0
                adjustedScheduleValue = offsetWeighting
            elif i >= beforeStepsFixed:
                adjustedScheduleValue = math.pow(
                    offsetWeighting + ((i - beforeStepsFixed) / numIterates) * maxWeightingStep, 1.0)
            # print("adjustedScheduleValue:")
            # print(adjustedScheduleValue)
            if optimResetCounter > ((numIterates + 1) * 1000.8):
                usingLR = usingLR * 0.94
                self.optimizer = optim.Adam(self.model.parameters(), lr=usingLR)
                scheduler1 = ExponentialLR(self.optimizer, gamma=0.99271)
                optimResetCounter = 0
                print("OPTIMIZER RESET")
            optimResetCounter = optimResetCounter + 1
            epoch_loss = 0.0
            if isFreeze==False:
                self.optimizer.zero_grad()
                for j in range(total):
                    # # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    # batchIndices = np.random.randint(len(X_train), size=BATCH_SIZE)
                    # x_img = X_train[batchIndices]
                    self.shuffleLocalData()  # MAIN ***
                    batchIndices = np.random.randint(len(self.points), size=BATCH_SIZE)  # MAIN***
                    # batchIndices = np.zeros(BATCH_SIZE)  # DEBUG
                    x_img = self.points[batchIndices]
                    loss = self.train_one(BATCH_SIZE, modelVersion, meanVal, varVal, x_img, cri, adjustedScheduleValue, isRunOnCPU,
                                          timesteps=timesteps, isChangeWeights=isChangeWeights,
                                          isAdvancedWeighting=isAdvancedWeighting,
                                          isAdvancedExponent=isAdvancedExponent, isFreeze=isFreeze, iterIndex=i)
                    # optimizer_input.step()
                    epoch_loss += loss.item() * 1

                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100, norm_type='inf')
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, norm_type=2)
                self.optimizer.step()
            else:
                x_img = None
                epoch_loss = self.train_one(BATCH_SIZE, modelVersion, meanVal, varVal, x_img, cri, adjustedScheduleValue, isRunOnCPU,
                                      timesteps=timesteps, isChangeWeights=isChangeWeights,
                                      isAdvancedWeighting=isAdvancedWeighting,
                                      isAdvancedExponent=isAdvancedExponent, isFreeze=isFreeze, total=total, iterIndex=i)
                # print("!!!")
            if i % 2 == 0:
                bar.set_description(
                    f'loss: {epoch_loss:.5f}, lr: {scheduler1.get_last_lr()[0]:.9f}, ASV: {adjustedScheduleValue}')
            scheduler1.step()
            allLoss[i] = epoch_loss
            if isVisualizeLoss == True:
                lv.values[i] = epoch_loss
        Path('genStored/' + testFolder).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), 'genStored/' + testFolder + '/model_' + modelVersion + '_' + str(self.row) + '_' + str(self.column) + '.pytorch')
        torch.save(self.optimizer.state_dict(), 'genStored/' + testFolder + '/optimizer_' + modelVersion + '_' + str(self.row) + '_' + str(self.column) + '.pytorch')
        print("MODEL SAVED!")
        if isVisualizeLoss == True:
            lv.visualize(saveFileName='Loss_pytorch_' + modelVersion + '.png',
                         titleText="Pytorch loss value over iterations. Model " + modelVersion + ".")
        return allLoss

    def train_one(self, BATCH_SIZE, modelVersion, meanVal, varVal, x_img, cr, learningScheduleTime, isRunOnCPU,
                  timesteps=16, isChangeWeights=True, isAdvancedWeighting=True, isAdvancedExponent=False,
                  isFreeze=False, total=1, iterIndex=-1):
        if isFreeze==False:
            x_ts = ModelClass.generate_ts(timesteps, len(x_img), learningScheduleTime, isChangeWeights,
                                             isAdvancedWeighting=isAdvancedWeighting)
            x_a, x_b = ModelClass.forward_noise_notNormalized(meanVal, varVal, timesteps, x_img, x_ts,
                                                              learningScheduleTime, isChangeWeights,
                                                              isVizualize=False,
                                                              isAdvancedWeighting=isAdvancedExponent)

            # x_a, x_b = ModelClass.forward_noise_circular(self.maxAvgDist, self.minAvgDist, meanVal, varVal, timesteps, x_img, x_ts,
            #                                                   learningScheduleTime, isChangeWeights,
            #                                                   isVizualize=True,
            #                                                   isAdvancedWeighting=isAdvancedExponent)

            if x_a.__class__.__name__ == "Tensor":
                x_a = x_a.to(torch.float32)
                x_b = x_b.to(torch.float32)
            elif x_a.__class__.__name__ == "ndarray":
                x_a = torch.from_numpy(x_a).to(torch.float32)
                x_b = torch.from_numpy(x_b).to(torch.float32)
            x_ts = torch.from_numpy(x_ts / timesteps).to(torch.float32)

            x_ts = x_ts.unsqueeze(1)
            cond1 = torch.unsqueeze(x_img[:, 0, 0], dim=1).to(torch.float32)
            cond2 = torch.unsqueeze(x_img[:, 0, 1], dim=1).to(torch.float32)
            # opt.zero_grad()
            if isRunOnCPU == False:
                outputTrajs = self.model(x_a.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
                if "LOSS" in modelVersion:
                    loss = cr(outputTrajs, x_b.cuda(), x_ts.cuda())
                else:
                    loss = cr(outputTrajs, x_b.cuda())
            else:
                outputTrajs = self.model(x_a, x_ts, cond1, cond2)
                if "LOSS" in modelVersion:
                    loss = cr(outputTrajs, x_b, x_ts)
                else:
                    loss = cr(outputTrajs, x_b)
            # loss.requires_grad = True
            loss.backward()
            # opt.step()
            # visualiza(x_a.shape[0], x_a)
            # visualiza(x_b.shape[0], x_b)
            return loss
        else:
            if "FLEXMULTIBRANCH" in modelVersion:
                lossVal = 0
                cmap_name = 'jet'  # Example: Use the 'jet' colormap
                cmap = cm.get_cmap(cmap_name, self.model.numBranches)
                branchLosses=[]
                if isRunOnCPU == False:
                    allGenTrajs = torch.zeros((0, self.points.shape[1], self.points.shape[2])).cuda()
                    allConds1 = torch.zeros((0, self.points.shape[1], self.points.shape[2])).cuda()
                    allConds2 = torch.zeros((0, self.points.shape[1], self.points.shape[2])).cuda()
                    allBs = torch.zeros((0, self.points.shape[1], self.points.shape[2])).cuda()
                else:
                    allGenTrajs = torch.zeros((0, self.points.shape[1], self.points.shape[2]))
                    allConds1 = torch.zeros((0,))
                    allConds2 = torch.zeros((0,))
                    allBs = torch.zeros((0,))
                for m in range(self.model.numBranches):
                    if ModelClass.fullVisGenAB == True:
                        ModelClass.oneTimeVisGenAB = True
                    self.optimizer.zero_grad()
                    branchLoss = 0
                    if isRunOnCPU == False:
                        buildingGenTrajs = torch.zeros((0, self.points.shape[1], self.points.shape[2])).cuda()
                        buildingConds1 = torch.zeros((0,1)).cuda()
                        buildingConds2 = torch.zeros((0,1)).cuda()
                        buildingBs = torch.zeros((0, self.points.shape[1], self.points.shape[2])).cuda()
                    else:
                        buildingGenTrajs = torch.zeros((0, self.points.shape[1], self.points.shape[2]))
                        buildingConds1 = torch.zeros((0,1))
                        buildingConds2 = torch.zeros((0,1))
                        buildingBs = torch.zeros((0, self.points.shape[1], self.points.shape[2]))

                    for j in range(total):
                        self.shuffleLocalData()  # MAIN ***
                        batchIndices = np.random.randint(len(self.points), size=BATCH_SIZE)  # MAIN***
                        # batchIndices = np.zeros(BATCH_SIZE)  # DEBUG
                        x_img = self.points[batchIndices]
                        if m < 6:
                            loss, oTrajs, oCond1, oCond2, relatedBs = self.train_one_interval_interior(modelVersion, x_img,
                                                                            self.model.branchRanges[m][0],
                                                                            self.model.branchRanges[m][1],
                                                                            meanVal, varVal, cr,
                                                                            learningScheduleTime,
                                                                            isRunOnCPU,
                                                                            timesteps=timesteps,
                                                                            isNoiseSeparate=False,
                                                                            isAdvancedExponent=isAdvancedExponent,
                                                                            trainBranch=m,
                                                                            genTrajs=allGenTrajs,allConds1=allConds1,
                                                                            allConds2=allConds2,addedBCandidates=allBs)
                        else:
                            loss, oTrajs, oCond1, oCond2, relatedBs = self.train_one_interval_interior(modelVersion, x_img,
                                                                            self.model.branchRanges[m][0],
                                                                            self.model.branchRanges[m][1],
                                                                            meanVal, varVal, cr,
                                                                            learningScheduleTime,
                                                                            isRunOnCPU,
                                                                            timesteps=timesteps,
                                                                            isNoiseSeparate=False,
                                                                            isAdvancedExponent=isAdvancedExponent,
                                                                            trainBranch=m,
                                                                            genTrajs=allGenTrajs,allConds1=allConds1,
                                                                            allConds2=allConds2,addedBCandidates=allBs)
                        buildingGenTrajs = torch.concat((oTrajs, buildingGenTrajs), dim=0)
                        buildingConds1 = torch.concat((oCond1, buildingConds1), dim=0)
                        buildingConds2 = torch.concat((oCond2, buildingConds2), dim=0)
                        buildingBs = torch.concat((relatedBs, buildingBs), dim=0)
                        # allGenTrajs.append(oTrajs)

                        branchLoss = branchLoss + loss.item()
                        lossVal = lossVal + loss.item()
                        ModelClass.oneTimeVisGenAB = False
                        for n in range(self.model.numBranches):
                            if n != m:
                                for p in self.model.branches[n].parameters():
                                    if p.grad != None:
                                        if p.grad.sum() > 0:
                                            print("NONE ZERO GRAD!!! LEAKED DATA TO OTHER BRANCHES")
                        #             # p.grad = None
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
                    self.optimizer.step()
                    color = cmap(m)
                    plt.scatter(iterIndex,branchLoss,c=color)
                    plt.show(block=False)
                    plt.pause(0.01)
                    branchLosses.append(branchLoss)
                    allGenTrajs = buildingGenTrajs
                    allGenTrajs = allGenTrajs.detach()
                    allConds1 = buildingConds1
                    allConds1 = allConds1.detach()
                    allConds2 = buildingConds2
                    allConds2 = allConds2.detach()
                    allBs = buildingBs
                    allBs = allBs.detach()

                ModelClass.fullVisGenAB = False
                formattedLosses = [f"{v:.2f}" for v in branchLosses]
                print(f"LOSSES: {formattedLosses}")
                return lossVal
            else:
                lossVal1 = 0
                lossVal2 = 0
                lossVal3 = 0
                if ModelClass.fullVisGenAB == True:
                    ModelClass.oneTimeVisGenAB = True
                self.optimizer.zero_grad()
                for j in range(total):
                    self.shuffleLocalData()  # MAIN ***
                    batchIndices = np.random.randint(len(self.points), size=BATCH_SIZE)  # MAIN***
                    # batchIndices = np.zeros(BATCH_SIZE)  # DEBUG
                    x_img = self.points[batchIndices]
                    loss = self.train_one_interval_interior(modelVersion, x_img, 0.0, 0.501,
                                                            meanVal, varVal, cr, learningScheduleTime, isRunOnCPU,
                                                            timesteps=timesteps,
                                                            isNoiseSeparate=False,
                                                            isAdvancedExponent=isAdvancedExponent)
                    lossVal1 = lossVal1 + loss.item()
                    ModelClass.oneTimeVisGenAB = False
                for p in self.model.branch2.parameters():
                    if p.grad != None:
                        if p.grad.sum() > 0:
                            print("NONE ZERO GRAD!!!")
                    p.grad = None
                for p in self.model.branch3.parameters():
                    if p.grad != None:
                        if p.grad.sum() > 0:
                            print("NONE ZERO GRAD!!!")
                    p.grad = None
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
                self.optimizer.step()
                if ModelClass.fullVisGenAB == True:
                    ModelClass.oneTimeVisGenAB = True
                self.optimizer.zero_grad()
                for j in range(total):
                    self.shuffleLocalData()  # MAIN ***
                    batchIndices = np.random.randint(len(self.points), size=BATCH_SIZE)  # MAIN***
                    # batchIndices = np.zeros(BATCH_SIZE)  # DEBUG
                    x_img = self.points[batchIndices]
                    loss = self.train_one_interval_interior(modelVersion, x_img, 0.501, 0.701,
                                                            meanVal, varVal, cr, learningScheduleTime, isRunOnCPU,
                                                            timesteps=timesteps,
                                                            isNoiseSeparate=False,
                                                            isAdvancedExponent=isAdvancedExponent)
                    lossVal2 = lossVal2 + loss.item()
                    ModelClass.oneTimeVisGenAB = False
                for p in self.model.branch1.parameters():
                    if p.grad != None:
                        if p.grad.sum() > 0:
                            print("NONE ZERO GRAD!!!")
                    p.grad = None
                for p in self.model.branch3.parameters():
                    if p.grad != None:
                        if p.grad.sum() > 0:
                            print("NONE ZERO GRAD!!!")
                    p.grad = None
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
                self.optimizer.step()
                if ModelClass.fullVisGenAB == True:
                    ModelClass.oneTimeVisGenAB = True
                self.optimizer.zero_grad()
                for j in range(total):
                    self.shuffleLocalData()  # MAIN ***
                    batchIndices = np.random.randint(len(self.points), size=BATCH_SIZE)  # MAIN***
                    # batchIndices = np.zeros(BATCH_SIZE)  # DEBUG
                    x_img = self.points[batchIndices]
                    loss = self.train_one_interval_interior(modelVersion, x_img, 0.701, 1.01,
                                                            meanVal, varVal, cr, learningScheduleTime, isRunOnCPU,
                                                            timesteps=timesteps,
                                                            isNoiseSeparate=False,
                                                            isAdvancedExponent=isAdvancedExponent)
                    lossVal3 = lossVal3 + loss.item()
                    ModelClass.oneTimeVisGenAB = False
                for p in self.model.branch1.parameters():
                    if p.grad != None:
                        if p.grad.sum() > 0:
                            print("NONE ZERO GRAD!!!")
                    p.grad = None
                for p in self.model.branch2.parameters():
                    if p.grad != None:
                        if p.grad.sum() > 0:
                            print("NONE ZERO GRAD!!!")
                    p.grad = None
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
                self.optimizer.step()
                ModelClass.fullVisGenAB = False
                # return lossVal1+lossVal2+lossVal3
                print(f"LOSS1: {lossVal1}, LOSS2: {lossVal2}, LOSS3: {lossVal3}")
                return lossVal1 + lossVal2 + lossVal3



    def train_one_interval_interior(self,modelVersion,x_img,minV,maxV,meanVal, varVal, cr, learningScheduleTime, isRunOnCPU,
                                    timesteps=16, isChangeWeights=True, isAdvancedWeighting=True, isAdvancedExponent=False,
                                    isNoiseSeparate=False, trainBranch=-1, genTrajs=None, allConds1=None, allConds2=None,
                                    addedBCandidates=None):
        x_ts = ModelClass.generate_ts_interval(timesteps,len(x_img), minV, maxV)

        # x_a, x_b = ModelClass.forward_noise_notNormalized(meanVal, varVal, timesteps, x_img, x_ts,
        #                                                    learningScheduleTime, isChangeWeights,
        #                                                    isVizualize=False, isAdvancedExponent=isAdvancedExponent,
        #                                                    isNoiseSeparate=isNoiseSeparate)

        # x_a, x_b = ModelClass.forward_noise_circular(self.maxAvgDist, self.minAvgDist, meanVal, varVal, timesteps, x_img, x_ts,
        #                                             learningScheduleTime, isChangeWeights,
        #                                             isVizualize=True,
        #                                             isAdvancedExponent=isAdvancedExponent,
        #                                             isNoiseSeparate=isNoiseSeparate)

        x_a, x_b = ModelClass.forward_noise_gaussianCircular(self.maxAvgDist, self.minAvgDist, meanVal, varVal, timesteps,
                                                     x_img, x_ts,
                                                     learningScheduleTime, isChangeWeights,
                                                     isVizualize=False,
                                                     isAdvancedExponent=isAdvancedExponent,
                                                     isNoiseSeparate=isNoiseSeparate)

        # for idx in range(x_a.shape[0]):
        #     # if t[idx]<11 or t[idx]>13:#DEBUG
        #     #     continue#DEBUG
        #     # color = cmap(t[idx])
        #     for h in range(1, x_a.shape[1]):
        #         plt.plot([x_a[idx, h - 1, 0], x_a[idx, h, 0]], [x_a[idx, h - 1, 1], x_a[idx, h, 1]],
        #                  marker='',
        #                  zorder=2, alpha=0.08, color='b')
        # for idx in range(x_b.shape[0]):
        #     for h in range(1, x_b.shape[1]):
        #         plt.plot([x_b[idx, h - 1, 0], x_b[idx, h, 0]], [x_b[idx, h - 1, 1], x_b[idx, h, 1]],
        #                  marker='',
        #                  zorder=2, alpha=0.08, color='g')
        # cpuGenTraj = genTrajs.cpu().detach().numpy()
        # for idx in range(cpuGenTraj.shape[0]):
        #     for h in range(1, cpuGenTraj.shape[1]):
        #         plt.plot([cpuGenTraj[idx, h - 1, 0], cpuGenTraj[idx, h, 0]], [cpuGenTraj[idx, h - 1, 1], cpuGenTraj[idx, h, 1]],
        #                  marker='',
        #                  zorder=2, alpha=0.09, color='r')
        # plt.show()

        if x_a.__class__.__name__ == "Tensor":
            x_a = x_a.to(torch.float32)
            x_b = x_b.to(torch.float32)
        elif x_a.__class__.__name__ == "ndarray":
            x_a = torch.from_numpy(x_a).to(torch.float32)
            x_b = torch.from_numpy(x_b).to(torch.float32)
        x_tsN = torch.from_numpy(x_ts / timesteps).to(torch.float32)

        x_tsN = x_tsN.unsqueeze(1)
        cond1 = torch.unsqueeze(x_img[:, 0, 0], dim=1).to(torch.float32)
        cond2 = torch.unsqueeze(x_img[:, 0, 1], dim=1).to(torch.float32)

        if genTrajs.shape[0]>0:
            # numAttached = (int)(x_a.shape[0]*learningScheduleTime)
            # addedBCandidates = x_b[np.where(x_tsN == x_tsN.min())[0], :, :]
            # indicesB = np.random.randint(addedBCandidates.shape[0], size=numAttached)
            # addedB = addedBCandidates[indicesB, :, :]
            addedB = addedBCandidates
            # indicesA = np.random.randint(genTrajs.shape[0], size=numAttached)
            # addedA = genTrajs[indicesA, :, :]
            # addedCond1 = allConds1[indicesA, :]
            # addedCond2 = allConds2[indicesA, :]
            addedA = genTrajs
            addedCond1 = allConds1
            addedCond2 = allConds2
            addedTs = np.expand_dims(np.full((addedB.shape[0]), x_tsN.min()),1)

            x_a = torch.concat((addedA.cpu(), x_a), dim=0)
            x_b = torch.concat((addedB.cpu(), x_b), dim=0)
            cond1 = torch.concat((addedCond1, cond1), dim=0)
            cond2 = torch.concat((addedCond2, cond2), dim=0)
            x_tsN = torch.concat((torch.from_numpy(addedTs).to(torch.float32), x_tsN), dim=0)

        # opt.zero_grad()
        if isRunOnCPU == False:
            if "FLEXMULTIBRANCH" in modelVersion:
                outputTrajs = self.model(x_a.cuda(), x_tsN.cuda(), cond1.cuda(), cond2.cuda(), trainBranch=trainBranch, isTraining=True)
            else:
                outputTrajs = self.model(x_a.cuda(), x_tsN.cuda(), cond1.cuda(), cond2.cuda())

            if "FLEXMULTIBRANCH" in modelVersion:
                if "LOSS" in modelVersion:
                    loss = cr(outputTrajs, x_b.cuda(), x_tsN.cuda(), self.model.branchRanges)
                else:
                    loss = cr(outputTrajs, x_b.cuda(), self.model.branchRanges)
            else:
                if "LOSS" in modelVersion:
                    loss = cr(outputTrajs, x_b.cuda(), x_tsN.cuda())
                else:
                    loss = cr(outputTrajs, x_b.cuda())
        else:
            outputTrajs = self.model(x_a, x_tsN, cond1, cond2)
            if "LOSS" in modelVersion:
                loss = cr(outputTrajs, x_b, x_tsN)
            else:
                loss = cr(outputTrajs, x_b)
        # loss.requires_grad = True
        loss.backward()
        refinedOTs=outputTrajs[np.where(x_tsN==x_tsN.max())[0],:,:]
        refinedCond1s = cond1[np.where(x_tsN == x_tsN.max())[0], :]
        refinedCond2s = cond2[np.where(x_tsN == x_tsN.max())[0], :]
        relatedBs = x_b[np.where(x_tsN == x_tsN.max())[0], :]

        # refinedOTsCPU=refinedOTs.cpu().detach().numpy()
        # relatedBsCPU = relatedBs.cpu().detach().numpy()
        # for idx in range(refinedOTs.shape[0]):
        #     for h in range(1, refinedOTs.shape[1]):
        #         plt.plot([refinedOTsCPU[idx, h - 1, 0], refinedOTsCPU[idx, h, 0]],
        #                  [refinedOTsCPU[idx, h - 1, 1], refinedOTsCPU[idx, h, 1]],
        #                  marker='',
        #                  zorder=2, alpha=0.08, color='b')
        #         plt.plot([relatedBsCPU[idx, h - 1, 0], relatedBsCPU[idx, h, 0]],
        #                  [relatedBsCPU[idx, h - 1, 1], relatedBsCPU[idx, h, 1]],
        #                  marker='',
        #                  zorder=2, alpha=0.08, color='r')
        # plt.show()
        return loss, refinedOTs, refinedCond1s, refinedCond2s, relatedBs.cuda()


    def train_one_pureDiffTime(self, modelVersion, meanVal, varVal, x_img, cr, learningScheduleTime, isRunOnCPU, timesteps=16, isChangeWeights=True, isAdvancedWeighting=True, isAdvancedExponent=False):
        x_ts = ModelClass.generate_ts(timesteps, len(x_img), learningScheduleTime, isChangeWeights,
                                             isAdvancedWeighting=isAdvancedWeighting)

        x_a, x_b  = ModelClass.forward_noise_notNormalized(meanVal, varVal, timesteps, x_img, x_ts,
                                                                 learningScheduleTime, isChangeWeights,
                                                                 isVizualize=True,
                                                                 isAdvancedWeighting=isAdvancedExponent)

        if x_a.__class__.__name__=="Tensor":
            x_a = x_a.to(torch.float32)
            x_b = x_b.to(torch.float32)
        elif x_a.__class__.__name__=="ndarray":
            x_a = torch.from_numpy(x_a).to(torch.float32)
            x_b = torch.from_numpy(x_b).to(torch.float32)
        x_ts = torch.from_numpy(x_ts / timesteps).to(torch.float32)


        x_ts = x_ts.unsqueeze(1)
        cond1 = torch.unsqueeze(x_img[:, 0, 0], dim=1).to(torch.float32)
        cond2 = torch.unsqueeze(x_img[:, 0, 1], dim=1).to(torch.float32)
        # opt.zero_grad()
        if isRunOnCPU == False:
            outputTrajs = self.model(x_a.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
            if "LOSS" in modelVersion:
                loss = cr(outputTrajs, x_b.cuda(), x_ts.cuda())
            else:
                loss = cr(outputTrajs, x_b.cuda())
        else:
            outputTrajs = self.model(x_a, x_ts, cond1, cond2)
            if "LOSS" in modelVersion:
                loss = cr(outputTrajs, x_b, x_ts)
            else:
                loss = cr(outputTrajs, x_b)
        # loss.requires_grad = True
        loss.backward()
        # opt.step()
        # visualiza(x_a.shape[0], x_a)
        # visualiza(x_b.shape[0], x_b)
        return loss

    def shuffleLocalData(self):
        indices = torch.randperm(self.points.shape[0])
        self.points = self.points[indices]
        self.initLats = self.initLats[indices]
        self.initLons = self.initLons[indices]
        self.cutLats = self.cutLats[indices]
        self.cutLons = self.cutLons[indices]
        # self.cells[r][c].isFromInits = cellIsFromInit
        pyList_indices = indices.tolist()
        shuffled_cellIsFromInit = [self.isFromInits[i] for i in pyList_indices]
        self.isFromInits = shuffled_cellIsFromInit

    def showDatas(self):
        # a=torch.diff(self.points, axis=1).cpu().numpy()
        plt.scatter(self.streetPoints[0, :].cpu().numpy(), self.streetPoints[1, :].cpu().numpy(),s=150)
        # xVals = self.points[:, :, 0].cpu().numpy()
        # yVals = self.points[:, :, 1].cpu().numpy()
        for i in range(self.points.shape[0]):
            plt.plot(self.points[i,:,0].cpu().numpy(),self.points[i,:,1].cpu().numpy())
            plt.scatter(self.points[i,:,0].cpu().numpy(),self.points[i,:,1].cpu().numpy(),s=20)
            # print(f"i: {i}")
        np.save("debug_streets.npy",self.streetPoints.cpu().numpy())
        np.save("debug_trajs.npy", self.points.cpu().numpy())
        plt.show()

class TrajCells:
    def __init__(self,r,c,tilesData):
        self.tilesData=tilesData
        self.cells=[]
        for ri in range(r):
            row=[]
            for ci in range(c):
                tc = TrajCell(ri,ci)
                row.append(tc)
            self.cells.append(row)

    def prepareTorchData(self, isRunOnCPU):
        for r in range(len(self.tilesData.trajGrid)):
            for c in range(len(self.tilesData.trajGrid[r])):
                cellPoints = []
                cellInitLats = []
                cellInitLons = []
                cellIsFromInit = []
                cellCutLats = []
                cellCutLons = []
                cellIsContinues = []
                cellStreetPoints=None
                for i in range(len(self.tilesData.trajGrid[r][c])):
                    cellPoints.append(self.tilesData.trajGrid[r][c][i].points)
                    cellInitLats.append(self.tilesData.trajGrid[r][c][i].initLat)
                    cellInitLons.append(self.tilesData.trajGrid[r][c][i].initLon)
                    cellIsFromInit.append(self.tilesData.trajGrid[r][c][i].isFromInit)
                    cellCutLats.append(self.tilesData.trajGrid[r][c][i].cutLat)
                    cellCutLons.append(self.tilesData.trajGrid[r][c][i].cutLon)
                    cellIsContinues.append(self.tilesData.trajGrid[r][c][i].isContinue)
                    cellStreetPoints = numpy.array(self.tilesData.trajGrid[r][c][i].streetPoints)
                if len(self.tilesData.trajGrid[r][c])>0:
                    cellPoints = torch.from_numpy(numpy.array(cellPoints))
                    cellInitLats = torch.from_numpy(numpy.array(cellInitLats))
                    cellInitLons = torch.from_numpy(numpy.array(cellInitLons))
                    # cellIsFromInit = cellIsFromInit
                    cellCutLats = torch.from_numpy(numpy.array(cellCutLats))
                    cellCutLons = torch.from_numpy(numpy.array(cellCutLons))
                    cellIsContinues = torch.from_numpy(numpy.array(cellIsContinues))
                    cellStreetPoints = torch.from_numpy(numpy.array(cellStreetPoints))
                    if isRunOnCPU == False:
                        cellPoints = cellPoints.cuda()
                        cellInitLats = cellInitLats.cuda()
                        cellInitLons = cellInitLons.cuda()
                        cellCutLats = cellCutLats.cuda()
                        cellCutLons = cellCutLons.cuda()
                        cellIsContinues = cellIsContinues.cuda()
                        cellStreetPoints = cellStreetPoints.cuda()
                    self.cells[r][c].points = cellPoints
                    self.cells[r][c].initLats = cellInitLats
                    self.cells[r][c].initLons = cellInitLons
                    self.cells[r][c].cutLats = cellCutLats
                    self.cells[r][c].cutLons = cellCutLons
                    self.cells[r][c].isContinues = cellIsContinues
                    self.cells[r][c].isFromInits = cellIsFromInit
                    self.cells[r][c].streetPoints = cellStreetPoints
                    avgPointDists = torch.diff(self.cells[r][c].points, axis=1).pow(2).sum(dim=2).sqrt().sum(dim=1)
                    self.cells[r][c].maxAvgDist = torch.max(avgPointDists)
                    self.cells[r][c].minAvgDist = torch.min(avgPointDists)



    def loadCellModel(self,r,c,testFolder,modelVersion,initialLR,isRunOnCPU, batch_size=-1):
        if len(self.tilesData.trajGrid[r][c])!=0:
            signature = inspect.signature(ModelClass.SimpleNN.__init__)
            if "maxAvgDist" in signature.parameters:
                model = ModelClass.SimpleNN(avgMaxDist_i=self.cells[r][c].maxAvgDist,
                                            avgMinDist_i=self.cells[r][c].minAvgDist,
                                            forbiddenSerialMap=self.tilesData.trajGrid[r][c][0].streetPoints,
                                            lenForbidden=self.tilesData.trajGrid[r][c][0].streetPoints.shape[1],
                                            maxLengthSize=self.tilesData.trajGrid[r][c][0].points.shape[0],
                                            temporalFeatureSize=2, convOffset=0)
            else:
                model = ModelClass.SimpleNN(forbiddenSerialMap=self.tilesData.trajGrid[r][c][0].streetPoints,
                                            lenForbidden=self.tilesData.trajGrid[r][c][0].streetPoints.shape[1],
                                            maxLengthSize=self.tilesData.trajGrid[r][c][0].points.shape[0],
                                            temporalFeatureSize=2, convOffset=0)



            if isRunOnCPU == False:
                model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=initialLR)
            my_file = Path(
                'genStored/' + testFolder + '/model_' + modelVersion + '_' + str(r) + '_' + str(c) + '.pytorch')
            if my_file.is_file() == True:
                try:
                    model.load_state_dict(torch.load(
                        'genStored/' + testFolder + '/model_' + modelVersion + '_' + str(r) + '_' + str(
                            c) + '.pytorch'))
                    if isRunOnCPU == False:
                        model.cuda()
                    optimizer.load_state_dict(torch.load(
                        'genStored/' + testFolder + '/optimizer_' + modelVersion + '_' + str(r) + '_' + str(
                            c) + '.pytorch'))
                    print("MODEL LOADED! row: "+str(r)+" col: "+str(c))
                    # summarize_weights_by_type(model)
                    # print("MODEL WEIGHTS!")

                    self.cells[r][c].model = model
                    self.cells[r][c].optimizer = optimizer
                except Exception as e:
                    print("FAILED TO LOAD WEIGHTS!")
                    print(f"{e}")
                    self.cells[r][c].model = model
                    if isRunOnCPU == False:
                        model.cuda()
                    self.cells[r][c].optimizer = optimizer
                    print("NEW MODEL INITIALIZED!")
            else:
                self.cells[r][c].model = model
                if isRunOnCPU == False:
                    model.cuda()
                self.cells[r][c].optimizer = optimizer
                print("MODEL NOT FOUND, NEW MODEL INITIALIZED!")
            self.cells[r][c].model.batchSize=batch_size
            if hasattr(self.cells[r][c].model, 'defaultGate'):
                self.cells[r][c].model.defaultGate = torch.ones(batch_size, device=torch.device('cuda')).unsqueeze(dim=1).unsqueeze(
                    dim=2)
                if isRunOnCPU == False:
                    self.cells[r][c].model.defaultGate = self.cells[r][c].model.defaultGate.cuda()

    def saveCellModel(self, r, c, testFolder, modelVersion):
        Path('genStored/' + testFolder).mkdir(parents=True, exist_ok=True)
        torch.save(self.cells[r][c].model.state_dict(), 'genStored/' + testFolder + '/model_' + modelVersion + '_' + str(r) + '_' + str(c) + '.pytorch')
        torch.save(self.cells[r][c].optimizer.state_dict(), 'genStored/' + testFolder + '/model_' + modelVersion + '_' + str(r) + '_' + str(c) + '.pytorch')
        print("MODEL SAVED!")

    def unloadCellModel(self,r,c):
        self.cells[r][c].model.cpu()
        del self.cells[r][c].model
        del self.cells[r][c].optimizer
        # self.cells[r][c].points.detach().cpu()

        torch.cuda.empty_cache()
        gc.collect()
        self.cells[r][c].model = None

    def visSaveNetwork(self,r,c,testFolder,modelVersion,BATCH_SIZE,maxTrajectoryLength,isRunOnCPU):
        # Visualizing the graph
        # Test input for visualization
        in1 = torch.randn(BATCH_SIZE, maxTrajectoryLength, 2)
        in2 = torch.randn(BATCH_SIZE, 1)
        cond1 = torch.randn(BATCH_SIZE, 1)
        cond2 = torch.randn(BATCH_SIZE, 1)
        # Forward pass
        model = self.cells[r][c].model
        if "FULLBRANCH" in modelVersion:
            model.branch1.psl.B = torch.from_numpy(model.branch1.psl.B).cuda()
            model.branch1.pslSum.B = torch.from_numpy(model.branch1.pslSum.B).cuda()

            model.branch2.psl.B = torch.from_numpy(model.branch2.psl.B).cuda()
            model.branch2.pslSum.B = torch.from_numpy(model.branch2.pslSum.B).cuda()

            model.branch3.psl.B = torch.from_numpy(model.branch3.psl.B).cuda()
            model.branch3.pslSum.B = torch.from_numpy(model.branch3.pslSum.B).cuda()
        elif "FLEXMULTIBRANCH" in modelVersion:
            for m in range(model.numBranches):
                model.branches[m].psl.B = torch.from_numpy(model.branches[m].psl.B).cuda()
                model.branches[m].pslSum.B = torch.from_numpy(model.branches[m].pslSum.B).cuda()
        else:
            model.psl.B = torch.from_numpy(model.psl.B).cuda()
            model.pslSum.B = torch.from_numpy(model.pslSum.B).cuda()
        if hasattr(model, 'defaultGate'):
            model.defaultGate = torch.ones(BATCH_SIZE, device=torch.device('cuda')).unsqueeze(dim=1).unsqueeze(
                dim=2)
            if isRunOnCPU == False:
                model.defaultGate = model.defaultGate.cuda()
        if isRunOnCPU == False:
            model = model.cuda()
            in1 = in1.cuda()
            in2 = in2.cuda()
            cond1 = cond1.cuda()
            cond2 = cond2.cuda()
        y = model(in1, in2, cond1, cond2)

        if isRunOnCPU == False:
            graph = draw_graph(model, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cuda")
        else:
            graph = draw_graph(model, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cpu")

        graph.visual_graph.render('genStored/' + testFolder + '/model_' + modelVersion + '_' + str(r) + '_' + str(c), format="png")

    def initCellModel(self,r,c,initialLR,isRunOnCPU, batch_size=-1):
        model = ModelClass.SimpleNN(forbiddenSerialMap=self.tilesData.trajGrid[r][c][0].streetPoints,
                                           lenForbidden=self.tilesData.trajGrid[r][c][0].streetPoints.shape[1],
                                           maxLengthSize=self.tilesData.trajGrid[r][c][0].points.shape[0], temporalFeatureSize=2, convOffset=0)
        optimizer = optim.Adam(model.parameters(), lr=initialLR)
        if isRunOnCPU == False:
            model.cuda()
        self.cells[r][c].model = model
        self.cells[r][c].optimizer = optimizer
        self.cells[r][c].model.batchSize = batch_size
        if hasattr(self.cells[r][c].model, 'defaultGate'):
            self.cells[r][c].model.defaultGate = torch.ones(batch_size, device=torch.device('cuda')).unsqueeze(
                dim=1).unsqueeze(
                dim=2)
            if isRunOnCPU == False:
                self.cells[r][c].model.defaultGate = self.cells[r][c].model.defaultGate.cuda()

    def shuffleData(self):
        time = datetime.now()  # MAIN
        torch.manual_seed(time.minute + time.hour + time.microsecond)
        for r in range(len(self.cells)):
            for c in range(len(self.cells[r])):
                if self.cells[r][c].points!=None:
                    indices = torch.randperm(self.cells[r][c].points.shape[0])
                    self.cells[r][c].points = self.cells[r][c].points[indices]
                    self.cells[r][c].initLats = self.cells[r][c].initLats[indices]
                    self.cells[r][c].initLons = self.cells[r][c].initLons[indices]
                    self.cells[r][c].cutLats = self.cells[r][c].cutLats[indices]
                    self.cells[r][c].cutLons = self.cells[r][c].cutLons[indices]
                    # self.cells[r][c].isFromInits = cellIsFromInit
                    pyList_indices = indices.tolist()
                    shuffled_cellIsFromInit = [self.cells[r][c].isFromInits[i] for i in pyList_indices]
                    self.cells[r][c].isFromInits = shuffled_cellIsFromInit
                    # print("!!!")

    @staticmethod
    def visualize_grid(selectedOrNotSelected, numInstances, input, maxTrajectoryLength, nGrid, axis=None, auxSelectedOrNotSelected=None):
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numInstances)
        for i in range(numInstances):
            for h in range(1, maxTrajectoryLength):
                color = cmap(i)
                if axis == None:
                    plt.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='', zorder=2, alpha=0.5)
                else:
                    axis.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                              marker='', zorder=2, alpha=0.5)

        if auxSelectedOrNotSelected is not None:
            selectedOrNotSelected = selectedOrNotSelected - 1 * auxSelectedOrNotSelected
        if axis == None:
            plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                       extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                       origin='lower', zorder=1, alpha=0.99)
        else:
            axis.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                        extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                        origin='lower', zorder=1, alpha=0.99)
        # plt.grid(True)
        if axis == None:
            plt.show()

    @staticmethod
    def visualize_extent(selectedOrNotSelected, numInstances, input, selectedOrNotSelectedSerialized, maxTrajectoryLength, axis=None,
                         auxSelectedOrNotSelected=None, saveName=None):
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numInstances)
        minX = selectedOrNotSelectedSerialized[0, :].cpu().detach().numpy().min()
        maxX = selectedOrNotSelectedSerialized[0, :].cpu().detach().numpy().max()
        minY = selectedOrNotSelectedSerialized[1, :].cpu().detach().numpy().min()
        maxY = selectedOrNotSelectedSerialized[1, :].cpu().detach().numpy().max()
        for i in range(numInstances):
            for h in range(1, maxTrajectoryLength):
                color = cmap(i)
                if axis == None:
                    plt.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                             marker='', zorder=2, alpha=0.5)
                else:
                    axis.plot([input[i, h - 1, 0], input[i, h, 0]], [input[i, h - 1, 1], input[i, h, 1]], color=color,
                              marker='', zorder=2, alpha=0.5)

        if auxSelectedOrNotSelected is not None:
            selectedOrNotSelected = selectedOrNotSelected - 1 * auxSelectedOrNotSelected
        if axis == None:
            plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                       extent=(minX, maxX, minY, maxY),
                       origin='lower', zorder=1, alpha=0.99)
        else:
            axis.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                        extent=(minX, maxX, minY, maxY),
                        origin='lower', zorder=1, alpha=0.99)
        # plt.grid(True)
        if axis == None:
            if saveName != None:
                plt.savefig(saveName + "_Results.png")
            plt.show()

    def predict(self, r,c, numPredicts, maxTrajectoryLength, meanVal, varVal, nGrid, scale, isRunOnCPU, modelVersion, timesteps=16):
        isEnded=False
        while isEnded==False:
            isEnded=True
            pred = self.predictFromInit(r,c,self.cells[r][c].streetPoints, maxTrajectoryLength, timesteps, meanVal, varVal, isRunOnCPU,
                                    modelVersion, numTraj=numPredicts)

            pred = pred.cpu().detach().numpy()

            streetPoints2D = Tiles.TilesSynthetic.gen2DMapFromStreetPoints(pred, nGrid)

            JSDValue = JSD(nGrid, self.cells[r][c].points.cpu().numpy(), pred, streetPoints2D, scale=scale)

            print("JSDValue")
            print(JSDValue.JSDValue)

            JSDValue_SingleB = JSD_SingleB(nGrid, self.cells[r][c].points.cpu().numpy(), pred, streetPoints2D)

            print("JSDValue_singleB")
            print(JSDValue_SingleB.JSDValue)
            print("B value")
            print(JSDValue_SingleB.minBValue)

            TrajCells.visualize_extent(streetPoints2D, numPredicts, pred, self.cells[r][c].streetPoints,
                                       maxTrajectoryLength, saveName=modelVersion)
            print("!!!")
            return pred, streetPoints2D



    def predictFromInit(self, r,c,serializedSelected, trajectoryLength, timesteps, meanVal, varVal, isRunOnCPU, modelVersion, numTraj=10):
        x = np.random.normal(loc=meanVal, scale=varVal, size=(numTraj, trajectoryLength, 2))
        # OLD
        # x = np.random.normal(loc=0.5, scale=0.5, size=(numTraj, trajectoryLength, 2))
        # x = np.random.normal(loc=0.5,scale=0.33,size=(numTraj, trajectoryLength, 2))
        # x = np.random.uniform(low=0, high=1, size=(numTraj, trajectoryLength, 2))

        for idx in range(x.shape[0]):  # MAIN
            for h in range(1, trajectoryLength):  # MAIN
                plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
                         zorder=2, alpha=0.5, color='g')  # MAIN
        plt.show()  # MAIN

        x = torch.from_numpy(x).to(torch.float32)
        self.cells[r][c].model.psl.lenForbidden = serializedSelected.shape[1]
        self.cells[r][c].model.pslSum.lenForbidden = serializedSelected.shape[1]
        # if isRunOnCPU == False:
        #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
        # else:
        #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
        self.cells[r][c].model.psl.B = serializedSelected
        self.cells[r][c].model.pslSum.B = serializedSelected

        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, timesteps)

        # fig, axs = plt.subplots(nrows=1, ncols=(int)(timesteps/1))
        with torch.no_grad():
            # indices = np.random.choice(serializedSelected.shape[1], numTraj, replace=True)
            # samples = serializedSelected[:,indices]
            # samples=torch.transpose(samples,0,1)
            # cond1 = torch.unsqueeze(samples[:, 0], dim=1)
            # cond2 = torch.unsqueeze(samples[:, 1], dim=1)

            indices = np.random.choice(self.cells[r][c].initLats.shape[0], numTraj, replace=True)
            cond1 = np.zeros((numTraj, 1))
            cond2 = np.zeros((numTraj, 1))
            for i in range(numTraj):
                cond1[i, 0] = self.cells[r][c].initLats[indices[i]]
                cond2[i, 0] = self.cells[r][c].initLons[indices[i]]
            cond1 = torch.from_numpy(cond1).to(torch.float32)
            cond2 = torch.from_numpy(cond2).to(torch.float32)

            # cond1 = torch.full((numTraj, 1), initLat)
            # cond2 = torch.full((numTraj, 1), initLon)
            for i in trange(timesteps):
                color = cmap(i)
                resX = x.cpu().detach().numpy()
                for idx in range(x.shape[0]):  # MAIN
                    for h in range(1, trajectoryLength):  # MAIN
                        plt.plot([resX[idx, h - 1, 0], resX[idx, h, 0]], [resX[idx, h - 1, 1], resX[idx, h, 1]],
                                 marker='',
                                 zorder=2, alpha=0.5, color=color)  # MAIN
                plt.show(block=False)
                plt.pause(0.5)

                ## colVal=i%(int)(timesteps/2)
                ## rowVal=math.floor(i/(int)(timesteps/2))
                # cond1 = torch.unsqueeze(x[:, 0, 0], dim=1)
                # cond2 = torch.unsqueeze(x[:, 0, 1], dim=1)
                t = i / timesteps
                x_ts = np.pow(np.full((numTraj), t), 7.0)
                x_ts = torch.from_numpy(x_ts).to(torch.float32)
                x_ts = x_ts.unsqueeze(1)
                if isRunOnCPU == False:
                    x_res = self.cells[r][c].model(x.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
                    x = x_res
                else:
                    x_res = self.cells[r][c].model(x, x_ts, cond1, cond2)
                    x = x_res
        #         visualiza(selectedOrNotSelected, numTraj, x.cpu().detach().numpy(), axis=axs[i])
        #         axs[i].title.set_text("Time: "+str(i))

        plt.savefig(modelVersion + "_Results_gradient.png")
        plt.show()  # MAIN
        return x

    def predictDetachedFromInit(self, r,c,serializedSelected, trajectoryLength, timesteps, meanVal, varVal, isRunOnCPU, modelVersion, numTraj=10):
        x = np.random.normal(loc=meanVal, scale=varVal, size=(numTraj, trajectoryLength, 2))
        # OLD
        # x = np.random.normal(loc=0.5, scale=0.5, size=(numTraj, trajectoryLength, 2))
        # x = np.random.normal(loc=0.5,scale=0.33,size=(numTraj, trajectoryLength, 2))
        # x = np.random.uniform(low=0, high=1, size=(numTraj, trajectoryLength, 2))

        for idx in range(x.shape[0]):  # MAIN
            for h in range(1, trajectoryLength):  # MAIN
                plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
                         zorder=2, alpha=0.5, color='g')  # MAIN
        plt.show()  # MAIN

        x = torch.from_numpy(x).to(torch.float32)
        self.cells[r][c].model.psl.lenForbidden = serializedSelected.shape[1]
        self.cells[r][c].model.pslSum.lenForbidden = serializedSelected.shape[1]
        # if isRunOnCPU == False:
        #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
        # else:
        #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
        self.cells[r][c].model.psl.B = serializedSelected
        self.cells[r][c].model.pslSum.B = serializedSelected

        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, timesteps)

        # fig, axs = plt.subplots(nrows=1, ncols=(int)(timesteps/1))
        with torch.no_grad():
            # indices = np.random.choice(serializedSelected.shape[1], numTraj, replace=True)
            # samples = serializedSelected[:,indices]
            # samples=torch.transpose(samples,0,1)
            # cond1 = torch.unsqueeze(samples[:, 0], dim=1)
            # cond2 = torch.unsqueeze(samples[:, 1], dim=1)

            indices = np.random.choice(self.cells[r][c].initLats.shape[0], numTraj, replace=True)
            cond1 = np.zeros((numTraj, 1))
            cond2 = np.zeros((numTraj, 1))
            for i in range(numTraj):
                cond1[i, 0] = self.cells[r][c].initLats[indices[i]]
                cond2[i, 0] = self.cells[r][c].initLons[indices[i]]
            cond1 = torch.from_numpy(cond1).to(torch.float32)
            cond2 = torch.from_numpy(cond2).to(torch.float32)

            # cond1 = torch.full((numTraj, 1), initLat)
            # cond2 = torch.full((numTraj, 1), initLon)
            for i in trange(timesteps):
                color = cmap(i)
                resX = x.cpu().detach().numpy()
                for idx in range(x.shape[0]):  # MAIN
                    for h in range(1, trajectoryLength):  # MAIN
                        plt.plot([resX[idx, h - 1, 0], resX[idx, h, 0]], [resX[idx, h - 1, 1], resX[idx, h, 1]],
                                 marker='',
                                 zorder=2, alpha=0.5, color=color)  # MAIN

                ## colVal=i%(int)(timesteps/2)
                ## rowVal=math.floor(i/(int)(timesteps/2))
                # cond1 = torch.unsqueeze(x[:, 0, 0], dim=1)
                # cond2 = torch.unsqueeze(x[:, 0, 1], dim=1)
                t = i / timesteps
                x_ts = np.pow(np.full((numTraj), t), 1.0)
                x_ts = torch.from_numpy(x_ts).to(torch.float32)
                x_ts = x_ts.unsqueeze(1)
                if isRunOnCPU == False:
                    x_res = self.cells[r][c].model(x.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
                    x = x_res
                else:
                    x_res = self.cells[r][c].model(x, x_ts, cond1, cond2)
                    x = x_res
        #         visualiza(selectedOrNotSelected, numTraj, x.cpu().detach().numpy(), axis=axs[i])
        #         axs[i].title.set_text("Time: "+str(i))

        plt.savefig(modelVersion + "_Results_gradient.png")
        plt.show()  # MAIN
        return x

    def predictDEBUG(self, r,c, modelVersion, numPredicts, maxTrajectoryLength, timesteps, nGrid, scale, meanVal, varVal, isRunOnCPU):
        # r,c, modelVersion, self.cells[r][c].streetPoints, meanVal, varVal, trajectoryLength, timesteps, isRunOnCPU, numTraj=10
        pred = self.predictInitDEBUG(r,c, modelVersion, self.cells[r][c].streetPoints, meanVal, varVal,
                                     maxTrajectoryLength, timesteps, isRunOnCPU, numTraj=numPredicts)

        pred = pred.cpu().detach().numpy()

        streetPoints2D = Tiles.TilesSynthetic.gen2DMapFromStreetPoints(pred, nGrid)

        JSDValue = JSD(nGrid, self.cells[r][c].points.cpu().numpy(), pred, streetPoints2D, scale=scale)

        print("JSDValue")
        print(JSDValue.JSDValue)

        JSDValue_SingleB = JSD_SingleB(nGrid, self.cells[r][c].points.cpu().numpy(), pred, streetPoints2D)

        print("JSDValue_singleB")
        print(JSDValue_SingleB.JSDValue)
        print("B value")
        print(JSDValue_SingleB.minBValue)

        # visualize_extent(serializedSelected2DMatrixs[c], numPredicts, pred, selectedOrNotSelecteds[c],
        #                  saveName=modelVersion)
        #
        # mainSelectedOrNotSelected = serializedSelected2DMatrixs[0]
        #
        # # FANCY TEST! CHANGE THE CITY AND TEST!!!
        # # [dataT,nGrid,selectedOrNotSelectedT,serializedSelected2DMatrixT,_]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=1,visualize=False)
        # newNumGrid = 40
        # [data, nGrid, selectedOrNotSelectedT,
        #  serializedSelected2DMatrixT] = DataGeneratorFcn.generateSyntheticDataVariableLengthInputImageLastRepeat(
        #     "testStreets3.png", numTrajectories=numTrajectories,
        #     maxTrajectoryLength=maxTrajectoryLength, numGrid=newNumGrid,
        #     seed=123, visualize=False)
        # serializedSelected2DMatrixT = serializedSelected2DMatrixT * scale
        # data = data * scale
        #
        # # serializedSelected2DMatrixT=serializedSelected2DMatrixT[:,0:302]
        # if isRunOnCPU == False:
        #     serializedSelected2DMatrixT = torch.from_numpy(serializedSelected2DMatrixT).cuda()
        # else:
        #     serializedSelected2DMatrixT = torch.from_numpy(serializedSelected2DMatrixT)
        # model.psl.B = serializedSelected2DMatrixT
        # model.pslSum.B = serializedSelected2DMatrixT
        #
        # numPredicts = 20
        # pred = predict(serializedSelected2DMatrixT, maxTrajectoryLength, numTraj=numPredicts)
        #
        # pred = pred.cpu().detach().numpy()
        #
        # JSDValue = JSD(newNumGrid, data, pred, selectedOrNotSelectedT)
        #
        # print("JSDValue other city: ")
        # print(JSDValue.JSDValue)
        #
        # JSDValue_SingleB = JSD_SingleB(newNumGrid, data, pred, selectedOrNotSelected)
        #
        # print("JSDValue_singleB other city: ")
        # print(JSDValue_SingleB.JSDValue)
        # print("B value other city: ")
        # print(JSDValue_SingleB.minBValue)
        #
        # visualize_extent(selectedOrNotSelectedT, numPredicts, pred, serializedSelected2DMatrixT,
        #                  auxSelectedOrNotSelected=mainSelectedOrNotSelected, saveName=modelVersion + "_anotherCity_")
        #
        # # torch.save(model.state_dict(), 'model'+modelVersion+'.pytorch')

        print("!!!")
        return pred, streetPoints2D

    def predictInitDEBUG(self, r,c, modelVersion, serializedSelected, meanVal, varVal, trajectoryLength, timesteps, isRunOnCPU, numTraj=10):
        x_gauss = np.random.normal(loc=meanVal, scale=varVal, size=(numTraj, trajectoryLength, 2))

        # initialAngle = np.random.uniform(0.0, 2*np.pi, size=(numTraj, 1))
        # angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(numTraj, trajectoryLength - 1))
        # anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # # radii = np.random.normal(loc=(0, 0), scale=varVal / 12.0, size=(numTraj, trajectoryLength, 2)) + varVal * 10.0
        # radii = np.random.normal(loc=(0, 0), scale=varVal / 12.0, size=(numTraj, trajectoryLength, 2)) + meanVal
        # x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVal[0]
        # y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVal[1]
        # xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVal[0]
        # yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVal[1]
        # allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        # allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        # x = np.concat((allXs, allYs), axis=2)
        #
        # x = (x+x_gauss*2)/3

        x = x_gauss

        # OLD
        # x = np.random.normal(loc=0.5, scale=0.5, size=(numTraj, trajectoryLength, 2))
        # x = np.random.normal(loc=0.5,scale=0.33,size=(numTraj, trajectoryLength, 2))
        # x = np.random.uniform(low=0, high=1, size=(numTraj, trajectoryLength, 2))

        for idx in range(x.shape[0]):  # MAIN
            for h in range(1, trajectoryLength):  # MAIN
                plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
                         zorder=2, alpha=0.5, color='g')  # MAIN
        plt.show()  # MAIN

        x = torch.from_numpy(x).to(torch.float32)
        if "FULLBRANCH" in modelVersion:
            self.cells[r][c].model.branch1.psl.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.branch1.pslSum.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.branch1.psl.B = serializedSelected
            self.cells[r][c].model.branch1.pslSum.B = serializedSelected

            self.cells[r][c].model.branch2.psl.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.branch2.pslSum.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.branch2.psl.B = serializedSelected
            self.cells[r][c].model.branch2.pslSum.B = serializedSelected

            self.cells[r][c].model.branch3.psl.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.branch3.pslSum.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.branch3.psl.B = serializedSelected
            self.cells[r][c].model.branch3.pslSum.B = serializedSelected
        elif "FLEXMULTIBRANCH" in modelVersion:
            for m in range(self.cells[r][c].model.numBranches):
                self.cells[r][c].model.branches[m].psl.lenForbidden = serializedSelected.shape[1]
                self.cells[r][c].model.branches[m].pslSum.lenForbidden = serializedSelected.shape[1]
                self.cells[r][c].model.branches[m].psl.B = serializedSelected
                self.cells[r][c].model.branches[m].pslSum.B = serializedSelected
        else:
            self.cells[r][c].model.psl.lenForbidden = serializedSelected.shape[1]
            self.cells[r][c].model.pslSum.lenForbidden = serializedSelected.shape[1]
            # if isRunOnCPU == False:
            #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
            # else:
            #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
            self.cells[r][c].model.psl.B = serializedSelected
            self.cells[r][c].model.pslSum.B = serializedSelected


        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, timesteps)

        # fig, axs = plt.subplots(nrows=1, ncols=(int)(timesteps/1))
        with torch.no_grad():
            # indices = np.random.choice(serializedSelected.shape[1], numTraj, replace=True)
            # samples = serializedSelected[:,indices]
            # samples=torch.transpose(samples,0,1)
            # cond1 = torch.unsqueeze(samples[:, 0], dim=1)
            # cond2 = torch.unsqueeze(samples[:, 1], dim=1)

            indices = np.random.choice(self.cells[r][c].cutLats.shape[0], numTraj, replace=True)
            cond1 = np.zeros((numTraj, 1))
            cond2 = np.zeros((numTraj, 1))
            for i in range(numTraj):
                cond1[i, 0] = self.cells[r][c].cutLats[indices[i]]
                cond2[i, 0] = self.cells[r][c].cutLons[indices[i]]
            cond1 = torch.from_numpy(cond1).to(torch.float32)
            cond2 = torch.from_numpy(cond2).to(torch.float32)

            x=TrajCells.predictLoop(self.cells[r][c].model,x,numTraj, timesteps, cond1, cond2, isRunOnCPU, cmap, trajectoryLength)

        plt.savefig(modelVersion + "_Results_gradient.png")
        # plt.show()  # MAIN
        xFinal=x.cpu().detach().numpy()
        streetsCPU=self.cells[r][c].streetPoints.cpu().detach().numpy()
        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, x.shape[0])
        for m in range(x.shape[0]):
            plt.scatter(streetsCPU[0,:],streetsCPU[1,:],c='blue',alpha=0.2)
            color = cmap(m)
            plt.plot(xFinal[m,:,0],xFinal[m,:,1],color=color)
        plt.show(block=False)
        plt.pause(1)

        xFinalPrev=xFinal
        qualityThreshold=7.0
        qualities=TrajCells.getTrajQualities(modelVersion,self.cells[r][c].model,x).cpu()
        # print(qualities)
        numberOfTries=0
        while qualities.min()<qualityThreshold:
            updatingIndices = torch.argwhere(qualities.squeeze() < qualityThreshold).squeeze()
            #DEBUG
            # updatingIndices = torch.tensor(6)
            goodIndices = torch.argwhere(qualities.squeeze() >= qualityThreshold).squeeze()
            cond1R = cond1[updatingIndices]
            cond2R = cond2[updatingIndices]
            if updatingIndices.dim()==0:
                numTrajR=1
                cond1R = cond1R.unsqueeze(dim=1)
                cond2R = cond2R.unsqueeze(dim=1)
            else:
                numTrajR = updatingIndices.shape[0]
            xR_gauss = np.random.normal(loc=meanVal, scale=varVal, size=(numTrajR, trajectoryLength, 2))

            # initialAngle = np.random.uniform(0.0, 2 * np.pi, size=(numTrajR, 1))
            # angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(numTrajR, trajectoryLength - 1))
            # anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
            # # radii = np.random.normal(loc=(0, 0), scale=varVal / 12.0,
            # #                          size=(numTrajR, trajectoryLength, 2)) + varVal * 10.0
            # radii = np.random.normal(loc=(0, 0), scale=varVal / 12.0, size=(numTraj, trajectoryLength, 2)) + meanVal
            # x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVal[0]
            # y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVal[1]
            # xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVal[0]
            # yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVal[1]
            # allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
            # allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
            # xR = np.concat((allXs, allYs), axis=2)
            #
            # xR = (xR+xR_gauss*2)/3

            xR = xR_gauss

            if hasattr(self.cells[r][c].model, 'setDefaultGate') and callable(getattr(self.cells[r][c].model, 'setDefaultGate')):
                self.cells[r][c].model.setDefaultGate(numTrajR, isRunOnCPU)
            xR = torch.from_numpy(xR).to(torch.float32)
            with torch.no_grad():
                xR = TrajCells.predictLoop(self.cells[r][c].model, xR, numTrajR, timesteps, cond1R, cond2R, isRunOnCPU, cmap, trajectoryLength)

            xFinalR = xR.cpu().detach().numpy()
            mixedX=np.zeros((numTraj, trajectoryLength, 2))
            if updatingIndices.dim() == 0:
                mixedX[updatingIndices, :, :] = xFinalR
            else:
                for i in range(len(updatingIndices)):
                    mixedX[updatingIndices[i], :, :] = xFinalR[i]
            if goodIndices.dim() == 0:
                mixedX[goodIndices, :, :] = xFinalPrev[goodIndices]
            else:
                for i in range(len(goodIndices)):
                    mixedX[goodIndices[i], :, :] = xFinalPrev[goodIndices[i]]

            # mixedX = np.concat([xFinal[goodIndices], xFinalR], axis=0)
            mixedXT=torch.from_numpy(mixedX).to(torch.float32)
            if isRunOnCPU == False:
                mixedXT=mixedXT.cuda()
            # print(qualities.min())
            qualities = TrajCells.getTrajQualities(modelVersion,self.cells[r][c].model, mixedXT).cpu()
            # print(qualities.min())
            # print(qualities)
            xFinalPrev=mixedX
            # print("END")
            cmap_name = 'jet'  # Example: Use the 'jet' colormap
            cmap = cm.get_cmap(cmap_name, mixedX.shape[0])
            # plt.figure()
            # for m in range(mixedX.shape[0]):
            #     plt.scatter(streetsCPU[0, :], streetsCPU[1, :], c='blue',alpha=0.2)
            #     color = cmap(m)
            #     plt.plot(mixedX[m, :, 0], mixedX[m, :, 1], color=color)
            if qualities.min()<qualityThreshold:
                # plt.show(block=False)
                # plt.pause(1)
                print(f"STILL BAD TRAJECTORIES! {qualities.min()} {numTrajR}")
            else:
                plt.figure()
                for m in range(mixedX.shape[0]):
                    plt.scatter(streetsCPU[0, :], streetsCPU[1, :], c='blue', alpha=0.2)
                    color = cmap(m)
                    plt.plot(mixedX[m, :, 0], mixedX[m, :, 1], color=color)
                print(f"Number of tries: {numberOfTries}")
                plt.show(block=True)
            if numberOfTries>50:
                plt.figure()
                for m in range(mixedX.shape[0]):
                    plt.scatter(streetsCPU[0, :], streetsCPU[1, :], c='blue', alpha=0.2)
                    color = cmap(m)
                    plt.plot(mixedX[m, :, 0], mixedX[m, :, 1], color=color)
                print(f"Number of tries: {numberOfTries}")
                plt.show(block=True)
                x=mixedX
                break
            numberOfTries=numberOfTries+1
            # print("!!!")

        # plt.pause(0.1)
        return x

    @staticmethod
    def predictLoop(model,x,numTraj, timesteps, cond1, cond2, isRunOnCPU, cmap, maxTrajectoryLength):
            for i in trange(timesteps):
                # color = cmap(i)
                # resX = x.cpu().detach().numpy()
                # for idx in range(x.shape[0]):  # MAIN
                #     for h in range(1, maxTrajectoryLength):  # MAIN
                #         plt.plot([resX[idx, h - 1, 0], resX[idx, h, 0]], [resX[idx, h - 1, 1], resX[idx, h, 1]],
                #                  marker='',
                #                  zorder=2, alpha=0.5, color=color)  # MAIN
                # if i!=timesteps-1:
                #     plt.show(block=False)
                #     plt.pause(0.1)
                # else:
                #     plt.show()
                t = i / timesteps
                x_ts = np.pow(np.full((numTraj), t), 1.0)
                # temp = np.full((numTraj), t)
                # x_ts = np.pow(temp, 3) / (np.pow(temp, 5) + (1 - temp) * np.exp((2.0) * temp))
                x_ts = torch.from_numpy(x_ts).to(torch.float32)
                x_ts = x_ts.unsqueeze(1)
                if isRunOnCPU == False:
                    x_res = model(x.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
                    x = x_res
                else:
                    x_res = model(x, x_ts, cond1, cond2)
                    x = x_res
                # for m in range(model.numBranches):
                #     timeValues1 = model.allCustomFcnsGenerative[m](x_ts).squeeze(1)
                #     if timeValues1.sum()>0:
                #         print(f"GATE: {m}, active. Time: {t}")
            # plt.show()
            return x

    @staticmethod
    def getTrajQualities(modelVersion,model,input):
        if "FULLBRANCH" in modelVersion:
            plsPathSum = model.branch1.pslSum(input)
        elif "FLEXMULTIBRANCH" in modelVersion:
            plsPathSum = model.branches[0].pslSum(input)
        else:
            plsPathSum = model.pslSum(input)
        return plsPathSum