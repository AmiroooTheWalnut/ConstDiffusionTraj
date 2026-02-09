import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from triton.language import dtype, tensor
import matplotlib.cm as cm
import math

fullVisGenAB=True
oneTimeVisGenAB=True

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

class CustomStepFcn(nn.Module):
    def __init__(self, x0_init):
        super(CustomStepFcn,self).__init__()
        self.x0 = torch.tensor(x0_init)

    def forward(self, t):
        with torch.no_grad():
            return (t<self.x0).to(t.dtype)

class CustomStepFcnTwoSided(nn.Module):
    def __init__(self, x0_init, x1_init):
        super(CustomStepFcnTwoSided,self).__init__()
        self.x0 = torch.tensor(x0_init)
        self.x1 = torch.tensor(x1_init)

    def forward(self, t):
        with torch.no_grad():
            return ((t>=self.x0) & (t<=self.x1)).to(t.dtype)

class UNet1D(nn.Module):
    def __init__(self, input_channels=2, base_channels=64, num_blocks=4):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, base_channels, kernel_size=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.res_skips = []

        ch = base_channels
        for i in range(num_blocks):
            self.downs.append(nn.Sequential(
                ResidualBlock1D(ch),
                nn.Conv1d(ch, ch * 2, kernel_size=2, stride=2),  # Downsample
            ))
            ch *= 2

        for i in range(num_blocks):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=2, stride=2),  # Upsample
                ResidualBlock1D(ch // 2),
            ))
            ch //= 2

        self.output_proj = nn.Conv1d(base_channels, input_channels, kernel_size=1)

    def forward(self, x):
        # Input: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips = skips[::-1]
        for i, up in enumerate(self.ups):
            x = up(x)
            # Crop if shape mismatch due to odd-length sequences
            if x.shape[-1] > skips[i].shape[-1]:
                x = x[..., :skips[i].shape[-1]]
            elif x.shape[-1] < skips[i].shape[-1]:
                skips[i] = skips[i][..., :x.shape[-1]]
            x = x + skips[i]  # Residual skip connection

        x = self.output_proj(x)
        return x.permute(0, 2, 1)  # Back to (B, T, F)

class CustomLoss(nn.Module):
    def __init__(self, forbiddens,lenForbidden,maxLengthSize,timesteps):
        super(CustomLoss, self).__init__()
        self.forbiddens = forbiddens
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize
        self.timesteps=timesteps

    def custom_activation(self,x):
        a = torch.sigmoid(8*x)
        b = torch.sigmoid(-8*x)
        return a * b

    def forward(self, y_pred, y_true, time, branchRanges=None):
        mae_loss = torch.abs(y_true - y_pred)
        # A_expanded = torch.unsqueeze(y_pred, 3)
        # B_expanded = torch.unsqueeze(self.forbiddens, 0)
        # C = A_expanded - B_expanded
        # # C1 = self.custom_activation(C)
        # processedTime = torch.relu(((torch.squeeze(time) + 1)/self.timesteps)-0.6)*1
        # # processedTime = torch.squeeze(time)
        # processedTime = processedTime*processedTime
        # # penaltyPred = processedTime * (torch.mean(C1, dim=(1, 2, 3), keepdim=False))
        #
        # minsPred = torch.min(torch.abs(torch.sum(C, 2, keepdim=True)), 3).values
        #
        # penaltyPred = processedTime * (torch.mean(minsPred, dim=(1, 2), keepdim=False))
        #
        # A_expanded = torch.unsqueeze(y_true, 3)
        # # B_expanded = torch.unsqueeze(self.forbiddens, 0)
        # C = A_expanded - B_expanded
        # # C1 = self.custom_activation(C)
        # # processedTime = torch.relu(((torch.squeeze(time) + 1) / self.timesteps) - 0.8) * 1
        # # penaltyTrue = processedTime * (torch.mean(C1, dim=(1, 2, 3), keepdim=False))
        #
        # minsTrue = torch.min(torch.sum(torch.abs(C), 2, keepdim=True), 3).values
        #
        # penaltyTrue = processedTime * (torch.mean(minsTrue, dim=(1, 2), keepdim=False))
        #
        # # temp=torch.amax(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        # # temp2 = torch.mean(C1, dim=(1, 2, 3), keepdim=False).cpu().detach().numpy()
        # # #return torch.max((torch.max(torch.max(mae_loss, dim=1).values, dim=1).values) ) + 0.0 * torch.sum(penalty)
        # # test=mae_loss.cpu().detach().numpy()
        # # detachedTime=torch.squeeze(time).cpu().detach().numpy()
        #
        # #return torch.mean((torch.mean(torch.mean(mae_loss, dim=1), dim=1))) + 0.001 * torch.sum(penalty)
        # #return torch.mean(torch.pow(mae_loss,2)) + torch.mean(mae_loss) + 0.001 * torch.sum(penalty)
        # #return torch.mean(torch.pow(mae_loss,2)) + 0.0 * torch.mean(penalty)
        # # return torch.mean(mae_loss) + 0.01 * torch.mean(penalty)
        # return torch.mean(mae_loss) + torch.mean(torch.abs(penaltyTrue-penaltyPred))
        # # return torch.mean(mae_loss)

        # timeIndex = torch.where((time.squeeze() >= 0.0) & (time.squeeze() <= 0.51))[0].cpu()
        # for m in range(timeIndex.shape[0]):
        #     plt.plot(y_true.cpu().detach().numpy()[timeIndex[m], :, 0],
        #              y_true.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'b')
        #     plt.plot(y_pred.cpu().detach().numpy()[timeIndex[m], :, 0],
        #              y_pred.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'r')
        # print(time.cpu().detach().numpy()[timeIndex, 0])
        # print("GROUP1")
        # plt.show()
        #
        # timeIndex = torch.where((time.squeeze()>=0.51) & (time.squeeze()<=0.81))[0].cpu()
        # for m in range(timeIndex.shape[0]):
        #     plt.plot(y_true.cpu().detach().numpy()[timeIndex[m], :, 0], y_true.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'b')
        #     plt.plot(y_pred.cpu().detach().numpy()[timeIndex[m], :, 0], y_pred.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'r')
        # print(time.cpu().detach().numpy()[timeIndex, 0])
        # print("GROUP2")
        # plt.show()
        #
        # timeIndex = torch.where((time.squeeze() >= 0.81) & (time.squeeze() <= 1.01))[0].cpu()
        # for m in range(timeIndex.shape[0]):
        #     plt.plot(y_true.cpu().detach().numpy()[timeIndex[m], :, 0],
        #              y_true.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'b')
        #     plt.plot(y_pred.cpu().detach().numpy()[timeIndex[m], :, 0],
        #              y_pred.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'r')
        # print(time.cpu().detach().numpy()[timeIndex, 0])
        # print("GROUP3")
        # print("END OF LOSS")
        # plt.show()

        # timeIndex = torch.where((time.squeeze() >= 0.0) & (time.squeeze() <= 0.01))[0].cpu()
        # for m in range(timeIndex.shape[0]):
        #     plt.plot(y_true.cpu().detach().numpy()[timeIndex[m], :, 0],
        #              y_true.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'b')
        #     plt.plot(y_pred.cpu().detach().numpy()[timeIndex[m], :, 0],
        #              y_pred.cpu().detach().numpy()[timeIndex[m], :, 1],
        #              'r')
        # print(time.cpu().detach().numpy()[timeIndex, 0])
        # plt.show()


        # if branchRanges!=None:
        #     for i in range(len(branchRanges)):
        #         timeIndex = torch.where((time.squeeze() >= branchRanges[i][0]) & (time.squeeze() <= branchRanges[i][1]))[0].cpu()
        #         for m in range(timeIndex.shape[0]):
        #             plt.plot(y_true.cpu().detach().numpy()[timeIndex[m], :, 0],
        #                      y_true.cpu().detach().numpy()[timeIndex[m], :, 1],
        #                      'b')
        #             plt.plot(y_pred.cpu().detach().numpy()[timeIndex[m], :, 0],
        #                      y_pred.cpu().detach().numpy()[timeIndex[m], :, 1],
        #                      'r')
        #         print(time.cpu().detach().numpy()[timeIndex, 0])
        #         print(f"GROUP: {i+1}")
        #         print("END OF LOSS")
        #         plt.show()


        # return torch.mean(torch.mean(mae_loss, dim=(1, 2)) * torch.squeeze(torch.pow(2 - time, 2)))# ORIGINAL***

        # return torch.mean(torch.mean(mae_loss, dim=(1, 2)) * torch.squeeze(torch.pow(2 - time, 2.0)))  # ORIGINAL REVISED***

        return torch.sum(torch.mean(mae_loss, dim=(1, 2)) * torch.squeeze(torch.pow(time + 0.1, 1.0)))
        # return torch.sum(torch.mean(mae_loss, dim=(1, 2)))

# # Custom Activation Layer
# class Activation(nn.Module):
#     def forward(self, x1):
#         return nn.ReLU(x1)  # Element-wise addition

# Custom Add2 Layer
class Add2(nn.Module):
    def forward(self, x1, x2):
        result = x1 + x2
        return result  # Element-wise addition

# Custom Add3 Layer
class Add3(nn.Module):
    def forward(self, x1, x2, x3):
        result = x1 + x2 + x3
        return result  # Element-wise addition

# Custom dist Layer
class PairwiseSubtractionLayer(nn.Module):
    def __init__(self, B,lenForbidden,maxLengthSize):
        super(PairwiseSubtractionLayer, self).__init__()
        self.B = B
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize

    def custom_activation(self,x):
        # tAdj = torch.unsqueeze(torch.unsqueeze(t,dim=2),dim=3)
        a = torch.sigmoid(6 * x)
        b = torch.sigmoid(-6 * x)
        d = torch.sign(x)

        d[d == 0] = 1
        return a * b * d

    def forward(self, A):
        # Reshape for broadcasting
        A_expanded = torch.unsqueeze(A, 3)  # Shape (batch, 2, 1)
        B_expanded = torch.unsqueeze(self.B, 0)
        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        # C1 = torch.tanh(C)
        # C1 = custom_activation(C)

        finIndex = torch.argmin(torch.max(torch.abs(C), dim=2, keepdim=True).values, dim=3)

        finIndex2 = finIndex.repeat(repeats=(1, 1, 2))

        valsMin = torch.gather(C, dim=3, index=finIndex2.unsqueeze(dim=3))

        # finFin = self.custom_activation(valsMin.squeeze())
        finFin = valsMin.squeeze(dim=3)
        return finFin

# Custom dist Layer
class PairwiseSubtractionSumLayer(nn.Module):
    def __init__(self, B,lenForbidden,maxLengthSize):
        super(PairwiseSubtractionSumLayer, self).__init__()
        self.B = B
        self.lenForbidden = lenForbidden
        self.maxLengthSize = maxLengthSize

    def custom_activation(self,x):
        # tAdj = torch.unsqueeze(torch.unsqueeze(t,dim=2),dim=3)
        a = torch.sigmoid(6*x)
        b = torch.sigmoid(-6*x)
        # return a * b - torch.nn.ELU()(tAdj)
        # return a * b - (tAdj * x)
        result = a * b
        return result

    def forward(self, A):
        # Reshape for broadcasting
        A_expanded = torch.unsqueeze(A, 3)  # Shape (batch, 2, 1)
        B_expanded = torch.unsqueeze(self.B, 0)

        C = A_expanded - B_expanded  # Shape (batch, 2, 5)

        C1 = self.custom_activation(C)

        # return torch.min(C1, dim=3).values

        splittedChannels = torch.split(C1, [1, 1], dim=2)

        min_channels = torch.minimum(splittedChannels[0], splittedChannels[1])
        max_pool_2d = torch.nn.MaxPool2d((1, self.lenForbidden), stride=1)

        poolValue2 = max_pool_2d(min_channels)
        poolValue3 = torch.squeeze(poolValue2, dim=[2, 3])
        poolValue4 = torch.sum(poolValue3, dim=1, keepdim=True)

        return poolValue4

# Custom Multiply Layer
class Multiply(nn.Module):
    # def __init__(self):
    #     self.name="Multiply"
    def forward(self, x1, x2):
        result = x1 * x2
        return result  # Element-wise multiplication

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, avgMaxDist_i=1, avgMinDist_i=0, forbiddenSerialMap=None, lenForbidden=10, maxLengthSize = 10, temporalFeatureSize=2, convOffset=0):
        super(SimpleNN, self).__init__()
        self.size = 200
        self.numBranches = 12
        self.branchWidthForward = 0.0
        self.branchWidthBackward = 4.0
        self.branches = nn.ModuleList()
        self.branchRanges = []
        # self.branchRangesTrain = []
        self.avgMaxDist = avgMaxDist_i
        self.avgMinDist = avgMinDist_i
        self.maxLengthSize = maxLengthSize
        self.temporalFeatureSize = temporalFeatureSize
        self.allCustomFcnsGenerative = []
        self.allCustomFcnsTraining = []
        eps=0.0001
        for m in range(self.numBranches):
            # self.branchRanges.append([(0+m/self.numBranches),((m+1)/self.numBranches)])
            # self.branchRanges.append(
            #     [max(0.0, ((m) / self.numBranches) - (1 / self.numBranches) * self.branchWidth),
            #      min(1.0, ((m + 1) / self.numBranches) + (1 / self.numBranches) * self.branchWidth)])
            self.branchRanges.append(
                [max(0.0, ((m) / self.numBranches) - (1 / self.numBranches) * self.branchWidthBackward),
                 min(1.0, ((m + 1) / self.numBranches) + (1 / self.numBranches) * self.branchWidthForward)])
            # self.branchRangesTrain.append([(0 + m / self.numBranches), ((m + 1) / self.numBranches)])
            if m<-1:
                branchTemp = Branch(self.size, gateIndex=1, maxLengthSize=self.maxLengthSize,
                                    temporalFeatureSize=self.temporalFeatureSize,
                                    forbiddenSerialMap=forbiddenSerialMap, lenForbidden=lenForbidden)
            else:
                branchTemp = Branch(self.size, gateIndex=3, maxLengthSize=self.maxLengthSize,
                                    temporalFeatureSize=self.temporalFeatureSize,
                                    forbiddenSerialMap=forbiddenSerialMap, lenForbidden=lenForbidden)

            self.branches.append(branchTemp)
            # self.allCustomFcnsTraining.append(CustomStepFcnTwoSided(
            #     max(0.0,((m)/self.numBranches)-(1/self.numBranches)*self.branchWidth),
            #     min(1.0,((m+1)/self.numBranches)+(1/self.numBranches)*self.branchWidth)))
            # self.allCustomFcnsTraining.append(CustomStepFcnTwoSided(
            #     max(0.0, ((m) / self.numBranches) - (1 / self.numBranches) * self.branchWidth),
            #     min(1.0, ((m + 1) / self.numBranches) + (1 / self.numBranches) * self.branchWidth)))
            self.allCustomFcnsTraining.append(CustomStepFcnTwoSided(
                max(0.0, ((m) / self.numBranches) - (1 / self.numBranches) * self.branchWidthBackward),
                min(1.0, ((m + 1) / self.numBranches) + (1 / self.numBranches) * self.branchWidthForward)))
            if m==0:
                self.allCustomFcnsGenerative.append(CustomStepFcnTwoSided(
                    max(0.0, ((m) / self.numBranches)),
                    min(1.0, ((m + 1) / self.numBranches))))
            elif m==self.numBranches-1:
                self.allCustomFcnsGenerative.append(CustomStepFcnTwoSided(
                    max(0.0, ((m) / self.numBranches) + eps),
                    min(1.0, ((m + 1) / self.numBranches))))
            else:
                self.allCustomFcnsGenerative.append(CustomStepFcnTwoSided(
                    max(0.0, ((m) / self.numBranches) + eps),
                    min(1.0, ((m + 1) / self.numBranches))))
        # self.branches=nn.ModuleList([Branch(self.size, gateIndex=1, maxLengthSize=self.maxLengthSize,
        #                           temporalFeatureSize=self.temporalFeatureSize,
        #                           forbiddenSerialMap=forbiddenSerialMap, lenForbidden=lenForbidden)
        #                           for _ in range(self.numBranches)])

        self.defaultGate = None
        self.multTrajGate = Multiply()

        # self.diffTimeThresh1 = 0.5
        # self.LHS_diff1 = CustomStepFcn(self.diffTimeThresh1)
        # self.diffTimeThresh2 = 0.7
        # self.LHS_diff2 = CustomStepFcn(self.diffTimeThresh2)

        # self.branch1 = Branch(self.size, gateIndex=1,maxLengthSize=self.maxLengthSize,temporalFeatureSize=self.temporalFeatureSize,
        #                       forbiddenSerialMap=forbiddenSerialMap, lenForbidden=lenForbidden)
        # self.branch2 = Branch(self.size, gateIndex=3,maxLengthSize=self.maxLengthSize,temporalFeatureSize=self.temporalFeatureSize,
        #                       forbiddenSerialMap=forbiddenSerialMap, lenForbidden=lenForbidden)
        # self.branch3 = Branch(self.size, gateIndex=1,maxLengthSize=self.maxLengthSize,temporalFeatureSize=self.temporalFeatureSize,
        #                       forbiddenSerialMap=forbiddenSerialMap, lenForbidden=lenForbidden)

        # self.convLast_1 = nn.Conv1d(2, 2, 7,padding=(7-1)//2,
        #                             bias=False, groups=2, padding_mode='replicate')

    def forward(self, traj, time, condLat, condLon, trainBranch=-1, isTraining=False):
        # timeValues1 = self.LHS_diff1(time).squeeze(1)
        # lHs_diff_time1 = timeValues1.unsqueeze(dim=1).unsqueeze(dim=2)
        # timeValues2 = self.LHS_diff2(time).squeeze(1)
        # lHs_diff_time2 = timeValues2.unsqueeze(dim=1).unsqueeze(dim=2)

        # gate1 = lHs_diff_time1
        # gate2 = lHs_diff_time2 * (1 - lHs_diff_time1)
        # gate3 = (self.defaultGate - (lHs_diff_time2)) * (1 - lHs_diff_time1)

        if trainBranch==-1:
            result = 0
            # if time.device.type == "cuda":
            #     cumulativeGate = torch.zeros(traj.shape[0]).cuda()
            # else:
            #     cumulativeGate = torch.zeros(traj.shape[0])

            isNonZeroFound=False
            gateIndexNonZero=-1
            for m in range(self.numBranches):
                if isTraining==True:
                    timeValues1 = self.allCustomFcnsTraining[m](time).squeeze(1)
                else:
                    timeValues1 = self.allCustomFcnsGenerative[m](time).squeeze(1)
                gate1 = (timeValues1).unsqueeze(dim=1).unsqueeze(dim=2)
                finalPathC5 = self.branches[m](gate1, traj, time, condLat, condLon)
                finalPathC5 = self.multTrajGate(gate1, finalPathC5)
                if gate1.sum()>0:
                    if isNonZeroFound==False:
                        isNonZeroFound=True
                        gateIndexNonZero=m
                    else:
                        print(f"TWO GATES ACTIVE! prev gate: {gateIndexNonZero} current gate: {m}")
                        # print(f"GATE: {m}, SUM VALUES: {gate1.sum()}")

                result = result + finalPathC5
            if isNonZeroFound==False:
                print("NO GATE ACTIVATED!!!")

            # result_1 = result.permute(0, 2, 1)
            # result_2 = self.convLast_1(result_1)
            # result_3 = result_2.permute(0, 2, 1)

            return result
        else:
            # result = 0

            # if time.device.type == "cuda":
            #     cumulativeGate = torch.zeros(traj.shape[0]).cuda()
            # else:
            #     cumulativeGate = torch.zeros(traj.shape[0])

            if isTraining == True:
                timeValues1 = self.allCustomFcnsTraining[trainBranch](time).squeeze(1)
            else:
                timeValues1 = self.allCustomFcnsGenerative[trainBranch](time).squeeze(1)
            gate1 = (timeValues1).unsqueeze(dim=1).unsqueeze(dim=2)
            finalPathC5 = self.branches[trainBranch](gate1, traj, time, condLat, condLon)
            finalPathC5 = self.multTrajGate(gate1, finalPathC5)
            # result = result + finalPathC5
            if timeValues1.min()==0:
                print("A TRAJECTORY IS OUTSIDE THE RANGE OF GATE.")

            # result_1 = finalPathC5.permute(0, 2, 1)
            # result_2 = self.convLast_1(result_1)
            # result_3 = result_2.permute(0, 2, 1)
            # return result_3

            # # return result
            return finalPathC5


    def setDefaultGate(self,bs,isRunOnCPU):
        self.defaultGate = torch.ones(bs, device=torch.device('cuda')).unsqueeze(
            dim=1).unsqueeze(
            dim=2)
        if isRunOnCPU == False:
            self.defaultGate = self.defaultGate.cuda()

class Branch(nn.Module):
    def __init__(self, size, gateIndex=-1, avgMaxDist_i=1, avgMinDist_i=0, forbiddenSerialMap=None, lenForbidden=10, maxLengthSize = 10, temporalFeatureSize=2, convOffset=0):
        super(Branch, self).__init__()
        self.size = size
        self.multTrajGate = Multiply()
        self.maxLengthSize = maxLengthSize

        self.multPrePsl = Multiply()

        self.timeForbMixedLinear = nn.Linear(2, self.size)
        self.multTrajCondForbTime = Multiply()

        psl = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        pslSum = PairwiseSubtractionSumLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        self.psl = psl
        self.pslSum = pslSum

        self.timeInput = nn.Linear(1 + 2, maxLengthSize * self.size)

        self.trajInput = nn.Linear(temporalFeatureSize, self.size)
        self.forbDense1 = nn.Linear(2, self.size)
        self.pslDense1 = nn.Linear(1 + 2, maxLengthSize * 10)
        self.trajInputDirect1 = nn.Linear(temporalFeatureSize, self.size)
        self.allInfoDense_1 = nn.Linear(self.size * 3, self.size)
        if gateIndex==3:
            self.ConvDirect1 = ConvLayersDetailed(maxLengthSize, self.size, 1, temporalFeatureSize, convOffset=convOffset)
            self.ConvDists1 = ConvLayersDetailed(maxLengthSize, self.size, 0, temporalFeatureSize, convOffset=convOffset)

            self.denseAfterCat4Traj_1 = nn.Linear(temporalFeatureSize * 6, self.size)
            self.denseAfterCat4Forb_1 = nn.Linear(temporalFeatureSize * 6, self.size)
        else:
            self.ConvDirect1 = ConvLayers(maxLengthSize, self.size, 1, temporalFeatureSize, convOffset=convOffset)
            self.ConvDists1 = ConvLayers(maxLengthSize, self.size, 0, temporalFeatureSize, convOffset=convOffset)

            self.denseAfterCat4Traj_1 = nn.Linear(temporalFeatureSize * 3, self.size)
            self.denseAfterCat4Forb_1 = nn.Linear(temporalFeatureSize * 3, self.size)

        self.denseAfterCat5_1 = nn.Linear(self.size * 5, self.size)
        self.lastAdd = Add3()
        self.lastDense_1 = nn.Linear(self.size + 10, 128)
        self.lastDenseAlt_1 = nn.Linear(128, 2)
        self.convLast_1 = nn.Conv1d(128, 2, 1)
        # self.convLast_2 = nn.Conv1d(2, 2, 7,padding=(7-1)//2,
        #                             bias=False, groups=2, padding_mode='replicate')

    def forward(self, gate1, traj, time, condLat, condLon):
        plsPath = self.psl(traj)
        plsPathSum = self.pslSum(traj)

        plsPath1 = self.multPrePsl(plsPathSum, time)
        mixedPlsPathCond = torch.cat((plsPath1, condLat, condLon), 1)

        condLatLon = torch.cat((torch.unsqueeze(condLat, 1), torch.unsqueeze(condLon, 1)), dim=2)

        mixedTimeCond = torch.cat((time, condLat, condLon), 1)
        timePath = self.timeInput(mixedTimeCond)
        timePath2 = torch.reshape(timePath, (-1, self.maxLengthSize, self.size))

        forbDensed1 = self.forbDense1(plsPath)
        plsPathD1 = self.pslDense1(mixedPlsPathCond)
        plsPathDRS1 = torch.reshape(plsPathD1, (-1, self.maxLengthSize, (int)(10)))
        plsPathDRS1 = self.multTrajGate(gate1, plsPathDRS1)
        trajBranch1 = self.multTrajGate(gate1, traj)
        mixedTrajCond1 = torch.cat((condLatLon, trajBranch1), 1)
        mixedTrajCond1 = self.multTrajGate(gate1, mixedTrajCond1)
        trajPath1 = self.trajInput(mixedTrajCond1)
        trajPathDirect1 = self.trajInputDirect1(traj)
        trajPathDirect1 = self.multTrajGate(gate1, trajPathDirect1)
        in_traj_b1 = self.multTrajGate(gate1, trajPath1)
        in_pls_b1 = self.multTrajGate(gate1, forbDensed1)
        convedTrajs1 = self.ConvDirect1(in_traj_b1, gate1)
        convedDists1 = self.ConvDists1(in_pls_b1, gate1)
        convedTrajs1 = self.multTrajGate(gate1, convedTrajs1)
        convedDists1 = self.multTrajGate(gate1, convedDists1)
        mergePathTraj1 = self.denseAfterCat4Traj_1(convedTrajs1)
        mergePathTraj1 = self.multTrajGate(gate1, mergePathTraj1)
        mergePathForb1 = self.denseAfterCat4Forb_1(convedDists1)
        mergePathForb1 = self.multTrajGate(gate1, mergePathForb1)
        allInfo1 = torch.cat((mergePathTraj1, mergePathForb1, timePath2), dim=2)
        allInfo1 = self.multTrajGate(gate1, allInfo1)
        allInfoDrnsed_1 = self.allInfoDense_1(allInfo1)
        allInfoDrnsed_1 = self.multTrajGate(gate1, allInfoDrnsed_1)
        finalPath_1 = torch.cat((timePath2, allInfoDrnsed_1, trajPathDirect1, mergePathTraj1, mergePathForb1), 2)
        finalPath_1 = self.multTrajGate(gate1, finalPath_1)
        finalPath2_1 = self.denseAfterCat5_1(finalPath_1)
        finalPath2_1 = self.multTrajGate(gate1, finalPath2_1)

        finalPathC_1 = torch.cat((finalPath2_1, plsPathDRS1), 2)
        finalPathC2_1 = self.lastDense_1(finalPathC_1)
        finalPathC2_1 = self.multTrajGate(gate1, finalPathC2_1)

        finalPathC3_1 = finalPathC2_1.permute(0, 2, 1)
        finalPathC4_1 = self.convLast_1(finalPathC3_1)
        # finalPathC5_1 = self.convLast_2(finalPathC4_1)
        finalPathC6_1 = self.multTrajGate(gate1, finalPathC4_1)
        finalPathC7_1 = finalPathC6_1.permute(0, 2, 1)

        return finalPathC7_1

def generate_ts(timesteps, num, learningScheduleTime, isChangeWeights, isAdvancedWeighting=True):
    orig = np.random.randint(0, timesteps, size=num)
    if isAdvancedWeighting==True:
        actualLearningScheduleTime=learningScheduleTime
    else:
        actualLearningScheduleTime=0
    if isChangeWeights==False:
        actualLearningScheduleTime=1
    numInternal = 100
    noisyOnes = np.random.randint(0, (int)(math.floor(timesteps * 0.8)),
                                  size=(int)(numInternal / (0.9 + actualLearningScheduleTime * 10.0)))
    lastOnes = np.random.randint((int)(math.floor(timesteps * 0.65)), timesteps,
                                 size=(int)(numInternal / (8.0 - actualLearningScheduleTime * 6.0)))
    lastOnes2 = np.random.randint((int)(math.floor(timesteps * 0.7)), timesteps,
                                  size=(int)(numInternal / (7.0 - actualLearningScheduleTime * 6.0)))
    lastOnes3 = np.random.randint((int)(math.floor(timesteps * 0.92)), timesteps,
                                  size=(int)(numInternal / (6.2 - actualLearningScheduleTime * 6.0)))

    finalTs = np.concat((orig,noisyOnes,lastOnes,lastOnes2,lastOnes3))
    finalTs2 = np.random.choice(finalTs,num)
    return finalTs2

def generate_ts_interval(timesteps, num, minVal, maxVal):
    orig = np.random.randint(max(0,(int)(math.ceil(timesteps * minVal))-0), min(timesteps,(int)(math.floor(timesteps * maxVal))+0), size=num)
    return orig

def forward_noise_notNormalized(meanVals, varVals, timesteps, x, t, learningScheduleTime, isChangeWeights, isVizualize=True, isAdvancedExponent=True, isNoiseSeparate=False):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    time_bar = time_bar + (np.random.uniform(low=-1.0,high=1.0,size=timesteps + 1))*0.01 # ADD SOME RANDOMNESS TO DIFF TIME
    minV = time_bar.min()
    maxV = time_bar.max()
    time_bar = (time_bar-minV)/(maxV-minV)
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    if isNoiseSeparate==False:
        noise = np.random.normal(loc=meanVals, scale=varVals, size=x.shape)  # noise mask
    else:
        noise_a = np.random.normal(loc=meanVals,scale=varVals,size=x.shape)  # noise mask
        noise_b = np.random.normal(loc=meanVals, scale=varVals, size=x.shape)  # noise mask

    # noise = np.random.normal(loc=0.5,scale=0.33,size=x.shape)  # noise mask
    # noise = np.random.uniform(low=0,high=1,size=x.shape)

    # for i in range(1, noise.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([noise[i, h - 1, 0], noise[i, h, 0]], [noise[i, h - 1, 1], noise[i, h, 1]], marker='', zorder=2, alpha=0.5,color='c')


    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    # img_a = x * (1 - a) + noise * a
    # img_b = x * (1 - b) + noise * b

    if x.device.type=="cuda":
        xCPU = x.cpu()
    else:
        xCPU = x

    if isAdvancedExponent == True:
        if isChangeWeights == False:
            learningScheduleTime = 1
        # img_a = xCPU * (1 - np.pow(a, 1.5 + learningScheduleTime * 4.0)) + noise * np.pow(a, 1.5 + learningScheduleTime * 4.0)
        # img_b = xCPU * (1 - np.pow(b, 1.5 + learningScheduleTime * 4.0)) + noise * np.pow(b, 1.5 + learningScheduleTime * 4.0)

        # powAddition = 0.0
        # if isNoiseSeparate == False:
        #     img_a = xCPU * (1 - np.pow(a, 1.0 + learningScheduleTime * powAddition)) + noise * np.pow(a,
        #                                   1.0 + learningScheduleTime * powAddition)
        #     img_b = xCPU * (1 - np.pow(b, 1.0 + learningScheduleTime * powAddition)) + noise * np.pow(b,
        #                                   1.0 + learningScheduleTime * powAddition)
        # else:
        #     img_a = xCPU * (1 - np.pow(a, 1.0 + learningScheduleTime * powAddition)) + noise_a * np.pow(a,
        #                                   1.0 + learningScheduleTime * powAddition)
        #     img_b = xCPU * (1 - np.pow(b, 1.0 + learningScheduleTime * powAddition)) + noise_b * np.pow(b,
        #                                   1.0 + learningScheduleTime * powAddition)

        # print((1 - a).squeeze())
        # print((1 - np.pow(a, 1.0 + learningScheduleTime * powAddition)).squeeze())
        # print("!!!")

        powAddition = 1.0
        # fcn_val_a = np.pow(a, 3) / (np.pow(a, 5) + (1 - a) * np.exp((1.0 + learningScheduleTime * powAddition) * a))
        # fcn_val_b = np.pow(b, 3) / (np.pow(b, 5) + (1 - b) * np.exp((1.0 + learningScheduleTime * powAddition) * b))
        # fcn_val_a = ((np.pow(a, 3) * (2 - np.pow(a, 2))) / (np.pow(a, 5) + (1 - a) * np.exp((1.4 + learningScheduleTime * powAddition) * a))) + np.pow(a, 2) * (1 - a)
        # fcn_val_b = ((np.pow(b, 3) * (2 - np.pow(b, 2))) / (np.pow(b, 5) + (1 - b) * np.exp((1.4 + learningScheduleTime * powAddition) * b))) + np.pow(b, 2) * (1 - b)
        # fcn_val_a = (1 - fcn_val_a)
        # fcn_val_b = (1 - fcn_val_b)

        fcn_val_a = (((np.pow(a, 10) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                                                              np.exp((1.8 + learningScheduleTime * powAddition) * a))) +
                     np.pow(a, 4.2) * (1 - a)) - 0.004 * (a - 0.5) * a * (1 - a)
        fcn_val_b = (((np.pow(b, 10) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                                                              np.exp((1.8 + learningScheduleTime * powAddition) * b))) +
                     np.pow(b, 4.2) * (1 - b)) - 0.004 * (b - 0.5) * b * (1 - b)
        if isNoiseSeparate == False:
            img_a = xCPU * (1 - fcn_val_a) + noise * fcn_val_a
            img_b = xCPU * (1 - fcn_val_b) + noise * fcn_val_b
        else:
            img_a = xCPU * (1 - fcn_val_a) + noise_a * fcn_val_a
            img_b = xCPU * (1 - fcn_val_b) + noise_b * fcn_val_b

    else:
        img_a = xCPU * (1 - np.pow(a, 4)) + noise * np.pow(a, 4)
        img_b = xCPU * (1 - np.pow(b, 4)) + noise * np.pow(b, 4)


    # img_a = x + noise * np.pow(a,2)
    # img_b = x + noise * np.pow(b,2)

    # idx=237
    # for h in range(1, 12):
    #     plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]], marker='', zorder=2, alpha=0.5,color='r')
    # for h in range(1, 12):
    #     plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]], marker='', zorder=2, alpha=0.5,color='g')
    # plt.show()

    # for idx in range(x.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
    #                  zorder=2, alpha=0.5, color='r')
    # # # plt.show()
    # for idx in range(x.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]], marker='',
    #                  zorder=2, alpha=0.5, color='g')
    # plt.show()

    if isVizualize == True:
        if oneTimeVisGenAB == True:
            cmap_name = 'jet'  # Example: Use the 'jet' colormap
            cmap = cm.get_cmap(cmap_name, timesteps)
            for idx in range(x.shape[0]):
                # if t[idx]<11 or t[idx]>13:#DEBUG
                #     continue#DEBUG
                color = cmap(t[idx])
                for h in range(1, x.shape[1]):
                    plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, x.shape[1]):
                    plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            # oneTimeVisGenAB = False

    isDEBUG = False
    if isDEBUG == True:
        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, (int)(time_bar.shape[0]))
        for y in range(xCPU.shape[0]):
            xDebug = xCPU[y, :, :]  # 8 10
            # for h in range(1, img_b.shape[1]):
            #     plt.plot([xDebug[h - 1, 0], xDebug[h, 0]], [xDebug[h - 1, 1], xDebug[h, 1]],
            #                  marker='', color='b')
            # plt.show()
            xDebugTimed = np.zeros((time_bar.shape[0], xCPU.shape[1], xCPU.shape[2]))
            debugTimed = np.expand_dims(time_bar, axis=1)
            # noise = np.random.normal(loc=meanVals, scale=varVals, size=xDebug.shape)

            # initialAngle = np.random.uniform(0.0, np.pi, size=(1, 1))
            # angles = np.random.uniform(0.01 * np.pi, np.pi * 0.2, size=(xDebugTimed.shape[1] - 1, 1))
            # anglesCumSum = np.cumsum(angles, axis=0) + initialAngle
            # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=xDebug.shape) + varVals * 2.0
            # # x0 = (radii[:, 0, 0] + meanVals[0]) * np.cos(initialAngle).squeeze()
            # # y0 = (radii[:, 0, 1] + meanVals[1]) * np.cos(initialAngle).squeeze()
            # x0 = ((radii[0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
            # x0 = np.array([[x0]])
            # y0 = ((radii[0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
            # y0 = np.array([[y0]])
            # xVals = (np.expand_dims(radii[1:, 0], axis=1) * np.cos(anglesCumSum)) + meanVals[0]
            # yVals = (np.expand_dims(radii[1:, 0], axis=1) * np.sin(anglesCumSum)) + meanVals[1]
            # allXs = np.concat((x0, xVals), axis=0)
            # allYs = np.concat((y0, yVals), axis=0)
            # noise = np.concat((allXs, allYs), axis=1)
            noise = np.random.normal(loc=meanVals, scale=varVals, size=(x.shape[1],x.shape[2]))
            plt.plot(xDebug[:, 0], xDebug[:, 1])
            # plt.plot(noise[:,0],noise[:,1])
            plt.show()

            powAddition = 2.0
            # powAddition = 0.0
            for tt in range(time_bar.shape[0]):
                xDebugTimed[tt, :, :] = (
                        xDebug * (1 - np.pow(debugTimed[tt], 2.0 + learningScheduleTime * powAddition)) +
                        noise * np.pow(debugTimed[tt], 2.0 + learningScheduleTime * powAddition))

                # fcn_val = ((np.pow(debugTimed[tt], 3) * (2 - np.pow(debugTimed[tt], 2))) / (
                #             np.pow(debugTimed[tt], 5) + (1 - debugTimed[tt]) * np.exp((2.8 +
                #             learningScheduleTime * powAddition) * debugTimed[tt]))) + np.pow(debugTimed[tt], 0.8) * (1 - debugTimed[tt])
                # xDebugTimed[tt, :, :] = xDebug * (1 - fcn_val) + noise * fcn_val

                # v = np.pow(debugTimed[tt], 3) / (np.pow(debugTimed[tt], 5) + (1 - debugTimed[tt]) * np.exp((2.0 + learningScheduleTime * powAddition) * debugTimed[tt]))
                # xDebugTimed[tt, :, :] = xDebug * (1-v) + noise * v

            for idx in range(xDebugTimed.shape[0]):
                color = cmap(idx)
                # if t[idx]<60:
                #     continue
                for h in range(1, xDebugTimed.shape[1]):
                    plt.plot([xDebugTimed[idx, h - 1, 0], xDebugTimed[idx, h, 0]],
                             [xDebugTimed[idx, h - 1, 1], xDebugTimed[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()

    return img_a, img_b


def forward_noise(timesteps, x, t, learningScheduleTime, isChangeWeights, isVizualize=True, isAdvancedExponent=True):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(loc=0.5,scale=0.5,size=x.shape)  # noise mask

    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))

    if isAdvancedExponent == True:
        if isChangeWeights == False:
            learningScheduleTime = 1
        img_a = x * (1 - np.pow(a, 1.5+learningScheduleTime*3.5)) + noise * np.pow(a, 1.5+learningScheduleTime*3.5)
        img_b = x * (1 - np.pow(b, 1.5+learningScheduleTime*3.5)) + noise * np.pow(b, 1.5+learningScheduleTime*3.5)
    else:
        img_a = x * (1 - np.pow(a, 3.9)) + noise * np.pow(a, 3.9)
        img_b = x * (1 - np.pow(b, 3.9)) + noise * np.pow(b, 3.9)

    if isVizualize==True:
        if oneTimeVisGenAB == True:
            cmap_name = 'jet'  # Example: Use the 'jet' colormap
            cmap = cm.get_cmap(cmap_name, timesteps)
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, 12):
                    plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, 12):
                    plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            # oneTimeVisGenAB = False

    return img_a, img_b

def forward_noise_circular(avgMaxDist, avgMinDist, meanVals, varVals, timesteps, x, t, learningScheduleTime, isChangeWeights, isVizualize=True, isAdvancedExponent=True, isNoiseSeparate=False):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    avgPointDists = torch.diff(x, axis=1).pow(2).sum(dim=2).sqrt().sum(dim=1)
    if avgMaxDist == avgMinDist:
        avgLen = (avgPointDists) / (avgMaxDist)
    else:
        avgLen = (avgPointDists - avgMinDist) / (avgMaxDist - avgMinDist)

    if isNoiseSeparate == False:
        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(x.shape[0], 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(x.shape[0], x.shape[1] - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + varVals * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + meanVals
        # x0 = (radii[:, 0, 0] + meanVals[0]) * np.cos(initialAngle).squeeze()
        # y0 = (radii[:, 0, 1] + meanVals[1]) * np.cos(initialAngle).squeeze()
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVals[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVals[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        noise = np.concat((allXs, allYs), axis=2)
    else:
        # noise = np.random.normal(loc=meanVals,scale=varVals,size=x.shape)  # noise mask
        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(x.shape[0], 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(x.shape[0], x.shape[1] - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + varVals * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + meanVals
        # x0 = (radii[:, 0, 0] + meanVals[0]) * np.cos(initialAngle).squeeze()
        # y0 = (radii[:, 0, 1] + meanVals[1]) * np.cos(initialAngle).squeeze()
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVals[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVals[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        noise_a = np.concat((allXs, allYs), axis=2)

        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(x.shape[0], 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(x.shape[0], x.shape[1] - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + varVals * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + meanVals
        # x0 = (radii[:, 0, 0] + meanVals[0]) * np.cos(initialAngle).squeeze()
        # y0 = (radii[:, 0, 1] + meanVals[1]) * np.cos(initialAngle).squeeze()
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVals[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVals[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        noise_b = np.concat((allXs, allYs), axis=2)



    # noise_b = np.random.normal(loc=meanVals, scale=varVals, size=x.shape)  # noise mask
    # noise = np.random.normal(loc=0.5,scale=0.33,size=x.shape)  # noise mask
    # noise = np.random.uniform(low=0,high=1,size=x.shape)

    # for i in range(1, noise.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([noise[i, h - 1, 0], noise[i, h, 0]], [noise[i, h - 1, 1], noise[i, h, 1]], marker='', zorder=2, alpha=0.5,color='c')


    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    # img_a = x * (1 - a) + noise * a
    # img_b = x * (1 - b) + noise * b

    if type(x).__name__ == "ndarray":
        xCPU = x
    elif type(x).__name__ == "Tensor":
        if x.device.type == "cuda":
            xCPU = x.cpu()
            avgLenCPU = avgLen.cpu()
        else:
            avgLenCPU = avgLen
            xCPU = x

    if isAdvancedExponent == True:
        if isChangeWeights == False:
            learningScheduleTime = 1
        # img_a = xCPU * (1 - np.pow(a, 1.5 + learningScheduleTime * 4.0)) + noise * np.pow(a, 1.5 + learningScheduleTime * 4.0)
        # img_b = xCPU * (1 - np.pow(b, 1.5 + learningScheduleTime * 4.0)) + noise * np.pow(b, 1.5 + learningScheduleTime * 4.0)
        # powAddition = 4.0 + np.expand_dims((avgLenCPU)*0.0,axis=(1,2))

        # powAddition = 2.5
        # if isNoiseSeparate==True:
        #     img_a = xCPU * (1 - np.pow(a, 2 + learningScheduleTime * powAddition)) + noise_a * np.pow(a,
        #                                                                                               2 + learningScheduleTime * powAddition)
        #     img_b = xCPU * (1 - np.pow(b, 2 + learningScheduleTime * powAddition)) + noise_b * np.pow(b,
        #                                                                                               2 + learningScheduleTime * powAddition)
        # else:
        #     img_a = xCPU * (1 - np.pow(a, 2 + learningScheduleTime * powAddition)) + noise * np.pow(a,
        #                                                                                               2 + learningScheduleTime * powAddition)
        #     img_b = xCPU * (1 - np.pow(b, 2 + learningScheduleTime * powAddition)) + noise * np.pow(b,
        #                                                                                               2 + learningScheduleTime * powAddition)


        powAddition = 1.0
        if isNoiseSeparate == True:
            fcn_val_a = (((np.pow(a, 10) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * a))) +
                        np.pow(a, 4.2) * (1 - a)) - 0.00004 * (a - 0.5) * a * (1 - a)
            fcn_val_b = (((np.pow(b, 10) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * b))) +
                        np.pow(b, 4.2) * (1 - b)) - 0.00004 * (b - 0.5) * b * (1 - b)
            # fcn_val_a = np.pow(a, 3) / (np.pow(a, 5) + (1 - a) * np.exp((1.0 + learningScheduleTime * powAddition) * a))
            # fcn_val_b = np.pow(b, 3) / (np.pow(b, 5) + (1 - b) * np.exp((1.0 + learningScheduleTime * powAddition) * b))
            img_a = xCPU * (1 - fcn_val_a) + noise_a * fcn_val_a
            img_b = xCPU * (1 - fcn_val_b) + noise_b * fcn_val_b
        else:
            fcn_val_a = (((np.pow(a, 7) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                        np.exp((2.8 + learningScheduleTime * powAddition) * a))) +
                        np.pow(a, 4.2) * (1 - a)) - 0.00004 * (a - 0.7) * a * (1 - a)
            fcn_val_b = (((np.pow(b, 7) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                        np.exp((2.8 + learningScheduleTime * powAddition) * b))) +
                        np.pow(b, 4.2) * (1 - b)) - 0.00004 * (b - 0.7) * b * (1 - b)
            # fcn_val_a = np.pow(a, 3) / (np.pow(a, 5) + (1 - a) * np.exp((1.0 + learningScheduleTime * powAddition) * a))
            # fcn_val_b = np.pow(b, 3) / (np.pow(b, 5) + (1 - b) * np.exp((1.0 + learningScheduleTime * powAddition) * b))
            img_a = xCPU * (1 - fcn_val_a) + noise * fcn_val_a
            img_b = xCPU * (1 - fcn_val_b) + noise * fcn_val_b

        # # print(learningScheduleTime)
    else:
        img_a = xCPU * (1 - np.pow(a, 7.0)) + noise_a * np.pow(a, 7.0)
        img_b = xCPU * (1 - np.pow(b, 7.0)) + noise_b * np.pow(b, 7.0)

    isDEBUG=True
    if isDEBUG == True:
        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, (int)(time_bar.shape[0]))
        for y in range(xCPU.shape[0]):
            xDebug = xCPU[y, :, :]  # 8 10
            # for h in range(1, img_b.shape[1]):
            #     plt.plot([xDebug[h - 1, 0], xDebug[h, 0]], [xDebug[h - 1, 1], xDebug[h, 1]],
            #                  marker='', color='b')
            # plt.show()
            xDebugTimed = np.zeros((time_bar.shape[0], xCPU.shape[1], xCPU.shape[2]))
            debugTimed = np.expand_dims(time_bar, axis=1)
            # noise = np.random.normal(loc=meanVals, scale=varVals, size=xDebug.shape)

            initialAngle = np.random.uniform(0.0, 2*np.pi, size=(1,1))
            angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(xDebugTimed.shape[1] - 1,1))
            anglesCumSum = np.cumsum(angles, axis=0) + initialAngle
            # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=xDebug.shape) + varVals * 10.0
            radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=xDebug.shape) + meanVals
            # x0 = (radii[:, 0, 0] + meanVals[0]) * np.cos(initialAngle).squeeze()
            # y0 = (radii[:, 0, 1] + meanVals[1]) * np.cos(initialAngle).squeeze()
            x0 = ((radii[0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
            x0 = np.array([[x0]])
            y0 = ((radii[0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
            y0 = np.array([[y0]])
            xVals = (np.expand_dims(radii[1:, 0],axis=1) * np.cos(anglesCumSum)) + meanVals[0]
            yVals = (np.expand_dims(radii[1:, 0],axis=1) * np.sin(anglesCumSum)) + meanVals[1]
            allXs = np.concat((x0, xVals), axis=0)
            allYs = np.concat((y0, yVals), axis=0)
            noise = np.concat((allXs, allYs), axis=1)
            plt.plot(xDebug[:, 0], xDebug[:, 1])
            # plt.plot(noise[:,0],noise[:,1])
            plt.show()

            powAddition = 0.0
            for tt in range(time_bar.shape[0]):
                # xDebugTimed[tt, :, :] = (
                #             xDebug * (1 - np.pow(debugTimed[tt], 0.5 + learningScheduleTime * powAddition)) +
                #             noise * np.pow(debugTimed[tt], 0.5 + learningScheduleTime * powAddition))


                # v = np.pow(debugTimed[tt], 3) / (np.pow(debugTimed[tt], 5) + (1 - debugTimed[tt]) * np.exp((2.0 + learningScheduleTime * powAddition) * debugTimed[tt]))

                v = (((np.pow(debugTimed[tt], 7) * (2 - np.pow(debugTimed[tt], 2))) / (np.pow(debugTimed[tt], 1) +
                    (1 - debugTimed[tt]) * np.exp((2.8 + learningScheduleTime * powAddition) * debugTimed[tt]))) +
                    np.pow(debugTimed[tt], 4.2) * (1 - debugTimed[tt]))-0.003*(debugTimed[tt]-0.7)*debugTimed[tt]*(1-debugTimed[tt])

                xDebugTimed[tt, :, :] = xDebug * (1-v) + noise * v

            for idx in range(xDebugTimed.shape[0]):
                color = cmap(idx)
                # if t[idx]<60:
                #     continue
                for h in range(1, xDebugTimed.shape[1]):
                    plt.plot([xDebugTimed[idx, h - 1, 0], xDebugTimed[idx, h, 0]],
                             [xDebugTimed[idx, h - 1, 1], xDebugTimed[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
        # print("!!!")

    if isVizualize==True:
        if oneTimeVisGenAB == True:
            cmap_name = 'jet'  # Example: Use the 'jet' colormap
            cmap = cm.get_cmap(cmap_name, (int)(timesteps))
            # for idx in range(x.shape[0]):
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                # if t[idx]<60:
                #     continue
                for h in range(1, img_b.shape[1]):
                    plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, img_a.shape[1]):
                    plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            # oneTimeVisGenAB = False

    # if type(x).__name__ == "Tensor":
    #     if x.device.type == "cuda":
    #         img_a = img_a.cuda()
    #         img_b = img_b.cuda()

    # if x.device=="cuda":
    #     img_a = img_a.cuda()
    #     img_b = img_b.cuda()
    return img_a, img_b

def forward_noise_gaussianCircular(avgMaxDist, avgMinDist, meanVals, varVals, timesteps, x, t, learningScheduleTime, isChangeWeights, isVizualize=True, isAdvancedExponent=True, isNoiseSeparate=False):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    avgPointDists = torch.diff(x, axis=1).pow(2).sum(dim=2).sqrt().sum(dim=1)
    if avgMaxDist == avgMinDist:
        avgLen = (avgPointDists) / (avgMaxDist)
    else:
        avgLen = (avgPointDists - avgMinDist) / (avgMaxDist - avgMinDist)

    if isNoiseSeparate==False:
        noise_gauss = np.random.normal(loc=meanVals, scale=varVals, size=x.shape)  # noise mask
    else:
        noise_a_gauss = np.random.normal(loc=meanVals,scale=varVals,size=x.shape)  # noise mask
        noise_b_gauss = np.random.normal(loc=meanVals, scale=varVals, size=x.shape)  # noise mask

    if isNoiseSeparate == False:
        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(x.shape[0], 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(x.shape[0], x.shape[1] - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + varVals * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + meanVals
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVals[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVals[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        noise = np.concat((allXs, allYs), axis=2)
    else:
        # noise = np.random.normal(loc=meanVals,scale=varVals,size=x.shape)  # noise mask
        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(x.shape[0], 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(x.shape[0], x.shape[1] - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + varVals * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + meanVals
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVals[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVals[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        noise_a = np.concat((allXs, allYs), axis=2)

        initialAngle = np.random.uniform(0.0, 2*np.pi, size=(x.shape[0], 1))
        angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(x.shape[0], x.shape[1] - 1))
        anglesCumSum = np.cumsum(angles, axis=1) + initialAngle
        # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + varVals * 10.0
        radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=x.shape) + meanVals
        x0 = ((radii[:, 0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
        y0 = ((radii[:, 0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
        xVals = (radii[:, 1:, 0] * np.cos(anglesCumSum)) + meanVals[0]
        yVals = (radii[:, 1:, 0] * np.sin(anglesCumSum)) + meanVals[1]
        allXs = np.expand_dims(np.concat((np.expand_dims(x0, axis=1), xVals), axis=1), axis=2)
        allYs = np.expand_dims(np.concat((np.expand_dims(y0, axis=1), yVals), axis=1), axis=2)
        noise_b = np.concat((allXs, allYs), axis=2)

    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    # img_a = x * (1 - a) + noise * a
    # img_b = x * (1 - b) + noise * b

    if type(x).__name__ == "ndarray":
        xCPU = x
    elif type(x).__name__ == "Tensor":
        if x.device.type == "cuda":
            xCPU = x.cpu()
            avgLenCPU = avgLen.cpu()
        else:
            avgLenCPU = avgLen
            xCPU = x

    if isAdvancedExponent == True:
        if isChangeWeights == False:
            learningScheduleTime = 1
        # img_a = xCPU * (1 - np.pow(a, 1.5 + learningScheduleTime * 4.0)) + noise * np.pow(a, 1.5 + learningScheduleTime * 4.0)
        # img_b = xCPU * (1 - np.pow(b, 1.5 + learningScheduleTime * 4.0)) + noise * np.pow(b, 1.5 + learningScheduleTime * 4.0)
        powAddition = 0.0 + (1.0-np.expand_dims((avgLenCPU)*1.0,axis=(1,2)))

        # powAddition = 2.5
        # if isNoiseSeparate==True:
        #     img_a = xCPU * (1 - np.pow(a, 2 + learningScheduleTime * powAddition)) + noise_a * np.pow(a,
        #                                                                                               2 + learningScheduleTime * powAddition)
        #     img_b = xCPU * (1 - np.pow(b, 2 + learningScheduleTime * powAddition)) + noise_b * np.pow(b,
        #                                                                                               2 + learningScheduleTime * powAddition)
        # else:
        #     img_a = xCPU * (1 - np.pow(a, 2 + learningScheduleTime * powAddition)) + noise * np.pow(a,
        #                                                                                               2 + learningScheduleTime * powAddition)
        #     img_b = xCPU * (1 - np.pow(b, 2 + learningScheduleTime * powAddition)) + noise * np.pow(b,
        #                                                                                               2 + learningScheduleTime * powAddition)


        # powAddition = 1.0
        if isNoiseSeparate == True:
            fcn_val_a = (((np.pow(a, 10) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * a))) +
                        np.pow(a, 3.2+powAddition*20) * (1 - a)) - 0.00004 * (a - 0.5) * a * (1 - a)
            fcn_val_b = (((np.pow(b, 10) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * b))) +
                        np.pow(b, 3.2+powAddition*20) * (1 - b)) - 0.00004 * (b - 0.5) * b * (1 - b)
            fcn_val_a_cir = (((np.pow(a, 3) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * a))) +
                        np.pow(a, 3.2+powAddition*20) * (1 - a)) - 0.00004 * (a - 0.5) * a * (1 - a)
            fcn_val_b_cir = (((np.pow(b, 3) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * b))) +
                        np.pow(b, 3.2+powAddition*20) * (1 - b)) - 0.00004 * (b - 0.5) * b * (1 - b)
            # fcn_val_a = np.pow(a, 3) / (np.pow(a, 5) + (1 - a) * np.exp((1.0 + learningScheduleTime * powAddition) * a))
            # fcn_val_b = np.pow(b, 3) / (np.pow(b, 5) + (1 - b) * np.exp((1.0 + learningScheduleTime * powAddition) * b))
            # fcn_val_a_p = np.pow(fcn_val_a, 3.0)
            # fcn_val_b_p = np.pow(fcn_val_b, 3.0)
            img_a = xCPU * (1 - fcn_val_a) + ((noise_a_gauss)) * fcn_val_a + fcn_val_a_cir*(1-fcn_val_a_cir)*noise_a
            img_b = xCPU * (1 - fcn_val_b) + ((noise_b_gauss)) * fcn_val_b + fcn_val_b_cir*(1-fcn_val_b_cir)*noise_b
        else:
            fcn_val_a = (((np.pow(a, 10+powAddition*10) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * a))) +
                        np.pow(a, 3.2+powAddition*10) * (1 - a)) - 0.00004 * (a - 0.7) * a * (1 - a)
            fcn_val_b = (((np.pow(b, 10+powAddition*10) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                        np.exp((2.8 + learningScheduleTime * powAddition) * b))) +
                        np.pow(b, 3.2+powAddition*10) * (1 - b)) - 0.00004 * (b - 0.7) * b * (1 - b)
            fcn_val_a_cir = (((np.pow(a, 3+powAddition*10) * (2 - np.pow(a, 2))) / (np.pow(a, 1) + (1 - a) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * a))) +
                        np.pow(a, 3.2+powAddition*10) * (1 - a)) - 0.00004 * (a - 0.7) * a * (1 - a)
            fcn_val_b_cir = (((np.pow(b, 3+powAddition*10) * (2 - np.pow(b, 2))) / (np.pow(b, 1) + (1 - b) *
                        np.exp((1.8 + learningScheduleTime * powAddition) * b))) +
                        np.pow(b, 3.2+powAddition*10) * (1 - b)) - 0.00004 * (b - 0.7) * b * (1 - b)
            fcn_val_a = np.maximum(0, fcn_val_a)
            fcn_val_b = np.maximum(0, fcn_val_b)
            fcn_val_a_cir = np.maximum(0, fcn_val_a_cir)
            fcn_val_b_cir = np.maximum(0, fcn_val_b_cir)
            # fcn_val_a = np.pow(a, 3) / (np.pow(a, 5) + (1 - a) * np.exp((1.0 + learningScheduleTime * powAddition) * a))
            # fcn_val_b = np.pow(b, 3) / (np.pow(b, 5) + (1 - b) * np.exp((1.0 + learningScheduleTime * powAddition) * b))
            # fcn_val_a_p = np.pow(fcn_val_a, 3.0)
            # fcn_val_b_p = np.pow(fcn_val_b, 3.0)
            img_a = xCPU * (1 - fcn_val_a) + ((noise_gauss)) * fcn_val_a + fcn_val_a_cir*(1-fcn_val_a_cir)*noise
            img_b = xCPU * (1 - fcn_val_b) + ((noise_gauss)) * fcn_val_b + fcn_val_b_cir*(1-fcn_val_b_cir)*noise

        # # print(learningScheduleTime)
    else:
        img_a = xCPU * (1 - np.pow(a, 7.0)) + noise_a * np.pow(a, 7.0)
        img_b = xCPU * (1 - np.pow(b, 7.0)) + noise_b * np.pow(b, 7.0)

    isDEBUG=False
    if isDEBUG == True:
        cmap_name = 'jet'  # Example: Use the 'jet' colormap
        cmap = cm.get_cmap(cmap_name, (int)(time_bar.shape[0]))
        for y in range(xCPU.shape[0]):
            xDebug = xCPU[y, :, :]  # 8 10
            # for h in range(1, img_b.shape[1]):
            #     plt.plot([xDebug[h - 1, 0], xDebug[h, 0]], [xDebug[h - 1, 1], xDebug[h, 1]],
            #                  marker='', color='b')
            # plt.show()
            xDebugTimed = np.zeros((time_bar.shape[0], xCPU.shape[1], xCPU.shape[2]))
            debugTimed = np.expand_dims(time_bar, axis=1)
            noise_gauss = np.random.normal(loc=meanVals, scale=varVals, size=xDebug.shape)

            initialAngle = np.random.uniform(0.0, 2*np.pi, size=(1,1))
            angles = np.random.uniform(0.01 * np.pi, np.pi * 0.05, size=(xDebugTimed.shape[1] - 1,1))
            anglesCumSum = np.cumsum(angles, axis=0) + initialAngle
            # radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=xDebug.shape) + varVals * 10.0
            radii = np.random.normal(loc=(0, 0), scale=varVals / 12.0, size=xDebug.shape) + meanVals
            # x0 = (radii[:, 0, 0] + meanVals[0]) * np.cos(initialAngle).squeeze()
            # y0 = (radii[:, 0, 1] + meanVals[1]) * np.cos(initialAngle).squeeze()
            x0 = ((radii[0, 0]) * np.cos(initialAngle).squeeze()) + meanVals[0]
            x0 = np.array([[x0]])
            y0 = ((radii[0, 1]) * np.sin(initialAngle).squeeze()) + meanVals[1]
            y0 = np.array([[y0]])
            xVals = (np.expand_dims(radii[1:, 0],axis=1) * np.cos(anglesCumSum)) + meanVals[0]
            yVals = (np.expand_dims(radii[1:, 0],axis=1) * np.sin(anglesCumSum)) + meanVals[1]
            allXs = np.concat((x0, xVals), axis=0)
            allYs = np.concat((y0, yVals), axis=0)
            noise_cir = np.concat((allXs, allYs), axis=1)

            plt.plot(xDebug[:, 0], xDebug[:, 1])
            # plt.plot(noise[:,0],noise[:,1])
            plt.show()

            powAddition = 0.0
            for tt in range(time_bar.shape[0]):
                # xDebugTimed[tt, :, :] = (
                #             xDebug * (1 - np.pow(debugTimed[tt], 0.5 + learningScheduleTime * powAddition)) +
                #             noise * np.pow(debugTimed[tt], 0.5 + learningScheduleTime * powAddition))


                # v = np.pow(debugTimed[tt], 3) / (np.pow(debugTimed[tt], 5) + (1 - debugTimed[tt]) * np.exp((2.0 + learningScheduleTime * powAddition) * debugTimed[tt]))

                v = (((np.pow(debugTimed[tt], 10) * (2 - np.pow(debugTimed[tt], 2))) / (np.pow(debugTimed[tt], 1) +
                    (1 - debugTimed[tt]) * np.exp((2.8 + learningScheduleTime * powAddition) * debugTimed[tt]))) +
                    np.pow(debugTimed[tt], 4.2) * (1 - debugTimed[tt]))-0.003*(debugTimed[tt]-0.7)*debugTimed[tt]*(1-debugTimed[tt])
                v_cir = (((np.pow(debugTimed[tt], 3) * (2 - np.pow(debugTimed[tt], 2))) / (np.pow(debugTimed[tt], 1) +
                     (1 - debugTimed[tt]) * np.exp((2.8 + learningScheduleTime * powAddition) * debugTimed[tt]))) +
                     np.pow(debugTimed[tt], 4.2) * (1 - debugTimed[tt])) - 0.003 * (debugTimed[tt] - 0.7) * debugTimed[ tt] * (1 - debugTimed[tt])

                v_p=np.pow(v,1.0)
                v_pp = np.pow(v_cir, 1.0)
                xDebugTimed[tt, :, :] = xDebug * (1-v_p) + noise_gauss * v_p + v_pp*(1-v_pp)*noise_cir

            for idx in range(xDebugTimed.shape[0]):
                color = cmap(idx)
                # if t[idx]<60:
                #     continue
                for h in range(1, xDebugTimed.shape[1]):
                    plt.plot([xDebugTimed[idx, h - 1, 0], xDebugTimed[idx, h, 0]],
                             [xDebugTimed[idx, h - 1, 1], xDebugTimed[idx, h, 1]],
                             marker='',
                             zorder=2, alpha=0.5, color=color)
            plt.show()

    if isVizualize==True:
        if oneTimeVisGenAB == True:
            cmap_name = 'jet'  # Example: Use the 'jet' colormap
            cmap = cm.get_cmap(cmap_name, (int)(timesteps))
            # for idx in range(x.shape[0]):
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                # if t[idx]<60:
                #     continue
                for h in range(1, img_b.shape[1]):
                    plt.plot([img_b[idx, h - 1, 0], img_b[idx, h, 0]], [img_b[idx, h - 1, 1], img_b[idx, h, 1]],
                             marker='', linewidth=0.1+powAddition[idx,0,0]*5,
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            for idx in range(x.shape[0]):
                color = cmap(t[idx])
                for h in range(1, img_a.shape[1]):
                    plt.plot([img_a[idx, h - 1, 0], img_a[idx, h, 0]], [img_a[idx, h - 1, 1], img_a[idx, h, 1]],
                             marker='', linewidth=0.1+powAddition[idx,0,0]*5,
                             zorder=2, alpha=0.5, color=color)
            plt.show()
            # oneTimeVisGenAB = False
    return img_a, img_b

class ConvLayersDetailed(nn.Module):
    def __init__(self, maxLengthSize,size,lenOffest,featureSize, convOffset=0):
        super(ConvLayersDetailed,self).__init__()
        self.multTrajGate = Multiply()
        self.size = size
        self.convOffset = convOffset
        self.maxLengthSize = maxLengthSize
        self.featureSize = featureSize
        self.lenOffest = lenOffest

        # self.beforeTRE = nn.LayerNorm([self.maxLengthSize+lenOffest,self.size])
        self.beforeTREDense = nn.Linear(self.size, 64)#MAIN
        self.TRE = nn.TransformerEncoderLayer(d_model=64, nhead=4)#MAIN
        self.TRED1 = nn.Linear(64, 2)#MAIN
        self.TRED2 = nn.Linear(self.maxLengthSize, self.featureSize)
        self.TREConv1 = nn.Conv1d(self.maxLengthSize+lenOffest, self.maxLengthSize, 3, padding=1)#MAIN
        self.LSTM = nn.LSTM(self.size, self.maxLengthSize+lenOffest, batch_first=True)
        self.LSTMD1 = nn.Linear(self.maxLengthSize+lenOffest, self.featureSize)
        # self.LSTMD2 = nn.Linear(self.maxLengthSize, self.featureSize)
        self.LSTMConv1 = nn.Conv1d(self.maxLengthSize+lenOffest, self.maxLengthSize, 3, padding=1)

        self.activation1_1 = nn.PReLU()
        self.activation1_2 = nn.PReLU()
        self.activation1_3 = nn.PReLU()
        self.activation1_4 = nn.PReLU()
        self.activation1_5 = nn.PReLU()
        self.activation1_6 = nn.PReLU()
        self.activation1_7 = nn.PReLU()
        self.activation1_8 = nn.PReLU()

        self.activation2_1 = nn.PReLU()
        self.activation2_2 = nn.PReLU()
        self.activation2_3 = nn.PReLU()
        self.activation2_4 = nn.PReLU()
        self.activation2_5 = nn.PReLU()
        self.activation2_6 = nn.PReLU()
        self.activation2_7 = nn.PReLU()
        self.activation2_8 = nn.PReLU()

        self.activation3_1 = nn.PReLU()
        self.activation3_2 = nn.PReLU()
        self.activation3_3 = nn.PReLU()
        self.activation3_4 = nn.PReLU()
        self.activation3_5 = nn.PReLU()
        self.activation3_6 = nn.PReLU()
        self.activation3_7 = nn.PReLU()
        self.activation3_8 = nn.PReLU()

        self.conv6d1 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 7 + convOffset,
                                 padding=(int)(3 + convOffset / 2))
        self.conv7d1 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)(2 + convOffset / 2))
        self.conv8d1 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)(1 + convOffset / 2))

        # self.auxDense = nn.Linear(510, 512)

        # self.conv1d2 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset, padding=(int)((8 + convOffset / 2) * 2), dilation=2)
        # self.conv2d2 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 2), dilation=2)
        # self.conv3d2 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 2),  dilation=2)
        # self.conv4d2 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 2), dilation=2)
        # self.conv5d2 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 2), dilation=2)
        self.conv6d2 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 7 + convOffset,
                                 padding=(int)((3 + convOffset / 2) * 2), dilation=2)
        self.conv7d2 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 2),
                                 dilation=2)
        self.conv8d2 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 2),
                                 dilation=2)

        # self.conv1d3 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset, padding=(int)((8 + convOffset / 2) * 3), dilation=3)
        # self.conv2d3 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 3), dilation=3)
        # self.conv3d3 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 3), dilation=3)
        # self.conv4d3 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 3), dilation=3)
        # self.conv5d3 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 3), dilation=3)
        self.conv6d3 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 7 + convOffset,
                                 padding=(int)((3 + convOffset / 2) * 3), dilation=3)
        self.conv7d3 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 3),
                                 dilation=3)
        self.conv8d3 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 3),
                                 dilation=3)

        # self.inputProcess = nn.Linear(self.size,64)

        # self.lastDenseInPathAdjustTCN = nn.Linear(self.size, maxLengthSize)
        # self.lastDenseInPathAdjustUNET = nn.Linear(64, maxLengthSize)
        # self.lastDenseInPath = nn.Linear(self.maxLengthSize, self.featureSize)

        self.lastDenseInPathAdjustTCN1 = nn.Linear(self.size * 1, maxLengthSize)
        self.lastDenseInPathAdjustTCN2 = nn.Linear(self.size * 1, maxLengthSize)
        self.lastDenseInPathAdjustTCN3 = nn.Linear(self.size * 1, maxLengthSize)
        self.lastDenseInPathTCN1 = nn.Linear(self.maxLengthSize, self.featureSize)
        self.lastDenseInPathTCN2 = nn.Linear(self.maxLengthSize, self.featureSize)
        self.lastDenseInPathTCN3 = nn.Linear(self.maxLengthSize, self.featureSize)

        self.tcn = TCN(maxLengthSize, size, lenOffest, featureSize)

    def forward(self,input,gateValue):
        inDensed = self.beforeTREDense(input)#MAIN
        tre = self.TRE(inDensed)#MAIN
        treConv1 = self.TREConv1(tre)#MAIN
        treD1 = self.TRED1(treConv1)#MAIN
        treD1 = self.multTrajGate(gateValue, treD1)
        lstm = self.LSTM(input)[0]
        lSTMConv1 = self.LSTMConv1(lstm)
        lstmD1 = self.LSTMD1(lSTMConv1)
        lstmD1 = self.multTrajGate(gateValue, lstmD1)

        mergePath111 = self.conv6d1(input)
        mergePath112 = self.activation1_6(mergePath111)

        mergePath113 = self.conv7d1(mergePath112)
        mergePath114 = self.activation1_7(mergePath113)

        mergePath115 = self.conv8d1(mergePath114)
        mergePath116 = self.activation1_8(mergePath115)

        mergePath211 = self.conv6d1(input)
        mergePath212 = self.activation2_6(mergePath211)

        mergePath213 = self.conv7d1(mergePath212)
        mergePath214 = self.activation2_7(mergePath213)

        mergePath215 = self.conv8d1(mergePath214)
        mergePath216 = self.activation2_8(mergePath215)

        mergePath311 = self.conv6d1(input)
        mergePath312 = self.activation3_6(mergePath311)

        mergePath313 = self.conv7d1(mergePath312)
        mergePath314 = self.activation3_7(mergePath313)

        mergePath315 = self.conv8d1(mergePath314)
        mergePath316 = self.activation3_8(mergePath315)

        mergePath117 = self.lastDenseInPathAdjustTCN1(mergePath116)
        mergePath117 = self.multTrajGate(gateValue, mergePath117)
        mergePath217 = self.lastDenseInPathAdjustTCN2(mergePath216)
        mergePath217 = self.multTrajGate(gateValue, mergePath217)
        mergePath317 = self.lastDenseInPathAdjustTCN3(mergePath316)
        mergePath317 = self.multTrajGate(gateValue, mergePath317)

        mergePath1N = self.lastDenseInPathTCN1(mergePath117)
        mergePath1N = self.multTrajGate(gateValue, mergePath1N)
        mergePath2N = self.lastDenseInPathTCN2(mergePath217)
        mergePath2N = self.multTrajGate(gateValue, mergePath2N)
        mergePath3N = self.lastDenseInPathTCN3(mergePath317)
        mergePath3N = self.multTrajGate(gateValue, mergePath3N)

        tcnOut = self.tcn(input)

        # mergePath3Total = torch.cat((mergePath1N, mergePath2N, mergePath3N, input[:,self.lenOffest:,:]), 2)
        mergePath3Total = torch.cat((mergePath1N, mergePath2N, mergePath3N, treD1, lstmD1, tcnOut), 2)

        return mergePath3Total

# class ConvLayersDetailed(nn.Module):
#     def __init__(self, maxLengthSize,size,lenOffest,featureSize, convOffset=0):
#         super(ConvLayersDetailed,self).__init__()
#         self.size=size
#         self.multTrajGate = Multiply()
#         self.convOffset=convOffset
#         self.maxLengthSize=maxLengthSize
#         self.featureSize=featureSize
#
#         # self.beforeTRE = nn.LayerNorm([self.maxLengthSize+lenOffest,self.size])
#         self.beforeTREDense = nn.Linear(self.size, 64)#MAIN
#         self.TRE = nn.TransformerEncoderLayer(d_model=64, nhead=4)#MAIN
#         self.TRED1 = nn.Linear(64, 2)#MAIN
#         self.TRED2 = nn.Linear(self.maxLengthSize, self.featureSize)
#         self.TREConv1 = nn.Conv1d(self.maxLengthSize+lenOffest, self.maxLengthSize, 3, padding=1)#MAIN
#         self.LSTM = nn.LSTM(self.size, self.maxLengthSize+lenOffest, batch_first=True)
#         self.LSTMD1 = nn.Linear(self.maxLengthSize+lenOffest, self.featureSize)
#         # self.LSTMD2 = nn.Linear(self.maxLengthSize, self.featureSize)
#         self.LSTMConv1 = nn.Conv1d(self.maxLengthSize+lenOffest, self.maxLengthSize, 3, padding=1)
#
#         self.activation1 = nn.PReLU()
#         self.activation2 = nn.PReLU()
#         self.activation3 = nn.PReLU()
#         self.activation4 = nn.PReLU()
#         self.activation5 = nn.PReLU()
#         self.activation6 = nn.PReLU()
#         self.activation7 = nn.PReLU()
#         self.activation8 = nn.PReLU()
#         self.activation9 = nn.PReLU()
#         self.activation10 = nn.PReLU()
#
#         self.activation11 = nn.PReLU()
#         self.activation12 = nn.PReLU()
#         self.activation13 = nn.PReLU()
#         self.activation14 = nn.PReLU()
#         self.activation15 = nn.PReLU()
#         self.activation16 = nn.PReLU()
#         self.activation17 = nn.PReLU()
#         self.activation18 = nn.PReLU()
#
#
#         self.activation1_1 = nn.PReLU()
#         self.activation1_2 = nn.PReLU()
#         self.activation1_3 = nn.PReLU()
#         self.activation1_4 = nn.PReLU()
#         self.activation1_5 = nn.PReLU()
#         self.activation1_6 = nn.PReLU()
#         self.activation1_7 = nn.PReLU()
#         self.activation1_8 = nn.PReLU()
#
#         self.activation2_1 = nn.PReLU()
#         self.activation2_2 = nn.PReLU()
#         self.activation2_3 = nn.PReLU()
#         self.activation2_4 = nn.PReLU()
#         self.activation2_5 = nn.PReLU()
#         self.activation2_6 = nn.PReLU()
#         self.activation2_7 = nn.PReLU()
#         self.activation2_8 = nn.PReLU()
#
#         self.activation3_1 = nn.PReLU()
#         self.activation3_2 = nn.PReLU()
#         self.activation3_3 = nn.PReLU()
#         self.activation3_4 = nn.PReLU()
#         self.activation3_5 = nn.PReLU()
#         self.activation3_6 = nn.PReLU()
#         self.activation3_7 = nn.PReLU()
#         self.activation3_8 = nn.PReLU()
#
#         self.activation4_1 = nn.PReLU()
#         self.activation4_2 = nn.PReLU()
#         self.activation4_3 = nn.PReLU()
#         self.activation4_4 = nn.PReLU()
#         self.activation4_5 = nn.PReLU()
#         self.activation4_6 = nn.PReLU()
#         self.activation4_7 = nn.PReLU()
#         self.activation4_8 = nn.PReLU()
#
#
#         self.activationFinal = nn.Tanh()
#         self.conv1d1 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset, padding=(int)(8 + convOffset / 2))
#         self.conv2d1 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)(7 + convOffset / 2))
#         self.conv3d1 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)(6 + convOffset / 2))
#         self.conv4d1 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)(5 + convOffset / 2))
#         self.conv5d1 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)(4 + convOffset / 2))
#         self.conv6d1 = nn.Conv1d(self.size, self.size, 7 + convOffset, padding=(int)(3 + convOffset / 2))
#         self.conv7d1 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)(2 + convOffset / 2))
#         self.conv8d1 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)(1 + convOffset / 2))
#
#         # self.auxDense = nn.Linear(510, 512)
#
#         self.conv1d2 = nn.Conv1d(maxLengthSize +lenOffest, self.size, 17+convOffset, padding=(int)((8+convOffset/2)*2), dilation=2)
#         self.conv2d2 = nn.Conv1d(self.size, self.size, 15+convOffset, padding=(int)((7+convOffset/2)*2), dilation=2)
#         self.conv3d2 = nn.Conv1d(self.size, self.size, 13+convOffset, padding=(int)((6+convOffset/2)*2), dilation=2)
#         self.conv4d2 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2)*2), dilation=2)
#         self.conv5d2 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2)*2), dilation=2)
#         self.conv6d2 = nn.Conv1d(self.size, self.size, 7 + convOffset, padding=(int)((3 + convOffset / 2)*2), dilation=2)
#         self.conv7d2 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2)*2), dilation=2)
#         self.conv8d2 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2)*2), dilation=2)
#
#
#
#         self.conv1d3 = nn.Conv1d(maxLengthSize +lenOffest, self.size, 17+convOffset, padding=(int)((8+convOffset/2)*3), dilation=3)
#         self.conv2d3 = nn.Conv1d(self.size, self.size, 15+convOffset, padding=(int)((7+convOffset/2)*3), dilation=3)
#         self.conv3d3 = nn.Conv1d(self.size, self.size, 13+convOffset, padding=(int)((6+convOffset/2)*3), dilation=3)
#         self.conv4d3 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2)*3), dilation=3)
#         self.conv5d3 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2)*3), dilation=3)
#         self.conv6d3 = nn.Conv1d(self.size, self.size, 7 + convOffset, padding=(int)((3 + convOffset / 2)*3), dilation=3)
#         self.conv7d3 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2)*3), dilation=3)
#         self.conv8d3 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2)*3), dilation=3)
#
#         self.conv1d4 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset,
#                                  padding=(int)((8 + convOffset / 2) * 5), dilation=5)
#         self.conv2d4 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 5),
#                                  dilation=5)
#         self.conv3d4 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 5),
#                                  dilation=5)
#         self.conv4d4 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 5),
#                                  dilation=5)
#         self.conv5d4 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 5),
#                                  dilation=5)
#         self.conv6d4 = nn.Conv1d(self.size, self.size, 7 + convOffset, padding=(int)((3 + convOffset / 2) * 5),
#                                  dilation=5)
#         self.conv7d4 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 5),
#                                  dilation=5)
#         self.conv8d4 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 5),
#                                  dilation=5)
#
#         self.inputProcess = nn.Linear(self.size,64)
#
#         self.e11 = nn.Conv1d(maxLengthSize +lenOffest, 64, kernel_size=3, padding=1)  # output: 570x570x64
#         self.e12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 284x284x64
#
#         # input: 284x284x64
#         self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
#         self.e22 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 140x140x128
#
#         # input: 140x140x128
#         self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
#         self.e32 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 68x68x256
#
#         # input: 68x68x256
#         self.e41 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
#         self.e42 = nn.Conv1d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
#         self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 32x32x512
#
#         # input: 32x32x512
#         self.e51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
#         self.e52 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024
#
#         # Decoder
#         self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
#         self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
#         self.d12 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
#
#         self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
#         self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
#         self.d22 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
#
#         self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
#         self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
#         self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
#
#         self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
#         self.d41 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
#         self.d42 = nn.Conv1d(64, self.maxLengthSize, kernel_size=3, padding=1)
#
#         # self.lastDenseInPathAdjustTCN1 = nn.Linear(self.size, maxLengthSize)
#         # self.lastDenseInPathAdjustTCN2 = nn.Linear(self.size, maxLengthSize)
#         # self.lastDenseInPathAdjustTCN3 = nn.Linear(self.size, maxLengthSize)
#         # self.lastDenseInPathAdjustTCN4 = nn.Linear(self.size, maxLengthSize)
#         self.lastDenseInPathAdjustUNET = nn.Linear(64, maxLengthSize)
#         self.lastDenseInPathUNET = nn.Linear(self.maxLengthSize, self.featureSize)
#         # self.lastDenseInPathTCN1 = nn.Linear(self.maxLengthSize, self.featureSize)
#         # self.lastDenseInPathTCN2 = nn.Linear(self.maxLengthSize, self.featureSize)
#         # self.lastDenseInPathTCN3 = nn.Linear(self.maxLengthSize, self.featureSize)
#         # self.lastDenseInPathTCN4 = nn.Linear(self.maxLengthSize, self.featureSize)
#
#         # self.NormLayer = nn.LayerNorm([self.maxLengthSize, 6*self.featureSize])
#
#     def forward(self,input,gateValue):
#         inDensed = self.beforeTREDense(input)#MAIN
#         tre = self.TRE(inDensed)#MAIN
#         treConv1 = self.TREConv1(tre)#MAIN
#         treD1 = self.TRED1(treConv1)#MAIN
#         lstm = self.LSTM(input)[0]
#         lSTMConv1 = self.LSTMConv1(lstm)
#         lstmD1 = self.LSTMD1(lSTMConv1)
#
#         inputFixed = self.inputProcess(input)
#
#         xe11 = self.activation1(self.e11(inputFixed))
#         xe12 = self.activation2(self.e12(xe11))
#         xp1 = self.pool1(xe12)
#
#         xe21 = self.activation3(self.e21(xp1))
#         xe22 = self.activation4(self.e22(xe21))
#         xp2 = self.pool2(xe22)
#
#         xe31 = self.activation5(self.e31(xp2))
#         xe32 = self.activation6(self.e32(xe31))
#         xp3 = self.pool3(xe32)
#
#         xe41 = self.activation7(self.e41(xp3))
#         xe42 = self.activation8(self.e42(xe41))
#         xp4 = self.pool4(xe42)
#
#         xe51 = self.activation9(self.e51(xp4))
#         xe52 = self.activation10(self.e52(xe51))
#
#         # Decoder
#         xu1 = self.upconv1(xe52)
#         xu11 = torch.cat([xu1, xe42], dim=1)
#         xd11 = self.activation11(self.d11(xu11))
#         xd12 = self.activation12(self.d12(xd11))
#
#         xu2 = self.upconv2(xd12)
#         xu22 = torch.cat([xu2, xe32], dim=1)
#         xd21 = self.activation13(self.d21(xu22))
#         xd22 = self.activation14(self.d22(xd21))
#
#         xu3 = self.upconv3(xd22)
#         xu33 = torch.cat([xu3, xe22], dim=1)
#         xd31 = self.activation15(self.d31(xu33))
#         xd32 = self.activation16(self.d32(xd31))
#
#         xu4 = self.upconv4(xd32)
#         xu44 = torch.cat([xu4, xe12], dim=1)
#         xd41 = self.activation17(self.d41(xu44))
#         xd42 = self.activation18(self.d42(xd41))
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath1 = self.conv1d1(input)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath1 = self.activation1_1(mergePath1)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath1 = self.conv2d1(mergePath1)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath1 = self.activation1_2(mergePath1)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath1 = self.conv3d1(mergePath1)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath1 = self.activation1_3(mergePath1)
#
#         mergePath1 = self.conv4d1(mergePath1)
#         mergePath1 = self.activation1_4(mergePath1)
#
#         mergePath1 = self.conv5d1(mergePath1)
#         mergePath1 = self.activation1_5(mergePath1)
#
#         mergePath1 = self.conv6d1(mergePath1)
#         mergePath1 = self.activation1_6(mergePath1)
#
#         mergePath1 = self.conv7d1(mergePath1)
#         mergePath1 = self.activation1_7(mergePath1)
#
#         mergePath1 = self.conv8d1(mergePath1)
#         mergePath1 = self.activation1_8(mergePath1)
#
#
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath2 = self.conv1d2(input)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath2 = self.activation2_1(mergePath2)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath2 = self.conv2d2(mergePath2)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath2 = self.activation2_2(mergePath2)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath2 = self.conv3d2(mergePath2)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath2 = self.activation2_3(mergePath2)
#
#         mergePath2 = self.conv4d2(mergePath2)
#         mergePath2 = self.activation2_4(mergePath2)
#
#         mergePath2 = self.conv5d2(mergePath2)
#         mergePath2 = self.activation2_5(mergePath2)
#
#         mergePath2 = self.conv6d2(mergePath2)
#         mergePath2 = self.activation2_6(mergePath2)
#
#         mergePath2 = self.conv7d2(mergePath2)
#         mergePath2 = self.activation2_7(mergePath2)
#
#         mergePath2 = self.conv8d2(mergePath2)
#         mergePath2 = self.activation2_8(mergePath2)
#
#
#
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath3 = self.conv1d3(input)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath3 = self.activation3_1(mergePath3)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath3 = self.conv2d3(mergePath3)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath3 = self.activation3_2(mergePath3)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath3 = self.conv3d3(mergePath3)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath3 = self.activation3_3(mergePath3)
#
#         mergePath3 = self.conv4d3(mergePath3)
#         mergePath3 = self.activation3_4(mergePath3)
#
#         mergePath3 = self.conv5d3(mergePath3)
#         mergePath3 = self.activation3_5(mergePath3)
#
#         mergePath3 = self.conv6d3(mergePath3)
#         mergePath3 = self.activation3_6(mergePath3)
#
#         mergePath3 = self.conv7d3(mergePath3)
#         mergePath3 = self.activation3_7(mergePath3)
#
#         mergePath3 = self.conv8d3(mergePath3)
#         mergePath3 = self.activation3_8(mergePath3)
#
#
#
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath4 = self.conv1d4(input)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath4 = self.activation4_1(mergePath4)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath4 = self.conv2d4(mergePath4)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath4 = self.activation4_2(mergePath4)
#
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath4 = self.conv3d4(mergePath4)
#         # mergePath = mergePath.permute(0, 2, 1)
#         mergePath4 = self.activation4_3(mergePath4)
#
#         mergePath4 = self.conv4d4(mergePath4)
#         mergePath4 = self.activation4_4(mergePath4)
#
#         mergePath4 = self.conv5d4(mergePath4)
#         mergePath4 = self.activation4_5(mergePath4)
#
#         mergePath4 = self.conv6d4(mergePath4)
#         mergePath4 = self.activation4_6(mergePath4)
#
#         mergePath4 = self.conv7d4(mergePath4)
#         mergePath4 = self.activation4_7(mergePath4)
#
#         mergePath4 = self.conv8d4(mergePath4)
#         mergePath4 = self.activation4_8(mergePath4)
#
#
#
#         # mergePath1 = self.lastDenseInPathAdjustTCN1(mergePath1)
#         # mergePath2 = self.lastDenseInPathAdjustTCN2(mergePath2)
#         # mergePath3 = self.lastDenseInPathAdjustTCN3(mergePath3)
#         # mergePath4 = self.lastDenseInPathAdjustTCN4(mergePath4)
#         xd42 = self.lastDenseInPathAdjustUNET(xd42)
#
#         # mergePath1 = torch.reshape(mergePath1, (input.shape[0], self.maxLengthSize, -1))
#         # mergePath2 = torch.reshape(mergePath2, (input.shape[0], self.maxLengthSize, -1))
#         # mergePath3 = torch.reshape(mergePath3, (input.shape[0], self.maxLengthSize, -1))
#         # xd42 = torch.reshape(xd42, (input.shape[0], self.maxLengthSize, -1))
#
#         # mergePath1 = self.lastDenseInPathTCN1(mergePath1)
#         # mergePath2 = self.lastDenseInPathTCN2(mergePath2)
#         # mergePath3 = self.lastDenseInPathTCN3(mergePath3)
#         # mergePath4 = self.lastDenseInPathTCN4(mergePath4)
#         xd42 = self.lastDenseInPathUNET(xd42)
#
#         # mergePath3Total = torch.cat((mergePath1, mergePath2, mergePath3, mergePath4, xd42, treD1, lstmD1), 2)
#         mergePath3Total = torch.cat((mergePath1, mergePath2, mergePath3, mergePath4, xd42, treD1, lstmD1), 2)
#         # output = self.NormLayer(mergePath3Total)
#         mergePath3Total = self.multTrajGate(gateValue, mergePath3Total)
#         return mergePath3Total


class ConvLayers(nn.Module):
    def __init__(self, maxLengthSize,size,lenOffest,featureSize, convOffset=0):
        super(ConvLayers,self).__init__()
        self.multTrajGate = Multiply()
        self.size=size
        self.convOffset=convOffset
        self.maxLengthSize=maxLengthSize
        self.featureSize=featureSize
        self.lenOffest = lenOffest

        self.activation1_1 = nn.PReLU()
        self.activation1_2 = nn.PReLU()
        self.activation1_3 = nn.PReLU()
        self.activation1_4 = nn.PReLU()
        self.activation1_5 = nn.PReLU()
        self.activation1_6 = nn.PReLU()
        self.activation1_7 = nn.PReLU()
        self.activation1_8 = nn.PReLU()

        self.activation2_1 = nn.PReLU()
        self.activation2_2 = nn.PReLU()
        self.activation2_3 = nn.PReLU()
        self.activation2_4 = nn.PReLU()
        self.activation2_5 = nn.PReLU()
        self.activation2_6 = nn.PReLU()
        self.activation2_7 = nn.PReLU()
        self.activation2_8 = nn.PReLU()

        self.activation3_1 = nn.PReLU()
        self.activation3_2 = nn.PReLU()
        self.activation3_3 = nn.PReLU()
        self.activation3_4 = nn.PReLU()
        self.activation3_5 = nn.PReLU()
        self.activation3_6 = nn.PReLU()
        self.activation3_7 = nn.PReLU()
        self.activation3_8 = nn.PReLU()


        # self.activationFinal = nn.Tanh()
        # self.conv1d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 17+convOffset, padding=(int)(8+convOffset/2))
        # self.conv1d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv2d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 15+convOffset, padding=(int)(7+convOffset/2))
        # self.conv2d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv3d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 13+convOffset, padding=(int)(6+convOffset/2))
        # self.conv3d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv4d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 11 + convOffset, padding=(int)(5 + convOffset / 2))
        # self.conv4d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv5d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 9 + convOffset, padding=(int)(4 + convOffset / 2))
        # self.conv5d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv6d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 7 + convOffset, padding=(int)(3 + convOffset / 2))
        # self.conv6d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv7d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 5 + convOffset, padding=(int)(2 + convOffset / 2))
        # self.conv7d1Dense = nn.Linear(self.size, self.featureSize)
        # self.conv8d1 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 3 + convOffset, padding=(int)(1 + convOffset / 2))
        # self.conv8d1Dense = nn.Linear(self.size, self.featureSize)
        #
        # self.conv1d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 17 + convOffset, padding=(int)((8 + convOffset / 2) * 2), dilation=2)
        # self.conv1d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv2d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 2), dilation=2)
        # self.conv2d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv3d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 2), dilation=2)
        # self.conv3d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv4d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 2), dilation=2)
        # self.conv4d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv5d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 2), dilation=2)
        # self.conv5d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv6d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 7 + convOffset, padding=(int)((3 + convOffset / 2) * 2), dilation=2)
        # self.conv6d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv7d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 2), dilation=2)
        # self.conv7d2Dense = nn.Linear(self.size, self.featureSize)
        # self.conv8d2 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 2), dilation=2)
        # self.conv8d2Dense = nn.Linear(self.size, self.featureSize)
        #
        #
        # self.conv1d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 17 + convOffset, padding=(int)((8 + convOffset / 2) * 3), dilation=3)
        # self.conv1d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv2d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 3),  dilation=3)
        # self.conv2d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv3d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 3), dilation=3)
        # self.conv3d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv4d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 3), dilation=3)
        # self.conv4d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv5d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 3), dilation=3)
        # self.conv5d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv6d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 7 + convOffset, padding=(int)((3 + convOffset / 2) * 3), dilation=3)
        # self.conv6d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv7d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 3), dilation=3)
        # self.conv7d3Dense = nn.Linear(self.size, self.featureSize)
        # self.conv8d3 = nn.Conv1d(maxLengthSize + lenOffest, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 3), dilation=3)
        # self.conv8d3Dense = nn.Linear(self.size, self.featureSize)

        # self.conv1d1 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset, padding=(int)(8 + convOffset / 2))
        # self.conv2d1 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)(7 + convOffset / 2))
        # self.conv3d1 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)(6 + convOffset / 2))
        # self.conv4d1 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)(5 + convOffset / 2))
        # self.conv5d1 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)(4 + convOffset / 2))
        self.conv6d1 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 7 + convOffset, padding=(int)(3 + convOffset / 2))
        self.conv7d1 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)(2 + convOffset / 2))
        self.conv8d1 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)(1 + convOffset / 2))

        # self.auxDense = nn.Linear(510, 512)

        # self.conv1d2 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset, padding=(int)((8 + convOffset / 2) * 2), dilation=2)
        # self.conv2d2 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 2), dilation=2)
        # self.conv3d2 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 2),  dilation=2)
        # self.conv4d2 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 2), dilation=2)
        # self.conv5d2 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 2), dilation=2)
        self.conv6d2 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 7 + convOffset, padding=(int)((3 + convOffset / 2) * 2),  dilation=2)
        self.conv7d2 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 2), dilation=2)
        self.conv8d2 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 2), dilation=2)

        # self.conv1d3 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 17 + convOffset, padding=(int)((8 + convOffset / 2) * 3), dilation=3)
        # self.conv2d3 = nn.Conv1d(self.size, self.size, 15 + convOffset, padding=(int)((7 + convOffset / 2) * 3), dilation=3)
        # self.conv3d3 = nn.Conv1d(self.size, self.size, 13 + convOffset, padding=(int)((6 + convOffset / 2) * 3), dilation=3)
        # self.conv4d3 = nn.Conv1d(self.size, self.size, 11 + convOffset, padding=(int)((5 + convOffset / 2) * 3), dilation=3)
        # self.conv5d3 = nn.Conv1d(self.size, self.size, 9 + convOffset, padding=(int)((4 + convOffset / 2) * 3), dilation=3)
        self.conv6d3 = nn.Conv1d(maxLengthSize + lenOffest, self.size, 7 + convOffset, padding=(int)((3 + convOffset / 2) * 3), dilation=3)
        self.conv7d3 = nn.Conv1d(self.size, self.size, 5 + convOffset, padding=(int)((2 + convOffset / 2) * 3), dilation=3)
        self.conv8d3 = nn.Conv1d(self.size, self.maxLengthSize, 3 + convOffset, padding=(int)((1 + convOffset / 2) * 3), dilation=3)


        # self.inputProcess = nn.Linear(self.size,64)

        # self.lastDenseInPathAdjustTCN = nn.Linear(self.size, maxLengthSize)
        # self.lastDenseInPathAdjustUNET = nn.Linear(64, maxLengthSize)
        # self.lastDenseInPath = nn.Linear(self.maxLengthSize, self.featureSize)

        self.lastDenseInPathAdjustTCN1 = nn.Linear(self.size*1, maxLengthSize)
        self.lastDenseInPathAdjustTCN2 = nn.Linear(self.size*1, maxLengthSize)
        self.lastDenseInPathAdjustTCN3 = nn.Linear(self.size*1, maxLengthSize)
        self.lastDenseInPathTCN1 = nn.Linear(self.maxLengthSize, self.featureSize)
        self.lastDenseInPathTCN2 = nn.Linear(self.maxLengthSize, self.featureSize)
        self.lastDenseInPathTCN3 = nn.Linear(self.maxLengthSize, self.featureSize)

    def forward(self,input, gateValue):
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath11 = self.conv1d1(input)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath12 = self.activation1_1(mergePath11)
        #
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath13 = self.conv2d1(mergePath12)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath14 = self.activation1_2(mergePath13)
        #
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath15 = self.conv3d1(mergePath14)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath16 = self.activation1_3(mergePath15)
        #
        # mergePath17 = self.conv4d1(mergePath16)
        # mergePath18 = self.activation1_4(mergePath17)
        #
        # mergePath19 = self.conv5d1(mergePath18)
        # mergePath110 = self.activation1_5(mergePath19)

        mergePath111 = self.conv6d1(input)
        mergePath112 = self.activation1_6(mergePath111)

        mergePath113 = self.conv7d1(mergePath112)
        mergePath114 = self.activation1_7(mergePath113)

        mergePath115 = self.conv8d1(mergePath114)
        mergePath116 = self.activation1_8(mergePath115)

        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath21 = self.conv1d2(input)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath22 = self.activation2_1(mergePath21)
        #
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath23 = self.conv2d2(mergePath22)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath24 = self.activation2_2(mergePath23)
        #
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath25 = self.conv3d2(mergePath24)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath26 = self.activation2_3(mergePath25)
        #
        # mergePath27 = self.conv4d1(mergePath26)
        # mergePath28 = self.activation2_4(mergePath27)
        #
        # mergePath29 = self.conv5d1(mergePath28)
        # mergePath210 = self.activation2_5(mergePath29)

        mergePath211 = self.conv6d1(input)
        mergePath212 = self.activation2_6(mergePath211)

        mergePath213 = self.conv7d1(mergePath212)
        mergePath214 = self.activation2_7(mergePath213)

        mergePath215 = self.conv8d1(mergePath214)
        mergePath216 = self.activation2_8(mergePath215)

        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath31 = self.conv1d3(input)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath32 = self.activation3_1(mergePath31)
        #
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath33 = self.conv2d3(mergePath32)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath34 = self.activation3_2(mergePath33)
        #
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath35 = self.conv3d3(mergePath34)
        # # mergePath = mergePath.permute(0, 2, 1)
        # mergePath36 = self.activation3_3(mergePath35)
        #
        # mergePath37 = self.conv4d1(mergePath36)
        # mergePath38 = self.activation3_4(mergePath37)
        #
        # mergePath39 = self.conv5d1(mergePath38)
        # mergePath310 = self.activation3_5(mergePath39)

        mergePath311 = self.conv6d1(input)
        mergePath312 = self.activation3_6(mergePath311)

        mergePath313 = self.conv7d1(mergePath312)
        mergePath314 = self.activation3_7(mergePath313)

        mergePath315 = self.conv8d1(mergePath314)
        mergePath316 = self.activation3_8(mergePath315)

        mergePath117 = self.lastDenseInPathAdjustTCN1(mergePath116)
        mergePath117 = self.multTrajGate(gateValue, mergePath117)
        mergePath217 = self.lastDenseInPathAdjustTCN2(mergePath216)
        mergePath217 = self.multTrajGate(gateValue, mergePath217)
        mergePath317 = self.lastDenseInPathAdjustTCN3(mergePath316)
        mergePath317 = self.multTrajGate(gateValue, mergePath317)



        # mergePath1Total = torch.cat((mergePath1_1, mergePath1_2, mergePath1_3, mergePath1_4, mergePath1_5, mergePath1_6,
        #                              mergePath1_7, mergePath1_8), 2)
        # mergePath2Total = torch.cat((mergePath2_1, mergePath2_2, mergePath2_3, mergePath2_4, mergePath2_5, mergePath2_6,
        #                              mergePath2_7, mergePath2_8), 2)
        # mergePath3Total = torch.cat((mergePath3_1, mergePath3_2, mergePath3_3, mergePath3_4, mergePath3_5, mergePath3_6,
        #                              mergePath3_7, mergePath3_8), 2)

        # mergePath1 = self.lastDenseInPathAdjustTCN1(mergePath1)
        # mergePath2 = self.lastDenseInPathAdjustTCN2(mergePath2)
        # mergePath3 = self.lastDenseInPathAdjustTCN3(mergePath3)

        # mergePath1 = torch.reshape(mergePath1, (input.shape[0], self.maxLengthSize, -1))
        # mergePath2 = torch.reshape(mergePath2, (input.shape[0], self.maxLengthSize, -1))
        # mergePath3 = torch.reshape(mergePath3, (input.shape[0], self.maxLengthSize, -1))
        # xd42 = torch.reshape(xd42, (input.shape[0], self.maxLengthSize, -1))

        mergePath1N = self.lastDenseInPathTCN1(mergePath117)
        mergePath1N = self.multTrajGate(gateValue,mergePath1N)
        mergePath2N = self.lastDenseInPathTCN2(mergePath217)
        mergePath2N = self.multTrajGate(gateValue, mergePath2N)
        mergePath3N = self.lastDenseInPathTCN3(mergePath317)
        mergePath3N = self.multTrajGate(gateValue, mergePath3N)

        # mergePath3Total = torch.cat((mergePath1N, mergePath2N, mergePath3N, input[:,self.lenOffest:,:]), 2)
        mergePath3Total = torch.cat((mergePath1N, mergePath2N, mergePath3N), 2)

        # counter = 0
        # for param in self.parameters():
        #     if param.requires_grad==True and param.is_leaf==True:
        #         counter = counter +1
        #         print(param.shape)
        # print(counter)
        # print("FINISHED INSIDE TCN UNNAMED")
        # counter = 0
        # for name, param in self.named_parameters():
        #     if param.requires_grad==True and param.is_leaf==True:
        #         counter = counter + 1
        #         print(name, param.shape)
        # print(counter)

        # allVars = list(locals().items())
        # for var in allVars:
        #     if isinstance(var[1], torch.Tensor)==True:
        #         if var[1].requires_grad==True and var[1].is_leaf==True:
        #             print(var[0])
        # print("FINISHED INSIDE TCN")
        return mergePath3Total

class TCN(nn.Module):
    def __init__(self, maxLengthSize, size, lenOffest, featureSize):
        super(TCN, self).__init__()
        self.size = size
        self.maxLengthSize = maxLengthSize
        self.featureSize = featureSize
        self.inputProcess = nn.Linear(self.size, 64)

        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation3 = nn.PReLU()
        self.activation4 = nn.PReLU()
        self.activation5 = nn.PReLU()
        self.activation6 = nn.PReLU()
        self.activation7 = nn.PReLU()
        self.activation8 = nn.PReLU()
        self.activation9 = nn.PReLU()
        self.activation10 = nn.PReLU()

        self.activation11 = nn.PReLU()
        self.activation12 = nn.PReLU()
        self.activation13 = nn.PReLU()
        self.activation14 = nn.PReLU()
        self.activation15 = nn.PReLU()
        self.activation16 = nn.PReLU()
        self.activation17 = nn.PReLU()
        self.activation18 = nn.PReLU()

        self.e11 = nn.Conv1d(maxLengthSize + lenOffest, 64, kernel_size=3, padding=1)  # output: 570x570x64
        self.e12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
        self.e22 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
        self.e32 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
        self.e42 = nn.Conv1d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
        self.e52 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv1d(64, self.maxLengthSize, kernel_size=3, padding=1)

        self.lastDenseInPathAdjustUNET = nn.Linear(64, maxLengthSize)
        self.lastDenseInPathUNET = nn.Linear(self.maxLengthSize, self.featureSize)

    def forward(self, input):
        inputFixed = self.inputProcess(input)

        xe11 = self.activation1(self.e11(inputFixed))
        xe12 = self.activation2(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.activation3(self.e21(xp1))
        xe22 = self.activation4(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.activation5(self.e31(xp2))
        xe32 = self.activation6(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.activation7(self.e41(xp3))
        xe42 = self.activation8(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.activation9(self.e51(xp4))
        xe52 = self.activation10(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.activation11(self.d11(xu11))
        xd12 = self.activation12(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.activation13(self.d21(xu22))
        xd22 = self.activation14(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.activation15(self.d31(xu33))
        xd32 = self.activation16(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.activation17(self.d41(xu44))
        xd42 = self.activation18(self.d42(xd41))

        o1 = self.lastDenseInPathAdjustUNET(xd42)

        o2 = self.lastDenseInPathUNET(o1)

        return o2