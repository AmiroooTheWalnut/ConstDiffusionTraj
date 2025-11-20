import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from triton.language import dtype, tensor
import matplotlib.cm as cm
import math

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

    def forward(self, y_pred, y_true, time):
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
        return torch.mean(torch.mean(mae_loss, dim=(1, 2)) * torch.squeeze(torch.pow(2 - time, 2)))

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
        finFin = valsMin.squeeze()
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

        poolValue3 = torch.squeeze(poolValue2)

        # poolValue4 = torch.min(poolValue3, dim=1, keepdim=True).values
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
    def __init__(self, forbiddenSerialMap=None, lenForbidden=10, maxLengthSize = 10, temporalFeatureSize=2, convOffset=0):
        super(SimpleNN, self).__init__()
        self.size = 200
        self.maxLengthSize = maxLengthSize
        self.temporalFeatureSize = temporalFeatureSize
        self.multPrePsl = Multiply()

        self.timeDense = nn.Linear(1, self.size)
        self.forbDense = nn.Linear(2, self.size)
        self.timeForbMixedLinear = nn.Linear(2, self.size)
        self.multTrajCondForbTime = Multiply()

        psl = PairwiseSubtractionLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        pslSum = PairwiseSubtractionSumLayer(forbiddenSerialMap, lenForbidden, maxLengthSize)
        self.psl = psl
        self.pslSum = pslSum
        self.pslDense = nn.Linear(1 + 2, self.maxLengthSize * 10)
        # self.trajCondConcat=
        self.trajInput = nn.Linear(temporalFeatureSize, self.size)
        # self.conditionInput = nn.Linear(2, 16)
        self.trajInputDirect = nn.Linear(temporalFeatureSize, self.size)
        self.timeInput = nn.Linear(1 + 2, maxLengthSize * self.size)
        self.mult1 = Multiply()

        self.ConvDirect = ConvLayers(maxLengthSize,self.size,1,self.temporalFeatureSize,convOffset=convOffset)
        self.ConvDists = ConvLayers(maxLengthSize,self.size,0,self.temporalFeatureSize,convOffset=convOffset)

        self.allInfoDense = nn.Linear(self.size*3, self.size)

        # self.activation = nn.SiLU()
        # self.activationFinal = nn.Tanh()


        # self.denseAfterCat3 = nn.Linear(self.size * 3, self.size)

        self.denseAfterCat4Traj = nn.Linear(self.temporalFeatureSize * 3, self.size)
        self.denseAfterCat4Forb = nn.Linear(self.temporalFeatureSize * 3, self.size)
        # self.denseAfterCat4Traj = nn.Linear(8, self.size)
        # self.denseAfterCat4Forb = nn.Linear(8, self.size)
        self.denseAfterCat5 = nn.Linear(self.size * 5, self.size)
        self.lastAdd = Add3()
        self.lastDense = nn.Linear(self.size + 10, 128)
        self.lastDenseAlt = nn.Linear(128, 2)
        self.convLast = nn.Conv1d(128, 2, 1)

    def forward(self, traj, time, condLat, condLon):
        plsPath = self.psl(traj)
        plsPathSum = self.pslSum(traj)

        # plsPathUnsqueeze = torch.unsqueeze(plsPath, 1)
        # timeUnsqueeze = torch.unsqueeze(time, 1)
        # timeForbsMixed = torch.cat((plsPath, timeUnsqueeze), dim=2)

        forbDensed = self.forbDense(plsPath)

        # timeDensed = self.timeDense(timeUnsqueeze)
        # timeDensed = self.activationFinal(timeDensed)


        plsPath1 = self.multPrePsl(plsPathSum, time)
        mixedPlsPathCond = torch.cat((plsPath1, condLat, condLon), 1)
        plsPathD = self.pslDense(mixedPlsPathCond)
        plsPathDRS = torch.reshape(plsPathD, (-1, self.maxLengthSize, (int)(10)))

        condLatLon = torch.cat((torch.unsqueeze(condLat, 1), torch.unsqueeze(condLon, 1)), dim=2)
        mixedTrajCond = torch.cat((condLatLon, traj), 1)

        trajPath = self.trajInput(mixedTrajCond)

        # trajPathAfterMultForbTime = self.multTrajCondForbTime(timeForbsMixedLinearOut, trajPath)
        # trajPathAfterCatForbTime = torch.cat((forbDensed,trajPath),dim=1)

        trajPathDirect = self.trajInputDirect(traj)
        mixedTimeCond = torch.cat((time, condLat, condLon), 1)
        timePath = self.timeInput(mixedTimeCond)
        timePath2 = torch.reshape(timePath, (-1, self.maxLengthSize, self.size))
        # mergePathRoot = self.mult1(trajPath, timePath)

        convedTrajs = self.ConvDirect(trajPath)
        convedDists = self.ConvDists(forbDensed)

        mergePathTraj = self.denseAfterCat4Traj(convedTrajs)
        mergePathForb = self.denseAfterCat4Forb(convedDists)

        allInfo = torch.cat((mergePathTraj,mergePathForb,timePath2),dim=2)
        # mergePath3Total = self.mult1(mergePath3Total, timePath)
        allInfoDrnsed = self.allInfoDense(allInfo)
        # allInfoDrnsed = self.activationFinal(allInfoDrnsed)

        # finalPath = self.lastAdd(timePath, mergePath3Total, trajPathDirect)
        finalPath = torch.cat((timePath2, allInfoDrnsed, trajPathDirect,mergePathTraj,mergePathForb), 2)
        finalPath2 = self.denseAfterCat5(finalPath)
        # finalPath = self.activationFinal(finalPath)

        finalPathC = torch.cat((finalPath2, plsPathDRS), 2)
        finalPathC2 = self.lastDense(finalPathC)
        # finalPathC = self.activationFinal(finalPathC)

        finalPathC3 = finalPathC2.permute(0, 2, 1)
        finalPathC4 = self.convLast(finalPathC3)
        finalPathC5 = finalPathC4.permute(0, 2, 1)
        # finalPathC3 = self.lastDenseAlt(finalPathC2)

        # totalCount=0
        # counter = 0
        # for param in self.parameters():
        #     totalCount=totalCount+1
        #     if param.requires_grad == True and param.is_leaf == True:
        #         counter = counter + 1
        #         print(param.shape)
        # print(totalCount)
        # print(counter)
        # print("FINISHED UNNAMED")
        # totalCount = 0
        # counter = 0
        # for name, param in self.named_parameters():
        #     totalCount=totalCount+1
        #     if param.requires_grad == True and param.is_leaf == True:
        #         counter = counter + 1
        #         print(name, param.shape)
        # print(totalCount)
        # print(counter)
        #
        #
        # allVars = list(locals().items())
        # for var in allVars:
        #     if isinstance(var[1], torch.Tensor) == True:
        #         if var[1].requires_grad == True and var[1].is_leaf == True:
        #             print(var[0])
        # print("FINISHED")
        return finalPathC5

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

    # lastOnes = np.random.randint((int)(math.floor(timesteps * 0.4)), timesteps,
    #                              size=(int)(num / (0.68)))
    # lastOnes2 = np.random.randint((int)(math.floor(timesteps * 0.8)), timesteps,
    #                               size=(int)(num / (0.5)))
    # lastOnes3 = np.random.randint((int)(math.floor(timesteps * 0.94)), timesteps,
    #                               size=(int)(num / (0.11)))

    finalTs = np.concat((orig,noisyOnes,lastOnes,lastOnes2,lastOnes3))
    finalTs2 = np.random.choice(finalTs,num)
    return finalTs2
    # return np.arange(0,timesteps,1,dtype=int)
    # return np.random.randint(timesteps-1, timesteps, size=num)
    # return np.random.randint(timesteps-3, timesteps, size=num)

    # noisyOnes = np.random.randint(0, (int)(math.floor(timesteps * 0.99)),
    #                               size=(int)(num / (1)))
    # orig = np.random.randint(0, (int)(math.floor(timesteps * 0.99)),
    #                          size=(int)(num / (1)))
    # finalTs = np.concat((orig, noisyOnes))
    #
    # finalTs = np.random.choice(finalTs, num)
    # return finalTs

def forward_noise_notNormalized(meanVals, varVals, timesteps, x, t, learningScheduleTime, isChangeWeights, isVizualize=True, isAdvancedWeighting=True):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(loc=meanVals,scale=varVals,size=x.shape)  # noise mask
    # noise = np.random.normal(loc=0.5,scale=0.33,size=x.shape)  # noise mask
    # noise = np.random.uniform(low=0,high=1,size=x.shape)

    # for i in range(1, noise.shape[0]):
    #     for h in range(1, 12):
    #         plt.plot([noise[i, h - 1, 0], noise[i, h, 0]], [noise[i, h - 1, 1], noise[i, h, 1]], marker='', zorder=2, alpha=0.5,color='c')


    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))
    # img_a = x * (1 - a) + noise * a
    # img_b = x * (1 - b) + noise * b

    if isAdvancedWeighting == True:
        if isChangeWeights == False:
            learningScheduleTime = 1
        img_a = x * (1 - np.pow(a, 1.5 + learningScheduleTime * 3.0)) + noise * np.pow(a, 1.5 + learningScheduleTime * 3.0)
        img_b = x * (1 - np.pow(b, 1.5 + learningScheduleTime * 3.0)) + noise * np.pow(b, 1.5 + learningScheduleTime * 3.0)
    else:
        img_a = x * (1 - np.pow(a, 4)) + noise * np.pow(a, 4)
        img_b = x * (1 - np.pow(b, 4)) + noise * np.pow(b, 4)


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
            oneTimeVisGenAB = False

    return img_a, img_b


def forward_noise(timesteps, x, t, learningScheduleTime, isChangeWeights, isVizualize=True, isAdvancedWeighting=True):
    global oneTimeVisGenAB
    time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)  # linspace for timesteps
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(loc=0.5,scale=0.5,size=x.shape)  # noise mask

    a = a.reshape((-1, 1, 1))
    b = b.reshape((-1, 1, 1))

    if isAdvancedWeighting == True:
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
            oneTimeVisGenAB = False

    return img_a, img_b


class ConvLayers(nn.Module):
    def __init__(self, maxLengthSize,size,lenOffest,featureSize, convOffset=0):
        super(ConvLayers,self).__init__()
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

    def forward(self,input):
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
        mergePath217 = self.lastDenseInPathAdjustTCN2(mergePath216)
        mergePath317 = self.lastDenseInPathAdjustTCN3(mergePath316)



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
        mergePath2N = self.lastDenseInPathTCN2(mergePath217)
        mergePath3N = self.lastDenseInPathTCN3(mergePath317)

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