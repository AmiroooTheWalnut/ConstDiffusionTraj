# Version 5 with initial condition with 4 inputs
from sympy.abc import alpha
from torch.optim.lr_scheduler import ExponentialLR

import DataGeneratorFcn
from BatchCumulationCalc import batchCumulationCalc
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
from QualityMeasure import JSD, JSD_SingleB
from torchviz import make_dot
from torchview import draw_graph
import PIL.Image as Image
from debugWeights import summarize_weights_by_type
from LossVisualizer import LossVisualizer

def visualize_grid(selectedOrNotSelected,numInstances, input, axis=None, auxSelectedOrNotSelected=None):
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
        selectedOrNotSelected=selectedOrNotSelected-1*auxSelectedOrNotSelected
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

def visualize_extent(selectedOrNotSelected,numInstances, input, selectedOrNotSelectedSerialized, axis=None, auxSelectedOrNotSelected=None, saveName=None, serializedDirectPoint=None):
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
        selectedOrNotSelected=selectedOrNotSelected-1*auxSelectedOrNotSelected
    if axis == None:
        plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                    extent=(minX, maxX, minY, maxY),
                    origin='lower', zorder=1, alpha=0.99)
    else:
        axis.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                   extent=(minX, maxX, minY, maxY),
                   origin='lower', zorder=1, alpha=0.99)
    # plt.grid(True)
    if serializedDirectPoint!=None:
        serializedDirectPointI=serializedDirectPoint.detach().cpu()
        plt.scatter(serializedDirectPointI[0,:],serializedDirectPointI[1,:])
    if axis == None:
        if saveName != None:
            plt.savefig(saveName + "_Results.png")
        plt.show()

import V19 as MinimalisticModel

print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(enabled=True)

modelVersion="V19_real1_len12_200_ADAM_LOSS_test10Reduced"
isTrainModel=True
continueTrain=True
isChangeWeights=True
isRunOnCPU=False
isCompleteRandomCities=False
fixedOffset=0.999
weightingStepSize=0.8
beforeStepsFixed=0
numOptIterates=0
extraStepsFixed=50
initialLR=0.00001
totalBatchSize=950
BATCH_SIZE=70
repeatTimes=5
cumulativeIters=batchCumulationCalc(totalBatchSize,BATCH_SIZE)
scale=10000
timesteps=20

numTrajectories=5
maxTrajectoryLength=12

numCities = 1
numCitiesTrain = 1
rng = np.random.default_rng(0)
seeds = rng.integers(low=0, high=100, size=numCities)
# seeds[0] = 0#FOR DEBUGGING!
datas=[]
selectedOrNotSelecteds=[]
serializedSelected2DMatrixs=[]


# # DEBUGGING WITH SINGLE TRAJECTORIES
# # seed 19
# [dataDebug, nGrid, serializedSelected2DMatrix, selectedOrNotSelected]=DataGeneratorFcn.generateSyntheticDataFixedLengthInputImageLeftToRight("testStreets2.png", numTrajectories=numTrajectories,
#                                                                 trajectoryLength=maxTrajectoryLength, numGrid=40,
#                                                                 seed=19, visualize=True)
# selectedOrNotSelected=selectedOrNotSelected*scale
# data=np.zeros((1000,maxTrajectoryLength,2))
# row=0
# while row <1000:
#     for dp in range(dataDebug.shape[0]):
#         data[row,:,:]=dataDebug[dp,:,:]
#         row=row+1
#         if row>=1000:
#             break
#     if row >= 1000:
#         break
# data=data*scale
# initLats=np.float32(dataDebug[:,0,0]*scale)
# initLons=np.float32(dataDebug[:,0,1]*scale)
# print("!!!")


# [data, nGrid, serializedSelected2DMatrix,selectedOrNotSelected] =DataGeneratorFcn.generateSyntheticDataFixedLengthInputImage("testStreets2.png",numTrajectories=numTrajectories,trajectoryLength=maxTrajectoryLength,numGrid=40, seed=123,visualize=True)
#
# selectedOrNotSelected=selectedOrNotSelected*scale
# data=data*scale


# [data, nGrid, serializedSelected2DMatrix, selectedOrNotSelected]=DataGeneratorFcn.generateSyntheticDataFixedLengthInputImageLeftToRight("testStreets2.png", numTrajectories=numTrajectories,
#                                                                 trajectoryLength=maxTrajectoryLength, numGrid=40,
#                                                                 seed=123, visualize=True)





# [data, nGrid, selectedOrNotSelected, serializedSelected2DMatrix]=DataGeneratorFcn.generateSyntheticDataVariableLengthInputImageLastRepeat("testStreets2.png", numTrajectories=numTrajectories,
#                                                                 maxTrajectoryLength =maxTrajectoryLength, numGrid=40,
#                                                                 seed=123, visualize=False)
# [data, mask, nGrid, selectedOrNotSelected, serializedSelected2DMatrix]=DataGeneratorFcn.generateSyntheticDataVariableLengthInputImage("testStreets2.png", numTrajectories=numTrajectories,
#                                                                 maxTrajectoryLength =200, numGrid=40,
#                                                                 seed=123, visualize=False)
# dataDO=np.zeros(((int)(data.shape[0]*2),data.shape[1]),dtype=np.float32)
# counter=0#FOR DEBUGGING, OUTPUT TO FILE
# for r in range(0,data.shape[0]):#FOR DEBUGGING, OUTPUT TO FILE
#     dataDO[counter, :] = data[r, :, 0]#FOR DEBUGGING, OUTPUT TO FILE
#     counter = counter + 1
#     dataDO[counter, :] = data[r, :, 1]#FOR DEBUGGING, OUTPUT TO FILE
#     counter = counter + 1#FOR DEBUGGING, OUTPUT TO FILE
# np.savetxt("debugImageTrajs.csv", dataDO, delimiter=",")
# np.savetxt("debugImageSelectedOrNotSelected.csv", selectedOrNotSelected, delimiter=",")
# np.savetxt("debugImageSerializedSelected2DMatrix.csv", serializedSelected2DMatrix, delimiter=",")




# data1=np.genfromtxt("datasets/debugImageTrajs_updown.csv",delimiter=",",dtype=np.float32, filling_values=np.nan)
# data2=np.zeros(((int)(data1.shape[0]/2),data1.shape[1],2),dtype=np.float32)
# counter=0
# for r in range(0,data1.shape[0],2):
#     data2[counter, :, 0] = data1[r, :]
#     data2[counter, :, 1] = data1[r+1, :]
#     counter = counter + 1
# serializedSelected2DMatrix1=np.genfromtxt("datasets/debugImageSelectedOrNotSelected2DMatrix.csv",delimiter=",",dtype=np.float32, filling_values=np.nan)
# serializedSelected2DMatrix=serializedSelected2DMatrix1
# selectedOrNotSelected=np.genfromtxt("datasets/debugImageSerializedSelected.csv",delimiter=",",dtype=np.float32, filling_values=np.nan)
# selectedOrNotSelected=selectedOrNotSelected.transpose()*scale
# data=data2*scale
# nGrid=selectedOrNotSelected.shape[0]
# maxTrajectoryLength=data.shape[1]




data1=np.genfromtxt("datasets/tucson_mobilityJunctionSampleV4_3_4.csv",delimiter=",",dtype=np.float32, filling_values=np.nan)
data2=np.zeros(((int)(data1.shape[0]/2),data1.shape[1],2),dtype=np.float32)
debugReduction=10#DEBUG
debugLength=40#DEBUG
counter=0
for r in range(0,data1.shape[0],2):
    data2[counter, :, 0] = data1[r+1, :]
    data2[counter, :, 1] = data1[r, :]
    counter = counter + 1
data2=data2[0:0+debugReduction,:,:]#DEBUG
data2=data2[:,0:debugLength,:]#DEBUG
serializedSelected2DMatrix1=np.genfromtxt("datasets/tucson_allowableMatrixV4_3_4.csv",delimiter=",",dtype=np.float32, filling_values=np.nan)
serializedSelected2DMatrix=serializedSelected2DMatrix1
# serializedSelected2DMatrix = np.transpose(serializedSelected2DMatrix)
selectedOrNotSelectedRaw=np.genfromtxt("datasets/tucson_allowableNodesV4_3_4.csv",delimiter=",",dtype=np.float32, filling_values=np.nan)
selectedOrNotSelected2,data3=DataGeneratorFcn.normalizeCellTraj(selectedOrNotSelectedRaw.transpose(),data2)
# debugSize = (int)(data3.shape[0]/1)
# visualize_extent(serializedSelected2DMatrix.transpose(), debugReduction, data2, torch.from_numpy(selectedOrNotSelectedRaw.transpose()), saveName=None)
# visualize_extent(serializedSelected2DMatrix, debugReduction, data3, torch.from_numpy(selectedOrNotSelected2), saveName=None)
selectedOrNotSelected=selectedOrNotSelected2*scale
data=data3*scale
# visualize_extent(serializedSelected2DMatrix, debugReduction, data, torch.from_numpy(selectedOrNotSelected), saveName=None)
# visualize_extent(serializedSelected2DMatrix, debugReduction, data3, torch.from_numpy(selectedOrNotSelected2), saveName=None)
nGrid=selectedOrNotSelected.shape[1]
maxTrajectoryLength=data.shape[1]
initLats=np.float32(data[:,0,0])
initLons=np.float32(data[:,0,1])




meanVal=np.mean(selectedOrNotSelected,axis=1)
varVal=np.std(selectedOrNotSelected,axis=1)*10.0


if isRunOnCPU == False:
    selectedOrNotSelected = torch.from_numpy(selectedOrNotSelected).cuda()
else:
    selectedOrNotSelected = torch.from_numpy(selectedOrNotSelected)

time = datetime.now()#MAIN
np.random.seed(time.minute + time.hour + time.microsecond)#MAIN
# np.random.seed(time.minute + time.hour + time.microsecond)#DEBUG
visualize_extent(serializedSelected2DMatrix, debugReduction, data, selectedOrNotSelected, saveName=None,serializedDirectPoint=selectedOrNotSelected)
np.random.shuffle(data)#MAIN
# visualize_extent(serializedSelected2DMatrix, debugReduction, data, selectedOrNotSelected, saveName=None)
datas.append(data)

selectedOrNotSelecteds.append(selectedOrNotSelected)
serializedSelected2DMatrixs.append(serializedSelected2DMatrix)

for c in range(numCities-1):
    # [data, nGrid, selectedOrNotSelected,
     # serializedSelected2DMatrix] = DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,
     #                                                                                 trajectoryLength=trajectoryLength,
     #                                                                                 numGrid=40, seed=seeds[c],
     #                                                                                 visualize=False)

    [data, nGrid, selectedOrNotSelected, serializedSelected2DMatrix, _]=DataGeneratorFcn.generateSyntheticDataVariableLengthLastRepeat(numTrajectories=numTrajectories,
                                                                longestTrajectory=maxTrajectoryLength, numGrid=40,
                                                                seed=seeds[c], visualize=True)

    if isRunOnCPU == False:
        serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
    else:
        serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

    time = datetime.now()#MAIN
    np.random.seed(time.minute + time.hour + time.microsecond)#MAIN
    np.random.shuffle(data)#MAIN

    datas.append(data)

    selectedOrNotSelecteds.append(selectedOrNotSelected)
    serializedSelected2DMatrixs.append(serializedSelected2DMatrix)


#[data,nGrid,selectedOrNotSelected]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=3,visualize=False)
# DataGeneratorFcn.generateSyntheticDataVariableLength(numTrajectories=100,longestTrajectory=80,numGrid=50,seed=3,visualize=True)



# if isRunOnCPU==False:
#     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
# else:
#     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)

model = MinimalisticModel.SimpleNN(forbiddenSerialMap=selectedOrNotSelected,lenForbidden=selectedOrNotSelected.shape[1],maxLengthSize=maxTrajectoryLength,temporalFeatureSize=2, convOffset=0)
# OLD MODELS
# model = MinimalisticModel.SimpleNN(forbiddenSerialMap=selectedOrNotSelected,lenForbidden=selectedOrNotSelected.shape[1],maxLengthSize=maxTrajectoryLength,temporalFeatureSize=2)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_trainable_params}")

# Visualizing the graph
# Test input for visualization
in1=torch.randn(BATCH_SIZE,maxTrajectoryLength,2)
in2=torch.randn(BATCH_SIZE,1)
cond1=torch.randn(BATCH_SIZE,1)
cond2=torch.randn(BATCH_SIZE,1)
# Forward pass
if isRunOnCPU==False:
    model=model.cuda()
    in1=in1.cuda()
    in2=in2.cuda()
    cond1 = cond1.cuda()
    cond2 = cond2.cuda()
y = model(in1, in2, cond1, cond2)

if isRunOnCPU==False:
    graph = draw_graph(model, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cuda")
else:
    graph = draw_graph(model, input_data=(in1, in2, cond1, cond2), expand_nested=True, device="cpu")

graph.visual_graph.render('model_pytorch'+modelVersion, format="png")

# Loss and optimizer
if "LOSS" in modelVersion:
    criterion = MinimalisticModel.CustomLoss(serializedSelected2DMatrix,serializedSelected2DMatrix.shape[1],maxTrajectoryLength,timesteps)
    # criterion.requires_grad_(True)
else:
    criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=initialLR)
# optimizer = optim.Adadelta(model.parameters(),lr=initialLR)
# optimizer = optim.NAdam(model.parameters(),lr=0.00002)

my_file = Path('model'+modelVersion+'.pytorch')
if my_file.is_file() and continueTrain==True:
    try:
        model.load_state_dict(torch.load('model' + modelVersion + '.pytorch'))
        # optimizer.load_state_dict(torch.load('optim' + modelVersion + '.pytorch'))
        print("MODEL LOADED!")
        # summarize_weights_by_type(model)
        # print("MODEL WEIGHTS!")
    except Exception as e:
        print("FAILED TO LOAD WEGHTS!")
        print(f"{e}")
if isRunOnCPU==False:
    model.cuda()

    # model = tf.keras.models.load_model('model'+modelVersion+'.pytorch')

def train_one(x_img,opt,cr,learningScheduleTime, isAdvancedWeighting=True, isAdvancedExponent=False):
    global isChangeWeights
    x_ts = MinimalisticModel.generate_ts(timesteps, len(x_img), learningScheduleTime, isChangeWeights, isAdvancedWeighting=isAdvancedWeighting)
    # OLD MODEL
    # x_ts = MinimalisticModel.generate_ts(timesteps, len(x_img))#OLD
    # x_ts = MinimalisticModel.generate_ts(timesteps, timesteps)

    # idx=237
    # for h in range(1, 12):
    #     plt.plot([x_img[idx, h - 1, 0], x_img[idx, h, 0]], [x_img[idx, h - 1, 1], x_img[idx, h, 1]], marker='', zorder=2, alpha=0.5, color='b')
    # # plt.show()

    x_a, x_b = MinimalisticModel.forward_noise_notNormalized(meanVal, varVal, timesteps, x_img, x_ts, learningScheduleTime, isChangeWeights,isVizualize=True,isAdvancedWeighting=isAdvancedExponent)
    # OLD MODEL
    # x_a, x_b = MinimalisticModel.forward_noise(timesteps, x_img, x_ts, learningScheduleTime, isChangeWeights,isVizualize=False,isAdvancedWeighting=isAdvancedExponent)
    # x_a, x_b = MinimalisticModel.forward_noise(timesteps, x_img, x_ts, isVizualize=False)
    x_a = torch.from_numpy(x_a).to(torch.float32)
    x_ts = torch.from_numpy(x_ts/timesteps).to(torch.float32)
    x_b = torch.from_numpy(x_b).to(torch.float32)


    # fig, axs = plt.subplots(nrows=2, ncols=(int)(timesteps / 2))
    # for i in range(x_a.shape[0]):
    #     row = (int)(math.floor(i/(timesteps / 2)))
    #     col = (int)(i % (timesteps / 2))
    #     visualiza(selectedOrNotSelected, 16, x_a, axis=axs[row][col])
    #     axs[row][col].title.set_text("Time: "+str(i))
    # plt.show()
    #
    # fig1, axs1 = plt.subplots(nrows=1, ncols=(int)(timesteps / 1))
    # for i in range(x_a.shape[0]):
    #     row = (int)(math.floor(i / (timesteps / 2)))
    #     col = (int)(i % (timesteps / 2))
    #     visualiza(selectedOrNotSelected, 16, x_a, axis=axs[row][col])
    #     axs1[row][col].title.set_text("Time: " + str(i))
    # plt.show()

    x_ts = x_ts.unsqueeze(1)
    cond1 = torch.unsqueeze(torch.from_numpy(x_img[:,0,0]),dim=1).to(torch.float32)
    cond2 = torch.unsqueeze(torch.from_numpy(x_img[:,0,1]),dim=1).to(torch.float32)
    # opt.zero_grad()
    if isRunOnCPU == False:
        outputs = model(x_a.cuda(), x_ts.cuda(), cond1.cuda(), cond2.cuda())
        if "LOSS" in modelVersion:
            loss = cr(outputs, x_b.cuda(), x_ts.cuda())
        else:
            loss = cr(outputs, x_b.cuda())
    else:
        outputs = model(x_a, x_ts, cond1, cond2)
        if "LOSS" in modelVersion:
            loss = cr(outputs, x_b, x_ts)
        else:
            loss = cr(outputs, x_b)
    # loss.requires_grad = True
    loss.backward()
    # opt.step()
    # visualiza(x_a.shape[0], x_a)
    # visualiza(x_b.shape[0], x_b)
    return loss

def train(X_trains, selectedOrNotSelecteds, serializedSelected2DMatrixs, BATCH_SIZE,optimizer_input,cri, offsetWeighting=0, maxWeightingStep=0.1, numIterates=30, extraIterates=0, beforeStepsFixed=0, isVisualizeLoss=False, isAdvancedWeighting=True, isAdvancedExponent=False):
    allLoss=np.zeros(numIterates+extraIterates+beforeStepsFixed)
    bar = trange(numIterates+extraIterates+beforeStepsFixed)
    total = cumulativeIters
    optimizer_input.param_groups[0]['lr']=initialLR
    scheduler1 = ExponentialLR(optimizer_input, gamma=0.997)
    if isVisualizeLoss==True:
        lv=LossVisualizer(numIterates+extraIterates+beforeStepsFixed)
    optimResetCounter=0
    usingLR = initialLR
    for i in bar:
        if i >= beforeStepsFixed+numIterates:
            adjustedScheduleValue=offsetWeighting+maxWeightingStep
        elif i <= beforeStepsFixed:
            # adjustedScheduleValue=0
            adjustedScheduleValue = offsetWeighting
        elif i>=beforeStepsFixed:
            adjustedScheduleValue=math.pow(offsetWeighting+((i-beforeStepsFixed) / numIterates)*maxWeightingStep,1.0)
        # print("adjustedScheduleValue:")
        # print(adjustedScheduleValue)
        if optimResetCounter>((numIterates+1)*1000.8):
            usingLR=usingLR*0.8
            optimizer_input = optim.Adam(model.parameters(), lr=usingLR)
            scheduler1 = ExponentialLR(optimizer_input, gamma=0.997)
            optimResetCounter=0
            print("OPTIMIZER RESET")
        optimResetCounter=optimResetCounter+1
        epoch_loss = 0.0
        optimizer_input.zero_grad()
        for j in range(total):
            # optimizer_input.zero_grad()
            if isCompleteRandomCities==True:
                time = datetime.now()
                seedVal=time.minute + time.hour + time.microsecond
                [data, nGrid, selectedOrNotSelected,
                 serializedSelected2DMatrix,_] = DataGeneratorFcn.generateSyntheticDataVariableLengthLastRepeat(
                    numTrajectories=numTrajectories,
                    longestTrajectory=maxTrajectoryLength,
                    numGrid=40, seed=seedVal,
                    visualize=False)
                # randIndex = np.random.randint(len(X_trains))
                X_train = data
                # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                model.psl.lenForbidden = serializedSelected2DMatrix.shape[1]
                model.pslSum.lenForbidden = serializedSelected2DMatrix.shape[1]
                if isRunOnCPU == False:
                    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
                else:
                    serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
                model.psl.B = serializedSelected2DMatrix
                model.pslSum.B = serializedSelected2DMatrix
                cri.forbiddens = serializedSelected2DMatrix
                cri.lenForbidden = serializedSelected2DMatrix.shape[1]
                # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                np.random.seed(time.minute + time.hour + time.microsecond)  # MAIN
                x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
                loss = train_one(x_img, optimizer_input, cri, adjustedScheduleValue,isAdvancedWeighting=isAdvancedWeighting,isAdvancedExponent=isAdvancedExponent)
                epoch_loss += loss.item() * 1
            else:
                for d in range(numCitiesTrain):
                    randIndex = np.random.randint(len(X_trains))
                    # time = datetime.now()  # REDUNDANT
                    # np.random.seed(time.minute + time.hour + time.microsecond)  # REDUNDANT
                    # np.random.shuffle(X_trains[randIndex])  # REDUNDANT
                    X_train = X_trains[randIndex]
                    # model.sforbiddenSerialMap = selectedOrNotSelecteds[d]
                    model.psl.lenForbidden = selectedOrNotSelecteds[d].shape[1]
                    model.psl.B = selectedOrNotSelecteds[d]
                    model.pslSum.lenForbidden = selectedOrNotSelecteds[d].shape[1]
                    model.pslSum.B = selectedOrNotSelecteds[d]
                    cri.forbiddens=selectedOrNotSelecteds[d]
                    cri.lenForbidden=selectedOrNotSelecteds[d].shape[1]

                    # model.lenForbidden = serializedSelected2DMatrixs[d].shape[1]
                    batchIndices=np.random.randint(len(X_train), size=BATCH_SIZE)
                    x_img = X_train[batchIndices]
                    loss = train_one(x_img, optimizer_input, cri, adjustedScheduleValue,isAdvancedWeighting=isAdvancedWeighting,isAdvancedExponent=isAdvancedExponent)
                    # optimizer_input.step()
                    epoch_loss += loss.item() * 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer_input.step()
            #pg = (j / total) * 100
        # optimizer_input.step()
        if i % 3 == 0:
            bar.set_description(f'loss: {epoch_loss/total:.5f}, lr: {scheduler1.get_last_lr()[0]:.9f}, ASV: {adjustedScheduleValue}')
        scheduler1.step()
        allLoss[i]=epoch_loss/total
        if isVisualizeLoss == True:
            lv.values[i] = epoch_loss/total
    # model.eval()
    torch.save(model.state_dict(), 'model' + modelVersion + '.pytorch')
    torch.save(optimizer_input.state_dict(), 'optim' + modelVersion + '.pytorch')
    print("MODEL SAVED!")
    if isVisualizeLoss == True:
        lv.visualize(saveFileName='Loss_pytorch_' + modelVersion + '.png',
                     titleText="Pytorch loss value over iterations. Model " + modelVersion + ".")
    return allLoss

if isTrainModel==True:
    allLossValues=[]
    maxOffsetMovement=0.0
    for r in range(repeatTimes):
        print(f"repeatTimes: {r}")
        offsetValues = np.arange(np.minimum(fixedOffset+(r/repeatTimes)*maxOffsetMovement,0.9999), 1, weightingStepSize)
        print(offsetValues)
        for s in range(offsetValues.shape[0]):
            stepSize=np.minimum(weightingStepSize,1-offsetValues[s])
            print(stepSize)
            lossValuesRes=train(datas, selectedOrNotSelecteds, serializedSelected2DMatrixs, BATCH_SIZE, optimizer, criterion, offsetWeighting=offsetValues[s], maxWeightingStep=stepSize, numIterates=numOptIterates, extraIterates=extraStepsFixed, beforeStepsFixed=beforeStepsFixed, isVisualizeLoss=False, isAdvancedWeighting=True, isAdvancedExponent=False)
            # lossValuesRes = train(datas, selectedOrNotSelecteds, serializedSelected2DMatrixs, BATCH_SIZE, optimizer,
            #                       criterion, offsetWeighting=offsetValues[s], maxWeightingStep=stepSize,
            #                       numIterates=numOptIterates, extraIterates=extraStepsFixed,
            #                       beforeStepsFixed=beforeStepsFixed, isVisualizeLoss=False, isAdvancedWeighting=True,
            #                       isAdvancedExponent=False)
            # lossValuesRes = train(datas, selectedOrNotSelecteds, serializedSelected2DMatrixs, BATCH_SIZE, optimizer,
            #                       criterion, offsetWeighting=offsetValues[s], maxWeightingStep=stepSize,
            #                       numIterates=numOptIterates, extraIterates=extraStepsFixed,
            #                       beforeStepsFixed=beforeStepsFixed, isVisualizeLoss=False)
            allLossValues.append(lossValuesRes)
        initialLR = initialLR * 0.8
    resultAllLoss = np.concatenate(allLossValues)
    plt.plot(resultAllLoss)
    plt.title("All loss values during training")
    plt.savefig(modelVersion+"_LossValues.png")
    plt.show()


def predict(serializedSelected, trajectoryLength,numTraj=10):
    x = np.random.normal(loc=meanVal,scale=varVal,size=(numTraj, trajectoryLength, 2))
    # OLD
    # x = np.random.normal(loc=0.5, scale=0.5, size=(numTraj, trajectoryLength, 2))
    # x = np.random.normal(loc=0.5,scale=0.33,size=(numTraj, trajectoryLength, 2))
    # x = np.random.uniform(low=0, high=1, size=(numTraj, trajectoryLength, 2))

    for idx in range(x.shape[0]):#MAIN
        for h in range(1, maxTrajectoryLength):#MAIN
            plt.plot([x[idx, h - 1, 0], x[idx, h, 0]], [x[idx, h - 1, 1], x[idx, h, 1]], marker='',
                     zorder=2, alpha=0.5, color='g')#MAIN
    plt.show()#MAIN

    x = torch.from_numpy(x).to(torch.float32)
    model.psl.lenForbidden = serializedSelected.shape[1]
    model.pslSum.lenForbidden = serializedSelected.shape[1]
    # if isRunOnCPU == False:
    #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix).cuda()
    # else:
    #     serializedSelected2DMatrix = torch.from_numpy(serializedSelected2DMatrix)
    model.psl.B = serializedSelected
    model.pslSum.B = serializedSelected

    cmap_name = 'jet'  # Example: Use the 'jet' colormap
    cmap = cm.get_cmap(cmap_name, timesteps)

    # fig, axs = plt.subplots(nrows=1, ncols=(int)(timesteps/1))
    with torch.no_grad():
        # indices = np.random.choice(serializedSelected.shape[1], numTraj, replace=True)
        # samples = serializedSelected[:,indices]
        # samples=torch.transpose(samples,0,1)
        # cond1 = torch.unsqueeze(samples[:, 0], dim=1)
        # cond2 = torch.unsqueeze(samples[:, 1], dim=1)

        indices = np.random.choice(initLats.shape[0], numTraj, replace=True)
        cond1 = np.zeros((numTraj,1))
        cond2 = np.zeros((numTraj, 1))
        for i in range(numTraj):
            cond1[i, 0] = initLats[indices[i]]
            cond2[i, 0] = initLons[indices[i]]
        cond1 = torch.from_numpy(cond1).to(torch.float32)
        cond2 = torch.from_numpy(cond2).to(torch.float32)

        # cond1 = torch.full((numTraj, 1), initLat)
        # cond2 = torch.full((numTraj, 1), initLon)
        for i in trange(timesteps):
            color = cmap(i)
            resX = x.cpu().detach().numpy()
            for idx in range(x.shape[0]):#MAIN
                for h in range(1, maxTrajectoryLength):#MAIN
                    plt.plot([resX[idx, h - 1, 0], resX[idx, h, 0]], [resX[idx, h - 1, 1], resX[idx, h, 1]], marker='',
                             zorder=2, alpha=0.5, color=color)#MAIN

            ## colVal=i%(int)(timesteps/2)
            ## rowVal=math.floor(i/(int)(timesteps/2))
            # cond1 = torch.unsqueeze(x[:, 0, 0], dim=1)
            # cond2 = torch.unsqueeze(x[:, 0, 1], dim=1)
            t = i
            x_ts = np.full((numTraj), t)
            x_ts = torch.from_numpy(x_ts).to(torch.float32)
            x_ts = x_ts.unsqueeze(1)
            if isRunOnCPU == False:
                x_res = model(x.cuda(),x_ts.cuda(), cond1.cuda(), cond2.cuda())
                x = x_res
            else:
                x_res = model(x, x_ts, cond1, cond2)
                x = x_res
    #         visualiza(selectedOrNotSelected, numTraj, x.cpu().detach().numpy(), axis=axs[i])
    #         axs[i].title.set_text("Time: "+str(i))

    plt.savefig(modelVersion + "_Results_gradient.png")
    plt.show()#MAIN
    return x

visualize_extent(serializedSelected2DMatrixs[0], debugReduction, datas[0], selectedOrNotSelecteds[0], saveName=None)
numPredicts = 50
for c in range(numCities):
    pred = predict(selectedOrNotSelecteds[c], maxTrajectoryLength,numTraj=numPredicts)

    pred = pred.cpu().detach().numpy()

    # JSDValue = JSD(nGrid,datas[c],pred,serializedSelected2DMatrixs[c], scale=scale)

    # print("JSDValue")
    # print(JSDValue.JSDValue)

    # JSDValue_SingleB = JSD_SingleB(nGrid,datas[c],pred,serializedSelected2DMatrixs[c])

    # print("JSDValue_singleB")
    # print(JSDValue_SingleB.JSDValue)
    # print("B value")
    # print(JSDValue_SingleB.minBValue)

    visualize_extent(serializedSelected2DMatrixs[c],numPredicts, pred, selectedOrNotSelecteds[c], saveName=modelVersion)

mainSelectedOrNotSelected = serializedSelected2DMatrixs[0]

# FANCY TEST! CHANGE THE CITY AND TEST!!!
# [dataT,nGrid,selectedOrNotSelectedT,serializedSelected2DMatrixT,_]=DataGeneratorFcn.generateSyntheticDataFixedLength(numTrajectories=numTrajectories,trajectoryLength=trajectoryLength,numGrid=40,seed=1,visualize=False)
newNumGrid=40
[data, nGrid, selectedOrNotSelectedT, serializedSelected2DMatrixT]=DataGeneratorFcn.generateSyntheticDataVariableLengthInputImageLastRepeat("testStreets3.png", numTrajectories=numTrajectories,
                                                                maxTrajectoryLength=maxTrajectoryLength, numGrid=newNumGrid,
                                                                seed=123, visualize=False)
serializedSelected2DMatrixT=serializedSelected2DMatrixT*scale
data=data*scale

# serializedSelected2DMatrixT=serializedSelected2DMatrixT[:,0:302]
if isRunOnCPU==False:
    serializedSelected2DMatrixT=torch.from_numpy(serializedSelected2DMatrixT).cuda()
else:
    serializedSelected2DMatrixT = torch.from_numpy(serializedSelected2DMatrixT)
model.psl.B=serializedSelected2DMatrixT
model.pslSum.B=serializedSelected2DMatrixT

numPredicts = 20
pred = predict(serializedSelected2DMatrixT, maxTrajectoryLength,numTraj=numPredicts)

pred = pred.cpu().detach().numpy()

JSDValue = JSD(newNumGrid,data,pred,selectedOrNotSelectedT)

print("JSDValue other city: ")
print(JSDValue.JSDValue)

JSDValue_SingleB = JSD_SingleB(newNumGrid,data,pred,selectedOrNotSelected)

print("JSDValue_singleB other city: ")
print(JSDValue_SingleB.JSDValue)
print("B value other city: ")
print(JSDValue_SingleB.minBValue)

visualize_extent(selectedOrNotSelectedT,numPredicts, pred, serializedSelected2DMatrixT, auxSelectedOrNotSelected=mainSelectedOrNotSelected, saveName=modelVersion+"_anotherCity_")

#torch.save(model.state_dict(), 'model'+modelVersion+'.pytorch')

print("!!!")
