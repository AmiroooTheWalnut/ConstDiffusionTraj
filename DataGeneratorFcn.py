import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from QualityMeasure import JSD
from PIL import Image
from datetime import datetime

def generateSyntheticDataVariableLength(numTrajectories=5,longestTrajectory=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, longestTrajectory, 2))
    mask = np.zeros((numTrajectories, longestTrajectory, 2))
    np.random.seed(seed)

    nGrid = numGrid

    selectedOrNotSelected = np.zeros((nGrid, nGrid))
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if np.random.rand() < 0.2:
                # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})

    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix, dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i = 0
    maxTry = 10
    tryCounter = 0
    while i < numTrajectories:
        isTrajGenerated = True
        trajectoryLength = np.random.randint(4, high=longestTrajectory)
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated = False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated == True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid - 1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid - 1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
                mask[i, h, 0] = 1
                mask[i, h, 1] = 1
            i = i + 1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break

    if visualize == True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, genData.shape[1]):
                if genData[i, h, 0] > -1:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                   extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                   origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    serializedSelectedList = serializedSelected
    return [genData, mask, nGrid, selectedOrNotSelected, serializedSelected2DMatrix, serializedSelectedList]

def generateSyntheticDataVariableLengthLastRepeat(numTrajectories=5,longestTrajectory=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, longestTrajectory, 2))
    np.random.seed(seed)

    nGrid = numGrid

    selectedOrNotSelected = np.zeros((nGrid, nGrid))
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if np.random.rand() < 0.2:
                # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})

    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix, dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i = 0
    maxTry = 10
    tryCounter = 0
    while i < numTrajectories:
        isTrajGenerated = True
        trajectoryLength = np.random.randint(4, high=longestTrajectory)
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated = False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated == True:
            tryCounter = 0
            lastLat = 0
            lastLon = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid - 1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid - 1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
                lastLat = x
                lastLon = y
            for h in range(longestTrajectory):
                if genData[i, h, 0]==-1:
                    genData[i, h, 0] = lastLat
                    genData[i, h, 1] = lastLon
            i = i + 1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break

    if visualize == True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, genData.shape[1]):
                if genData[i, h, 0] > -1:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(), cmap="cool",
                   extent=(0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
                   origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    serializedSelectedList = serializedSelected
    return [genData, nGrid, selectedOrNotSelected, serializedSelected2DMatrix, serializedSelectedList]

def generateSyntheticDataFixedLength(numTrajectories=5,trajectoryLength=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, trajectoryLength, 2)) - 1
    np.random.seed(seed)

    nGrid = numGrid

    selectedOrNotSelected = np.zeros((nGrid, nGrid))
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if np.random.rand() < 0.2:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, trajectoryLength):
                if genData[i, h, 0] > -1:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    serializedSelectedList = serializedSelected
    return [genData,nGrid,selectedOrNotSelected,serializedSelected2DMatrix,serializedSelectedList]

def genOnlyTrajFixedLengthRandom(nGrid,numTrajectories,trajectoryLength,serializedSelected,selectedOrNotSelected):
    time = datetime.now()
    np.random.seed(time.minute + time.hour + time.microsecond)
    genData = np.zeros((numTrajectories, trajectoryLength, 2)) - 1
    i = 0
    maxTry = 10
    tryCounter = 0
    while i < numTrajectories:
        isTrajGenerated = True
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated = False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated == True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid - 1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid - 1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
            i = i + 1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break

    return genData

def trajToGrid(numGrids, generatedTrajs):
    gridData = np.zeros((numGrids, numGrids))
    for i in range(generatedTrajs.shape[0]):
        for j in range(generatedTrajs.shape[1]):
            gridX = np.floor(generatedTrajs[i, j, 0] * numGrids)
            gridY = np.floor(generatedTrajs[i, j, 1] * numGrids)
            if gridX>numGrids-1:
                gridX=numGrids-1
            if gridX<0:
                gridX=0
            if gridY>numGrids-1:
                gridY=numGrids-1
            if gridY<0:
                gridY=0
            gridData[int(gridX),int(gridY)] = gridData[int(gridX),int(gridY)]+1
    return gridData

def generateSyntheticDataFixedLengthInputImage(input, numTrajectories=5,trajectoryLength=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, trajectoryLength, 2)) - 1
    np.random.seed(seed)

    nGrid = numGrid

    img = Image.open(input)
    pixels = img.load()
    selectedOrNotSelected = np.zeros((nGrid, nGrid))
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if pixels[i,j][0] < 200:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, trajectoryLength):
                if genData[i, h, 0] > -1:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    return [genData,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]



def generateSyntheticDataVariableLengthInputImage(input, numTrajectories=5,maxTrajectoryLength=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, maxTrajectoryLength, 2))-1
    mask = np.zeros((numTrajectories, maxTrajectoryLength, 2))
    np.random.seed(seed)

    nGrid = numGrid

    img = Image.open(input)
    pixels = img.load()
    selectedOrNotSelected = np.zeros((nGrid, nGrid), dtype=np.float32)
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if pixels[i,j][0] < 200:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        trajectoryLength = np.random.randint(4,high=maxTrajectoryLength)
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
                mask[i, h, 0] = 1
                mask[i, h, 1] = 1
            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, genData.shape[1]):
                if mask[i, h, 0] > 0:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    return [genData,mask,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]

def generateSyntheticDataVariableLengthInputImageLastRepeat(input, numTrajectories=5,maxTrajectoryLength=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, maxTrajectoryLength, 2))-1
    np.random.seed(seed)

    nGrid = numGrid

    img = Image.open(input)
    pixels = img.load()
    selectedOrNotSelected = np.zeros((nGrid, nGrid), dtype=np.float32)
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if pixels[i,j][0] < 200:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        trajectoryLength = np.random.randint(4,high=maxTrajectoryLength)
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                # selectedNeighbor = 0
                # for m in range(len(listOfNeighbourIndices)):
                #     if x<listOfNeighbourIndices[m][0]:
                #         selectedNeighbor=m
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            lastLat = 0
            lastLon = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
                lastLat = x
                lastLon = y
            for h in range(maxTrajectoryLength):
                if genData[i, h, 0]==-1:
                    genData[i, h, 0] = lastLat
                    genData[i, h, 1] = lastLon

            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, genData.shape[1]):
                color = cmap(i)
                plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                         color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    return [genData,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]


def generateSyntheticDataVariableLengthInputImageLastRepeatDebug(input, numTrajectories=5,maxTrajectoryLength=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, maxTrajectoryLength, 2))-1
    np.random.seed(seed)

    nGrid = numGrid

    img = Image.open(input)
    pixels = img.load()
    selectedOrNotSelected = np.zeros((nGrid, nGrid), dtype=np.float32)
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if pixels[i,j][0] < 200:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        trajectoryLength = np.random.randint(4,high=maxTrajectoryLength)
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        leftStarts=[]
        for m in range(len(serializedSelected)):
            if list(serializedSelected[m].values())[0][0]==0:
                leftStarts.append(serializedSelected[m])
        # startIndex = np.random.randint(len(serializedSelected))
        startIndex = np.random.randint(len(leftStarts))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                # selectedNeighbor = 0
                found=False
                for m in range(len(listOfNeighbourIndices)):
                    if x<listOfNeighbourIndices[m][0]:
                        selectedNeighbor=m
                        found=True
                if found==False:
                    for m in range(len(listOfNeighbourIndices)):
                        if y < listOfNeighbourIndices[m][1]:
                            selectedNeighbor = m
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            lastLat = 0
            lastLon = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
                lastLat = x
                lastLon = y
            for h in range(maxTrajectoryLength):
                if genData[i, h, 0]==-1:
                    genData[i, h, 0] = lastLat
                    genData[i, h, 1] = lastLon

            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, genData.shape[1]):
                color = cmap(i)
                plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                         color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    return [genData,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]



def generateSyntheticDataFixedLengthInputImageLeftToRight(input, numTrajectories=5,trajectoryLength=10,numGrid=10,seed=0,visualize=False):
    genData = np.zeros((numTrajectories, trajectoryLength, 2)) - 1
    np.random.seed(seed)

    nGrid = numGrid

    img = Image.open(input)
    pixels = img.load()
    selectedOrNotSelected = np.zeros((nGrid, nGrid))
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if pixels[i,j][0] < 200:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        for j in range(1, trajectoryLength):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRaw[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                found = False
                for m in range(len(listOfNeighbourIndices)):
                    if x < listOfNeighbourIndices[m][0]:
                        selectedNeighbor = m
                        found = True
                if found == False:
                    for m in range(len(listOfNeighbourIndices)):
                        if y < listOfNeighbourIndices[m][1]:
                            selectedNeighbor = m
                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRaw[j] = m
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, trajectoryLength):
                if genData[i, h, 0] > -1:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    return [genData,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]

def generateSyntheticDataFixedLengthInputImageLeftToRightSparse(input, numTrajectories=5,trajectoryLength=10,numGrid=10,seed=0,minSkip=0,maxSkip=10,visualize=False):
    genData = np.zeros((numTrajectories, trajectoryLength, 2)) - 1
    np.random.seed(seed)

    nGrid = numGrid

    img = Image.open(input)
    pixels = img.load()
    selectedOrNotSelected = np.zeros((nGrid, nGrid))
    serializedSelected = []
    for i in range(nGrid):
        for j in range(nGrid):
            if pixels[i,j][0] < 200:
            # if j < nGrid/2:
                selectedOrNotSelected[i, j] = 1
    for i in range(nGrid):
        for j in range(nGrid):
            if selectedOrNotSelected[i, j] == 1:
                listOfNeighbourIndices = []
                if i - 1 > -1 and j - 1 > -1:
                    if selectedOrNotSelected[i - 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j - 1))
                if i - 1 > -1:
                    if selectedOrNotSelected[i - 1, j] == 1:
                        listOfNeighbourIndices.append((i - 1, j))
                if i - 1 > -1 and j + 1 < nGrid:
                    if selectedOrNotSelected[i - 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i - 1, j + 1))

                if j - 1 > -1:
                    if selectedOrNotSelected[i, j - 1] == 1:
                        listOfNeighbourIndices.append((i, j - 1))
                if j + 1 < nGrid:
                    if selectedOrNotSelected[i, j + 1] == 1:
                        listOfNeighbourIndices.append((i, j + 1))

                if i + 1 < nGrid and j - 1 > -1:
                    if selectedOrNotSelected[i + 1, j - 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j - 1))
                if i + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j] == 1:
                        listOfNeighbourIndices.append((i + 1, j))
                if i + 1 < nGrid and j + 1 < nGrid:
                    if selectedOrNotSelected[i + 1, j + 1] == 1:
                        listOfNeighbourIndices.append((i + 1, j + 1))

                if len(listOfNeighbourIndices) == 0:
                    selectedOrNotSelected[i, j] = 0
                else:
                    serializedSelected.append({i * nGrid + j: (i, j)})


    serializedSelected2DMatrix = []
    for i in range(len(serializedSelected)):
        serializedSelected2DMatrix.append(list(list(serializedSelected[i].values())[0]))
    serializedSelected2DMatrix = np.array(serializedSelected2DMatrix,dtype=np.float32).transpose()
    serializedSelected2DMatrix[0, :] = serializedSelected2DMatrix[0, :] / (nGrid - 1)
    serializedSelected2DMatrix[1, :] = serializedSelected2DMatrix[1, :] / (nGrid - 1)
    i=0
    maxTry=10
    tryCounter=0
    while i < numTrajectories:
        isTrajGenerated = True
        # if i == 176:
        #     print('DEBUG SEVERE ISSUE!!!')
        startIndex = np.random.randint(len(serializedSelected))
        offset=0
        allOffsets=np.zeros((trajectoryLength-1))
        for m in range(trajectoryLength-1):
            o=np.random.randint(minSkip, high=maxSkip)
            offset=offset+o
            allOffsets[m]=o
        oneFrameRaw = np.zeros(trajectoryLength) - 1
        oneFrameRaw[0] = startIndex
        oneFrameRawCounter=1
        oneFrameRawWithOffset = np.zeros(trajectoryLength + offset) - 1
        oneFrameRawWithOffset[0] = startIndex
        offsetCounter=0
        offsetIndex=0
        for j in range(1, trajectoryLength+offset):
            listOfNeighbourIndices = []
            x = list(serializedSelected[int(oneFrameRawWithOffset[j - 1])].values())[0][0]
            y = list(serializedSelected[int(oneFrameRawWithOffset[j - 1])].values())[0][1]
            if x - 1 > -1 and y - 1 > -1:
                if selectedOrNotSelected[x - 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y - 1))
            if x - 1 > -1:
                if selectedOrNotSelected[x - 1][y] == 1:
                    listOfNeighbourIndices.append((x - 1, y))
            if x - 1 > -1 and y + 1 < nGrid:
                if selectedOrNotSelected[x - 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x - 1, y + 1))

            if y - 1 > -1:
                if selectedOrNotSelected[x][y - 1] == 1:
                    listOfNeighbourIndices.append((x, y - 1))
            if y + 1 < nGrid:
                if selectedOrNotSelected[x][y + 1] == 1:
                    listOfNeighbourIndices.append((x, y + 1))

            if x + 1 < nGrid and y - 1 > -1:
                if selectedOrNotSelected[x + 1][y - 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y - 1))
            if x + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y] == 1:
                    listOfNeighbourIndices.append((x + 1, y))
            if x + 1 < nGrid and y + 1 < nGrid:
                if selectedOrNotSelected[x + 1][y + 1] == 1:
                    listOfNeighbourIndices.append((x + 1, y + 1))

            if len(listOfNeighbourIndices) == 0:
                print("TRAPPED TRAJECTORY, SKIPPED!")
                isTrajGenerated=False
                break
            else:
                selectedNeighbor = np.random.randint(len(listOfNeighbourIndices))
                found = False
                for m in range(len(listOfNeighbourIndices)):
                    if x < listOfNeighbourIndices[m][0]:
                        selectedNeighbor = m
                        found = True
                if found == False:
                    for m in range(len(listOfNeighbourIndices)):
                        if y < listOfNeighbourIndices[m][1]:
                            selectedNeighbor = m

                for m in range(len(serializedSelected)):
                    if list(serializedSelected[m].values())[0][0] == listOfNeighbourIndices[selectedNeighbor][0] and \
                            list(serializedSelected[m].values())[0][1] == listOfNeighbourIndices[selectedNeighbor][1]:
                        oneFrameRawWithOffset[j] = m
                        if offsetCounter>=allOffsets[offsetIndex]:
                            oneFrameRaw[oneFrameRawCounter] = m
                            offsetIndex=offsetIndex+1
                            offsetCounter=0
                            oneFrameRawCounter=oneFrameRawCounter+1
                        else:
                            offsetCounter=offsetCounter+1
                        break

        if isTrajGenerated==True:
            tryCounter = 0
            for h in range(len(oneFrameRaw)):
                x = list(serializedSelected[int(oneFrameRaw[h])].values())[0][0] / (nGrid-1)
                y = list(serializedSelected[int(oneFrameRaw[h])].values())[0][1] / (nGrid-1)
                genData[i, h, 0] = x
                genData[i, h, 1] = y
            i=i+1
        else:
            tryCounter = tryCounter + 1
            if tryCounter > maxTry:
                print("SEVERE ERROR: THERE IS NO NEIGHBOR IN MULTIPLE GENERATED TRAJECTORIES!")
                print("DATA GENERATION  HALTED!!!")
                break


    if visualize==True:
        # np.random.shuffle(genData)
        cmap_name = 'viridis'  # Example: Use the 'viridis' colormap
        cmap = cm.get_cmap(cmap_name, numTrajectories)
        for i in range(numTrajectories):
            for h in range(1, trajectoryLength):
                if genData[i, h, 0] > -1:
                    color = cmap(i)
                    plt.plot([genData[i, h - 1, 0], genData[i, h, 0]], [genData[i, h - 1, 1], genData[i, h, 1]],
                             color=color, marker='o')
        plt.imshow(selectedOrNotSelected.transpose(),cmap="cool",extent=(0-(1/nGrid)/2,1+(1/nGrid)/2,0-(1/nGrid)/2,1+(1/nGrid)/2),origin='lower')
        # plt.grid(True)
        # The commented code below is for debugging to sure that the traj to grid
        # code works fine. It's used in the quality measurement code later.
        # reversed_grid=trajToGrid(nGrid, genData)
        # plt.imshow(reversed_grid.transpose(), cmap="Oranges",
        #           extent=(
        #           0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2, 0 - (1 / nGrid) / 2, 1 + (1 / nGrid) / 2),
        #           origin='lower', alpha=0.5)
        plt.show()

    return [genData,nGrid,selectedOrNotSelected,serializedSelected2DMatrix]

def normalizeCellTraj(serializedAllowed, trajs):
    outSerializedAllowed = np.zeros(serializedAllowed.shape, dtype=np.float32)
    outTrajs = np.zeros(trajs.shape, dtype=np.float32)
    minX = serializedAllowed[0, :].min()
    maxX = serializedAllowed[0, :].max()
    minY = serializedAllowed[1, :].min()
    maxY = serializedAllowed[1, :].max()

    outTrajs[:, :, 0] = (trajs[:, :, 0] - minX) / (maxX - minX)
    outTrajs[:, :, 1] = (trajs[:, :, 1] - minY) / (maxY - minY)
    outSerializedAllowed[0, :] = (serializedAllowed[0, :] - minX) / (maxX - minX)
    outSerializedAllowed[1, :] = (serializedAllowed[1, :] - minY) / (maxY - minY)

    return outSerializedAllowed, outTrajs