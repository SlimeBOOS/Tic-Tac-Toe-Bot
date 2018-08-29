import numpy as np
import random
import cv2
def Sigmoid(x,deriv=False):
    if deriv:
        tmp = Sigmoid(x)
        return tmp - (1 - tmp)
    else:
        return 1/(1+np.exp(-x))

def AltSigmoid(x,deriv=False):
    if deriv:
       return (9.8*np.exp(4.9*x))/(np.exp(4.9*x)+1)**2
    else:
        return 2/(1+np.exp(-4.9*x))-1

def Nothing(x,deriv=False):
    return 0

def CreateRandomMatrix(rows,cols):
    randomList = list(list(random.random()*2-1 for i in range(cols)) for i in range(rows))
    randomMatrix = np.matrix(randomList)
    return randomMatrix

# NEEDS REFACTORIZING!
# Merge the output layer with the hidden layer (and input layer if needed)
class NN:
    def __init__(self,inputSize=1, hiddenSize=[1], outputSize=1, fileName=None):
        # Create skeletons for the input layers
        self.inputLayer = None
        self.hiddenLayer = None
        self.outputLayer = None

        self.activation = Sigmoid

        self.hiddenWeights = list()
        self.hiddenBias = list()
        
        # Load weights and biases from a file
        if (fileName != None):
            self.loadFromFile(fileName)
            # Dont load anythng else, just exit
            return
        
        # inputSize(Int) - Number of input nodes
        # hiddenSize(list(Int)) - A list of numbers for the amount of hidden nodes for each layer 
        # outputSize(Int) - Number of output nodes
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = 0.05

        # Create matrixes for the output layer
        self.outputWeights = CreateRandomMatrix(outputSize, hiddenSize[-1])
        self.outputBias = CreateRandomMatrix(outputSize,1)

        # Create matrixes for the hidden layers
        for i in range(len(self.hiddenSize)):
            if (i == 0):
                weightsList = CreateRandomMatrix(hiddenSize[i], inputSize)
            else:
                weightsList = CreateRandomMatrix(hiddenSize[i], hiddenSize[i-1])

            biasList = CreateRandomMatrix(hiddenSize[i], 1)

            self.hiddenWeights.append(weightsList)
            self.hiddenBias.append(biasList)

    def loadFromFile(self,fileName):
        with open(fileName,"r") as f:
            data = list(list(float(j) for j in k.split(",")) for k in f.readline().split(";"))
            
            self.inputSize = int(data[0][0])
            self.hiddenSize = list(int(k) for k in data[1])
            self.outputSize = int(data[2][0])
            self.learningRate = data[3][0]
            
            for i in range(len(self.hiddenSize)):
                index = i*2 + 4
                if i == 0:
                    weightsArray = np.array(data[index]).reshape(self.hiddenSize[i], self.inputSize)
                else:
                    weightsArray = np.array(data[index]).reshape(self.hiddenSize[i], self.hiddenSize[i-1])
                
                biasArray = np.array(data[index+1]).reshape(self.hiddenSize[i],1)

                weightsList = np.matrix(weightsArray)
                biasList = np.matrix(biasArray)
                
                self.hiddenWeights.append(weightsList)
                self.hiddenBias.append(biasList)

            self.outputWeights = data[-2]
            self.outputBias = data[-1]
            
    def saveToFile(self,fileName):
        with open(fileName, "w") as f:
            f.write("{};{};{};{}".format(self.inputSize, ",".join(str(k) for k in self.hiddenSize), self.outputSize,self.learningRate))
            for i in range(len(self.hiddenWeights)):
                text = ";" + ",".join(str(k) for k in self.hiddenWeights[i].flatten().tolist()[0])
                f.write(text)
                text = ";" + ",".join(str(k) for k in self.hiddenBias[i].flatten().tolist()[0])
                f.write(text)
                
            text = ";" + ",".join(str(k) for k in self.outputWeights.flatten().tolist()[0])
            f.write(text)
            text = ";" + ",".join(str(k) for k in self.outputBias.flatten().tolist()[0])
            f.write(text+"\n")
            
    def forward(self, inputs):
        self.inputLayer = np.matrix(inputs).transpose()
        self.hiddenLayer = list()
        for i in range(len(self.hiddenSize)):
            if (i == 0):
                self.hiddenLayer.append(self.activation(self.hiddenWeights[i] * self.inputLayer + self.hiddenBias[i]))
            else:
                self.hiddenLayer.append(self.activation(self.hiddenWeights[i] * self.hiddenLayer[i-1] + self.hiddenBias[i]))
        
        self.outputLayer = self.activation(self.outputWeights * self.hiddenLayer[-1] + self.outputBias)

        return self.outputLayer.tolist()[0]

    def backpropagation(self, inputs, answers):
        # Feed forward
        self.forward(inputs)

        # Create answer matrix
        answersLayer = np.matrix(answers)

        # Calculate the error = answers - outputs
        outputError = answersLayer - self.outputLayer
        hiddenErrors = list()
        for i in range(-1, -len(self.hiddenSize)-1, -1):
            if i == -1:
                hiddenErrors.append(np.dot(self.outputWeights.transpose(), outputError))
            else:
                hiddenErrors.append(np.dot(self.hiddenWeights[i+1].transpose(), hiddenErrors[i+1]))
        
        # Because i was adding the matrixes i have to reverse the whole list to get the right order
        hiddenErrors.reverse()
        
        # Calculate Deltas and adjust weights by them
        for i in range(len(self.hiddenSize)+1):
            if i == len(self.hiddenSize):
                delta = np.multiply(self.outputLayer, 1-self.outputLayer)
                delta = np.multiply(delta, outputError)
                delta = np.multiply(delta, self.learningRate)
                self.outputBias += delta
                self.outputWeights += np.dot(delta, self.hiddenLayer[-1].transpose())
            else:
                delta = np.multiply(self.hiddenLayer[i], 1-self.hiddenLayer[i])
                delta = np.multiply(delta, hiddenErrors[i])
                delta = np.multiply(delta, self.learningRate)
                self.hiddenBias[i] += delta
                if i == 0:
                    self.hiddenWeights[i] += np.dot(delta, self.inputLayer.transpose())
                else:
                    self.hiddenWeights[i] += np.dot(delta, self.hiddenLayer[i-1].transpose())

class NEAT:
    class NodeGene:
        def  __init__(self, index = 0, nodeType = "Hidden"):
            self.index = index
            self.type = nodeType

        def __repr__(self):
            return "NodeGene({},{})".format(self.index, self.type)

    class ConnectionGene:
        def __init__(self, connectionInput, connectionOutput, innov):
            self.input = connectionInput
            self.output = connectionOutput
            self.weight = random.random()*2-1
            self.enabled = True
            self.innov = innov

        def __repr__(self):
            return "ConnectionNode({},{},{})".format(self.input,self.output,self.innov)

    def __init__(self, param1=None, param2=None, fileName=None):
        self.nodeGenes = []
        self.connectionGenes = []
        self.mutationRate = 0.01
        self.connectionMutationChance = 0.3
        self.nodeMutationChance = 0.4
        self.adjustWeightMutationChance = 0.4
        self.nodeEnableMutationChance = 0.25
        self.nodeDisableMutationChance = 0.3
        self.innovation = 0
        self.nodeIndex = 0
        self.activation = Sigmoid
        self.fitness = 0

        if (fileName != None): # If filename is provided load from file
            self.loadFromFile(fileName)
            # Because I dont wanna apply a mutation
            return
        elif (type(param1) == int and type(param2) == int): # If param1 and param2 are both int: Intialize the NEAT network will random values
            inputSize = param1
            outputSize = param2

            # Intialize input genes
            for i in range(inputSize):
                self.addNode("Input")

            #Initialize the bias
            self.addNode("Bias")

            # Intialize output genes
            for i in range(outputSize):
                self.addNode("Output")

            # Initialize default connections
            for i in self.getNodes("Input") + self.getNodes("Bias"):
                for j in self.getNodes("Output"):
                    self.addConnection(i.index, j.index)
        elif (type(param1) == NEAT and type(param2) == NEAT): # If param1 and param2 are both NEAT networks: Do crossover
            # Pick both parents that parentA is fitter than parentB
            if param1.fitness > param2.fitness:
                parentA = param1
                parentB = param2
            else:
                parentA = param2
                parentB = param1

            # First copy all of the connection from the less fit parent so they could later be overriden by the more fit parent
            self.connectionGenes = parentB.connectionGenes.copy()
            
            # Copy over all connections from the fitter parent. If the conection exists overwrite it
            for i in parentA.connectionGenes:
                # 50% of the time copy over a connection to the child if parentB didint have it
                if self.getConnection(i.innov) == None and random.random() >= 0.5:
                    self.connectionGenes.append(i)
            
            # Pick the nodes from the parent that has more of them
            if len(parentA.nodeGenes) > len(parentB.nodeGenes):
                self.nodeGenes = parentA.nodeGenes.copy()
            else:
                self.nodeGenes = parentB.nodeGenes.copy()

            # Find the highest node index from nodes
            for i in self.nodeGenes:
                if i.index > self.nodeIndex:
                    self.nodeIndex = i.index
            # Find the highest innovation number from connections
            for i in self.connectionGenes:
                if i.innov > self.innovation:
                    self.innovation = i.innov
        else:
            # If none of the above applied just exit
            return
        
        self.Mutate()

    def __repr__(self):
        return "NEAT({},{},{})".format(len(self.nodeGenes),len(self.connectionGenes),self.fitness)

    def saveToFile(self,fileName):
        with open(fileName, "w") as f:
            f.write("{};{};{}".format(self.mutationRate,len(self.nodeGenes),len(self.connectionGenes)))
            for i in self.nodeGenes:
                f.write(";{};{}".format(i.index,i.type))

            for i in self.connectionGenes:
                f.write(";{};{};{};{};{}".format(i.input,i.output,i.weight,i.enabled,i.innov))

            f.write("\n")

    def loadFromFile(self, fileName):
        with open(fileName, "r") as f:
            data = f.read()[:-1].split(";")
            self.mutationRate = float(data[0])
            self.nodeIndex = 0
            self.innovation = 0
            
            nodeCount = int(data[1])
            connectionCount = int(data[2])
            
            self.nodeGenes = []
            for i in range(3,3+nodeCount*2,2):
                self.nodeGenes.append(self.NodeGene(int(data[i]),data[i+1]))
                if int(data[i]) > self.nodeIndex:
                    self.nodeIndex = int(data[i])

            self.connectionGenes = []
            for i in range(3+nodeCount*2,(3+nodeCount*2)+connectionCount*5,5):
                conn = self.ConnectionGene(int(data[i]), int(data[i+1]), int(data[i+4]))
                conn.enabled = bool(data[i+3])
                conn.weight = float(data[i+2])
                self.connectionGenes.append(conn)
                if conn.innov > self.innovation:
                    self.innovation = conn.innov

    def Mutate(self):
        if (random.random() <= self.mutationRate):
            for i in self.connectionGenes:
                if (random.random() <= self.adjustWeightMutationChance):
                    # Adjust weights
                    i.weight += random.random()*2-1
                elif (random.random() <= self.nodeEnableMutationChance and i.enabled == False):
                    # Enable a connection
                    i.enabled = True
                elif (random.random() <= self.nodeDisableMutationChance and i.enabled == True):
                    # Disable a connection
                    i.enabled = False

            # Add connection
            if (random.random() <= self.connectionMutationChance):
                # Get an input node index
                inputList = self.getNodes("Input").copy() + self.getNodes("Hidden").copy()
                inputIndex = random.choice(inputList).index
                
                # Get an output node index
                outputList = self.getNodes("Hidden").copy() + self.getNodes("Output").copy()
                for i in outputList:
                    if i.type == "Hidden" and i.index <= inputIndex:
                        outputList.remove(i)
                outputIndex = random.choice(outputList).index

                # Create the connection gene
                self.addConnection(inputIndex, outputIndex)

            # Add node
            if (random.random() <= self.nodeMutationChance):
                # Choose a random connection
                randomConnection = random.choice(self.connectionGenes)
                randomConnection.enabled = False

                # Create a new node
                nodeIndex = self.addNode("Hidden")
                
                # Create a connection from the input node to the new node with a weight of 1
                connectionInnov = self.addConnection(randomConnection.input, nodeIndex)
                self.getConnection(connectionInnov).weight = 1
                
                # Create a connection from the new node to the output connection with the weight of the old connection
                connectionInnov = self.addConnection(nodeIndex, randomConnection.output)
                self.getConnection(connectionInnov).weight = randomConnection.weight

    def forward(self, inputs):
        # I make a memo off all node values so i would not need to recalculate them.
        memo = list(None for _ in self.nodeGenes)
        
        # Calculates the value of that node
        def calculateNode(nodeIndex):
            memoIndex = nodeIndex - 1
            if self.getNode(nodeIndex).type == "Input":
                # This counts on that the inputs where created first and in order!
                return inputs[memoIndex]
            elif self.getNode(nodeIndex).type == "Bias":
                return 1
            
            # If I already calculated this value return it
            if memo[memoIndex] != None:
                return memo[memoIndex]

            value = 0
            for i in self.connectionGenes:
                if i.output == nodeIndex:
                    value += i.weight * calculateNode(i.input)

            # Save the calculated value for future use
            memo[memoIndex] = self.activation(value)
            return memo[memoIndex]
                        
        # Calculate the outputs
        outputNodes = self.getNodes("Output")
        outputs = []
        for i in range(len(outputNodes)):
            outputs.append(calculateNode(outputNodes[i].index))

        # Return the answer
        return outputs        

    def getConnection(self, innov):
        for i in self.connectionGenes:
            if i.innov == innov:
                return i
        return None

    def getNode(self, index):
        for i in self.nodeGenes:
            if i.index == index:
                return i

    def addConnection(self, connectionInput, connectionOutput):
        self.innovation += 1
        connection = self.ConnectionGene(connectionInput, connectionOutput, self.innovation)
        self.connectionGenes.append(connection)
        return self.innovation

    def addNode(self, nodeType = "Hidden"):
        self.nodeIndex += 1
        node = self.NodeGene(self.nodeIndex, nodeType)
        self.nodeGenes.append(node)
        return self.nodeIndex

    def getNodes(self, nodeType):
        nodeList = []
        for i in self.nodeGenes:
            if i.type == nodeType:
                nodeList.append(i)
        return nodeList

    def printNet(self):
        print("Fitness: {}".format(self.fitness))
        print("Nodes({}): {}".format(len(self.nodeGenes),self.nodeGenes))
        print("Connections({}): {}".format(len(self.connectionGenes),self.connectionGenes))

    def genImage(self,imageName):
        layerGap = 80
        nodeGap = 60
        nodeRadius = 20
        padding = 40
        bgColor = (230,230,230)
        
        hiddenColor = (50 , 150, 30 )
        biasColor =   (201, 50 , 50 )
        inputColor =  (27 , 226, 226)
        outputColor = (12 , 120, 220)

        fontColor = (5,5,5)
        fontScale = 0.6
        fontThickness = 2

        connectionColorEnabled = (0,0,150)
        connectionColorDisabled = (0,0,100)
        connectionThickness = 2
        connectionArrowSize = 0.08

        # Make a connection dictionary for the algorithm
        connectionDict = {}
        for node in self.nodeGenes:
            connectionDict[node.index] = []
        for conn in self.connectionGenes:
            connectionDict[conn.input].append(conn.output)
        
        # Make a enabled connection function for drawing
        def getEnabled(connInput, connOutput):
            for i in self.connectionGenes:
                if i.input == connInput and i.output == connOutput:
                    return i.enabled
            return None

        # The algorithm to sort all nodes into layers (SUCH A PAIN)
        layers = list()
        currentLayer = list(i.index for i in self.getNodes("Input") + self.getNodes("Bias"))
        nextLayer = []
        outputLayer = list(i.index for i in self.getNodes("Output"))
        while len(currentLayer) > 0:
            for node in currentLayer:
                for conn in connectionDict[node]:
                    if not(conn in nextLayer) and not(conn in outputLayer):
                        nextLayer.append(conn)
                
                for prevLayer in layers:
                    for conn in currentLayer:
                        if conn in prevLayer:
                            prevLayer.remove(conn)
            
            layers.append(currentLayer.copy())
            currentLayer = nextLayer.copy()
            nextLayer = []

        layers.append(outputLayer)

        # Find the widest part in the neural net
        widestLayerLength = len(layers[0])
        for layer in layers:
            if (len(layer) > widestLayerLength):
                widestLayerLength = len(layer)

        # Calculate the image height and width that will be needed to fit the neural net
        width = padding*2 + (widestLayerLength-1)*nodeGap 
        height = padding*2 + (len(layers)-1)*layerGap

        # Create a blank canvas
        img = np.array(bgColor*width*height,np.uint8)
        img = img.reshape(height,width,3)

        # Calculate point positions
        nodePoints = {}
        for i in range(len(layers)):
            for j in range(len(layers[i])):
                x = int((width-(len(layers[i])-1)*nodeGap)/2 + nodeGap * j)
                y = height - padding + -i*layerGap
                nodePoints[layers[i][j]] = (x, y)

        # Draw all of the connection arrows
        for key, value in connectionDict.items():
            for i in value:
                connectionColor = connectionColorEnabled
                if not getEnabled(key, i):
                    connectionColor = connectionColorDisabled
                 
                direction = [nodePoints[i][0] - nodePoints[key][0], nodePoints[i][1] - nodePoints[key][1]]
                angle = np.arctan2(direction[0], direction[1])
                xOffset = int(np.sin(angle) * nodeRadius)
                yOffset = int(np.cos(angle) * nodeRadius)
                pt1 = (nodePoints[key][0]+xOffset, nodePoints[key][1]+yOffset)
                pt2 = (nodePoints[i][0]-xOffset, nodePoints[i][1]-yOffset)
                cv2.arrowedLine(img, pt1, pt2, connectionColor, connectionThickness, 8, 0, connectionArrowSize)

        # Plot the nodes
        for i in range(len(layers)):
            for j in range(len(layers[i])):
                nodeType = self.getNode(layers[i][j]).type
                nodeColor = hiddenColor

                if nodeType == "Input":
                    nodeColor = inputColor
                elif nodeType == "Output":
                    nodeColor = outputColor
                elif nodeType == "Bias":
                    nodeColor = biasColor
                    
                cv2.circle(img, nodePoints[layers[i][j]], nodeRadius, nodeColor, -1)

                size, _ = cv2.getTextSize(str(layers[i][j]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
                cv2.putText(img, str(layers[i][j]), (nodePoints[layers[i][j]][0]-size[0]//2, nodePoints[layers[i][j]][1]+size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness)

        cv2.imwrite(imageName+".png",img)


if __name__ == "__main__":
    print("Module neural net loaded as main")
    
